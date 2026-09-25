"""Differentiable physical operators on a latitude-longitude sphere."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_harmonics import InverseRealVectorSHT, RealVectorSHT
from torch_harmonics.quadrature import precompute_latitudes


class GridAdapter(nn.Module):
    """Differentiable ERA5 cell-centre ↔ official equiangular grid adapter."""
    def __init__(self, era_latitudes: Tensor, nlon: int = 64):
        super().__init__()
        era = torch.as_tensor(era_latitudes, dtype=torch.float32)
        theta, _ = precompute_latitudes(len(era), "equiangular")
        official = (math.pi / 2 - theta).float()  # north to south, includes poles
        self.register_buffer("era_latitudes", era)
        self.register_buffer("official_latitudes", official)
        self.nlon = nlon

    def _scalar(self, value: Tensor, source: Tensor, target: Tensor, pole_mean: bool) -> Tensor:
        ascending, data = (source, value) if source[0] < source[-1] else (source.flip(0), value.flip(-2))
        row = torch.searchsorted(ascending, target).clamp(1, len(ascending)-1)
        lo, hi = row-1, row
        f = ((target-ascending[lo])/(ascending[hi]-ascending[lo])).to(value.dtype)
        out = data[...,lo,:]*(1-f[...,None]) + data[...,hi,:]*f[...,None]
        if pole_mean:
            for i, latitude in enumerate(target):
                if abs(float(latitude)) >= math.pi/2-1e-7:
                    nearest = data[..., -1 if latitude > 0 else 0, :].mean(-1,keepdim=True)
                    out[...,i,:] = nearest
        return out

    def scalar_to_official(self, value: Tensor) -> Tensor:
        return self._scalar(value, self.era_latitudes, self.official_latitudes, True)

    def scalar_to_era(self, value: Tensor) -> Tensor:
        return self._scalar(value, self.official_latitudes, self.era_latitudes, False)

    def _vector(self, wind: Tensor, source: Tensor, target: Tensor) -> Tensor:
        lon = torch.arange(self.nlon,device=wind.device,dtype=wind.dtype)*(2*math.pi/self.nlon)
        phi = source.to(wind.device,dtype=wind.dtype)[:,None]
        east = torch.stack((-lon.sin().expand_as(phi+lon), lon.cos().expand_as(phi+lon), torch.zeros_like(phi+lon)),-3)
        north = torch.stack((-phi.sin()*lon.cos(),-phi.sin()*lon.sin(),phi.cos().expand_as(phi+lon)),-3)
        xyz = wind[...,0:1,:,:]*east + wind[...,1:2,:,:]*north
        xyz_i = self._scalar(xyz, source, target, False)
        phi_t = target.to(wind.device,dtype=wind.dtype)[:,None]
        east_t = torch.stack((-lon.sin().expand_as(phi_t+lon),lon.cos().expand_as(phi_t+lon),torch.zeros_like(phi_t+lon)),-3)
        north_t = torch.stack((-phi_t.sin()*lon.cos(),-phi_t.sin()*lon.sin(),phi_t.cos().expand_as(phi_t+lon)),-3)
        return torch.stack(((xyz_i*east_t).sum(-3),(xyz_i*north_t).sum(-3)),-3)

    def vector_to_official(self, wind: Tensor) -> Tensor:
        return self._vector(wind,self.era_latitudes,self.official_latitudes)

    def vector_to_era(self, wind: Tensor) -> Tensor:
        return self._vector(wind,self.official_latitudes,self.era_latitudes)


def unit_sphere_cell_areas(latitudes: Tensor, nlon: int) -> Tensor:
    """Return exact unit-sphere cell areas for latitude cell centers.

    Latitude faces are midpoints between adjacent centers; the exterior faces
    are the geographic poles.  Longitude cells are uniform and periodic.
    """
    latitudes = torch.as_tensor(latitudes)
    if latitudes.ndim != 1 or latitudes.numel() < 2:
        raise ValueError("latitudes must be a one-dimensional tensor with at least two centers")
    if nlon <= 0:
        raise ValueError("nlon must be positive")
    if not bool(torch.all(latitudes[1:] > latitudes[:-1])):
        raise ValueError("latitudes must be strictly increasing (south to north)")
    if bool((latitudes < -math.pi / 2).any() or (latitudes > math.pi / 2).any()):
        raise ValueError("latitudes must lie in [-pi/2, pi/2]")

    faces = torch.empty(latitudes.numel() + 1, dtype=latitudes.dtype, device=latitudes.device)
    faces[0] = -math.pi / 2
    faces[-1] = math.pi / 2
    faces[1:-1] = 0.5 * (latitudes[:-1] + latitudes[1:])
    bands = (2.0 * math.pi / nlon) * (faces[1:].sin() - faces[:-1].sin())
    return bands[:, None].expand(-1, nlon).clone()


class SemiLagrangianTransport(nn.Module):
    """Periodic two-substep transport using Cartesian great-circle backtraces."""

    def __init__(self, latitudes: Tensor, nlon: int, dt: float = 21600.0, radius: float = 6371229.0):
        super().__init__()
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        if lat.ndim != 1 or lat.numel() < 2:
            raise ValueError("latitudes must contain at least two centers")
        lon = torch.arange(nlon, dtype=torch.float32) * (2.0 * math.pi / nlon)
        phi, lam = torch.meshgrid(lat, lon, indexing="ij")
        cos_phi = phi.cos()
        position = torch.stack((cos_phi * lam.cos(), cos_phi * lam.sin(), phi.sin()), dim=-1)
        east = torch.stack((-lam.sin(), lam.cos(), torch.zeros_like(lam)), dim=-1)
        north = torch.stack((-phi.sin() * lam.cos(), -phi.sin() * lam.sin(), cos_phi), dim=-1)
        self.register_buffer("latitudes", lat)
        self.register_buffer("areas", unit_sphere_cell_areas(lat.double(), nlon).float())
        self.register_buffer("position", position)
        self.register_buffer("east", east)
        self.register_buffer("north", north)
        self.nlon = int(nlon)
        self.substep = float(dt) / 2.0
        self.radius = float(radius)

    @staticmethod
    def _as_batched(field: Tensor) -> tuple[Tensor, bool]:
        if field.ndim == 3:
            return field.unsqueeze(0), True
        if field.ndim == 4:
            return field, False
        raise ValueError("field must have shape [C,H,W] or [B,C,H,W]")

    @staticmethod
    def _wind(wind: Tensor, batch: int, height: int, width: int) -> Tensor:
        if wind.ndim == 2:
            wind = wind[None, None]
        elif wind.ndim == 3:
            wind = wind[:, None]
        elif wind.ndim != 4:
            raise ValueError("wind must have shape [H,W], [B,H,W], or [B,1,H,W]")
        if wind.shape[0] == 1 and batch != 1:
            wind = wind.expand(batch, -1, -1, -1)
        if wind.shape != (batch, 1, height, width):
            raise ValueError("wind and field grids/batches do not match")
        return wind

    def _sample_periodic(self, field: Tensor, phi: Tensor, lam: Tensor) -> Tensor:
        # One wrapped column on either side makes grid_sample bilinear at 0/2pi.
        padded = torch.cat((field[..., -1:], field, field[..., :1]), dim=-1)
        x_index = torch.remainder(lam, 2.0 * math.pi) * (self.nlon / (2.0 * math.pi)) + 1.0
        x = 2.0 * x_index / (self.nlon + 1.0) - 1.0
        lat = self.latitudes.to(device=field.device, dtype=field.dtype)
        row = torch.searchsorted(lat, phi.contiguous()).clamp(1, lat.numel() - 1)
        lo = row - 1
        frac = (phi - lat[lo]) / (lat[row] - lat[lo])
        y_index = lo.to(field.dtype) + frac
        y = 2.0 * y_index / (lat.numel() - 1.0) - 1.0
        grid = torch.stack((x, y), dim=-1)
        return F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _backtrace(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        dtype, device = u.dtype, u.device
        r = self.position.to(device=device, dtype=dtype).unsqueeze(0)
        tangent = u[..., None] * self.east.to(device=device, dtype=dtype) + v[..., None] * self.north.to(device=device, dtype=dtype)
        speed = torch.linalg.vector_norm(tangent, dim=-1)
        alpha = speed * (self.substep / self.radius)
        direction = tangent / speed.clamp_min(torch.finfo(dtype).tiny)[..., None]
        departure = r * alpha.cos()[..., None] - direction * alpha.sin()[..., None]
        departure = torch.where((speed == 0)[..., None], r, departure)
        departure = departure / torch.linalg.vector_norm(departure, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        # asin(z) has an infinite derivative when rounding/clamping puts a
        # backtrace exactly on a pole.  Longitude is undefined there too.
        # atan2(z, rho) is equivalent away from the poles; safe longitude
        # inputs make the arbitrary polar longitude constant in backward.
        x, y, z = departure.unbind(-1)
        epsilon = torch.finfo(dtype).eps
        rho2 = x.square() + y.square()
        pole = rho2 <= epsilon * epsilon
        rho = rho2.clamp_min(epsilon * epsilon).sqrt()
        phi = torch.atan2(z, rho)
        safe_x = torch.where(pole, torch.ones_like(x), x)
        safe_y = torch.where(pole, torch.zeros_like(y), y)
        lam = torch.atan2(safe_y, safe_x)
        return phi, lam

    def forward(self, field: Tensor, u: Tensor, v: Tensor) -> Tensor:
        field, squeezed = self._as_batched(field.float())
        batch, _, height, width = field.shape
        if (height, width) != (self.latitudes.numel(), self.nlon):
            raise ValueError("field grid does not match the configured grid")
        u = self._wind(u.float(), batch, height, width)[:, 0]
        v = self._wind(v.float(), batch, height, width)[:, 0]
        if not bool(torch.any(u != 0) or torch.any(v != 0)):
            return field.squeeze(0) if squeezed else field
        phi, lam = self._backtrace(u, v)
        output = self._sample_periodic(field, phi, lam)
        output = self._sample_periodic(output, phi, lam)
        area = self.areas.to(device=field.device, dtype=field.dtype)
        denom = area.sum()
        input_mean = (field * area).sum(dim=(-2, -1), keepdim=True) / denom
        output_mean = (output * area).sum(dim=(-2, -1), keepdim=True) / denom
        output = output - output_mean + input_mean
        return output.squeeze(0) if squeezed else output


class SharedEdgeDivergence(nn.Module):
    """Map one canonical transfer per shared edge to conservative cell tendencies.

    ``east[..., i, j]`` is oriented from cell ``(i,j)`` to ``(i,j+1 mod W)``.
    ``north[..., i, j]`` is oriented from ``(i,j)`` to ``(i+1,j)``; there are no
    transfers through the two exterior latitude boundaries.
    """

    def __init__(self, areas: Tensor):
        super().__init__()
        areas = torch.as_tensor(areas, dtype=torch.float32)
        if areas.ndim != 2 or bool((areas <= 0).any()):
            raise ValueError("areas must be a positive [H,W] tensor")
        self.register_buffer("areas", areas)

    def forward(self, east: Tensor, north: Tensor) -> Tensor:
        if east.ndim < 3:
            raise ValueError("east must end in [K,H,W]")
        height, width = self.areas.shape
        if east.shape[-3:] != (3, height, width):
            raise ValueError("east must have exactly three scalar channels and shape [...,3,H,W]")
        if north.shape != east.shape[:-2] + (height - 1, width):
            raise ValueError("north must have shape [...,3,H-1,W]")
        net = east.roll(1, dims=-1) - east
        net = net.clone()
        net[..., :-1, :] -= north
        net[..., 1:, :] += north
        return net / self.areas.to(device=net.device, dtype=net.dtype)


class HodgeToroidalProjection(nn.Module):
    """Project official equiangular [U east,V north] wind onto toroidal modes."""

    def __init__(self, nlat: int = 32, nlon: int = 64, lmax: int = 16, mmax: int = 16):
        super().__init__()
        self.sht = RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax,
                                 grid="equiangular", csphase=False)
        self.isht = InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax,
                                         grid="equiangular", csphase=False)

    @staticmethod
    def _spherical(wind: Tensor) -> Tensor:
        if wind.shape[-3] != 2:
            raise ValueError("wind must end in [2,H,W] ordered as [U,V]")
        return torch.stack((-wind[..., 1, :, :], wind[..., 0, :, :]), dim=-3)

    @staticmethod
    def _era5(vector: Tensor) -> Tensor:
        return torch.stack((vector[..., 1, :, :], -vector[..., 0, :, :]), dim=-3)

    def forward(self, wind: Tensor) -> Tensor:
        original_dtype = wind.dtype
        vector = self._spherical(wind.float())
        coefficients = self.sht(vector)
        coefficients[..., 0, :, :] = 0
        return self._era5(self.isht(coefficients)).to(original_dtype)

    def spheroidal_ratio(self, wind: Tensor) -> float:
        coefficients = self.sht(self._spherical(wind.float()))
        spheroidal = coefficients[..., 0, :, :].abs().square().sum().sqrt()
        total = coefficients.abs().square().sum().sqrt().clamp_min(1e-30)
        return float((spheroidal / total).detach().cpu())


class GeostrophicWind(nn.Module):
    """Regularized spherical geostrophic baseline from Z500 geopotential."""

    def __init__(
        self,
        latitudes: Tensor,
        nlon: int,
        radius: float = 6371229.0,
        omega: float = 7.292115e-5,
        equatorial_width_degrees: float = 20.0,
    ):
        super().__init__()
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        if lat.ndim != 1 or lat.numel() < 3:
            raise ValueError("latitudes must contain at least three centers")
        self.register_buffer("latitudes", lat)
        self.nlon = int(nlon)
        self.radius = float(radius)
        self.omega = float(omega)
        self.sin_phi0 = math.sin(math.radians(equatorial_width_degrees))

    def forward(self, geopotential: Tensor, *, is_height: bool = False) -> tuple[Tensor, Tensor]:
        phi = geopotential.float()
        if phi.shape[-2:] != (self.latitudes.numel(), self.nlon):
            raise ValueError("geopotential grid does not match the configured grid")
        if is_height:
            phi = phi * 9.80665
        lat = self.latitudes.to(device=phi.device, dtype=phi.dtype)

        dphi = torch.empty_like(phi)
        gaps = lat[2:] - lat[:-2]
        dphi[..., 1:-1, :] = (phi[..., 2:, :] - phi[..., :-2, :]) / gaps[:, None]
        dphi[..., 0, :] = (phi[..., 1, :] - phi[..., 0, :]) / (lat[1] - lat[0])
        dphi[..., -1, :] = (phi[..., -1, :] - phi[..., -2, :]) / (lat[-1] - lat[-2])
        dlambda = (phi.roll(-1, dims=-1) - phi.roll(1, dims=-1)) / (4.0 * math.pi / self.nlon)

        sin_lat = lat.sin()
        chi = torch.clamp(sin_lat.abs() / self.sin_phi0, max=1.0)
        sign = torch.where(lat >= 0, 1.0, -1.0)
        f_regularized = 2.0 * self.omega * sign * sin_lat.abs().clamp_min(self.sin_phi0)
        scale = chi / (self.radius * f_regularized)
        u = -scale[:, None] * dphi
        v = scale[:, None] * dlambda / lat.cos().abs().clamp_min(1e-6)[:, None]
        equator = lat.abs() <= 8.0 * torch.finfo(lat.dtype).eps
        if bool(equator.any()):
            u = u.masked_fill(equator[:, None], 0.0)
            v = v.masked_fill(equator[:, None], 0.0)
        return u, v
