"""Structured residual corrector with exact scalar and wind physics."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch_harmonics import InverseRealVectorSHT, RealVectorSHT

from .physics import GridAdapter, unit_sphere_cell_areas


_GRID_HEIGHT = 32
_GRID_WIDTH = 64
_FEATURE_DIM = 128


def _as_vector(value: Tensor | Any | None, length: int, default: float, name: str) -> Tensor:
    result = torch.full((length,), default, dtype=torch.float32) if value is None else torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.clone()


def _positive_vector(value: Tensor | Any | None, length: int, name: str) -> Tensor:
    result = _as_vector(value, length, 1.0, name)
    if bool((result <= 0).any()):
        raise ValueError(f"{name} must be positive")
    return result.clamp_min(1e-6)


def _default_latitudes() -> Tensor:
    return torch.linspace(-87.1875, 87.1875, _GRID_HEIGHT, dtype=torch.float32).deg2rad()


def _latitudes(value: Tensor | Any | None) -> Tensor:
    result = _default_latitudes() if value is None else torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (_GRID_HEIGHT,):
        raise ValueError("latitudes must have shape [32]")
    if not torch.isfinite(result).all() or not bool(torch.all(result[1:] > result[:-1])):
        raise ValueError("latitudes must be finite and strictly south-to-north")
    if bool((result < -math.pi / 2).any() or (result > math.pi / 2).any()):
        raise ValueError("latitudes must lie in [-pi/2, pi/2]")
    return result.clone()


def _zero_final(module: nn.Sequential) -> nn.Sequential:
    final = module[-1]
    if not isinstance(final, (nn.Linear, nn.Conv2d)):
        raise TypeError("the final head layer must be linear")
    nn.init.zeros_(final.weight)
    if final.bias is not None:
        nn.init.zeros_(final.bias)
    return module


class ChannelLastLayerNorm(nn.Module):
    """Apply LayerNorm over channels independently at every grid cell."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.norm.normalized_shape[0]:
            raise ValueError("inputs must have shape [B,C,H,W] with the configured channel count")
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _CorrectorFeatures(nn.Module):
    """Common exact 15-channel residual adapter."""

    def __init__(
        self,
        *,
        feature_dim: int = _FEATURE_DIM,
        state_mean: Tensor | Any | None = None,
        state_std: Tensor | Any | None = None,
        std_dx: Tensor | Any | None = None,
    ) -> None:
        super().__init__()
        if feature_dim != _FEATURE_DIM:
            raise ValueError("the exact corrector architecture requires feature_dim=128")
        self.feature_dim = feature_dim
        self.state_proj = nn.Sequential(
            nn.Conv2d(15, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
        )
        self.channel_norm = ChannelLastLayerNorm(128)
        self.cell_mlp = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
        )
        self.register_buffer("state_mean", _as_vector(state_mean, 5, 0.0, "state_mean"))
        self.register_buffer("state_std", _positive_vector(state_std, 5, "state_std"))
        self.register_buffer("std_dx", _positive_vector(std_dx, 5, "std_dx"))

    def _features(self, current: Tensor, base: Tensor, decoder_features: Tensor) -> Tensor:
        if current.ndim != 4 or current.shape[1:] != (5, _GRID_HEIGHT, _GRID_WIDTH):
            raise ValueError("current must have shape [B,5,32,64]")
        if base.shape != current.shape:
            raise ValueError("base must have the same shape as current")
        if decoder_features.shape != (current.shape[0], 128, _GRID_HEIGHT, _GRID_WIDTH):
            raise ValueError("decoder_features must have shape [B,128,32,64]")
        mean = self.state_mean[None, :, None, None]
        state_scale = self.state_std[None, :, None, None]
        delta_scale = self.std_dx[None, :, None, None]
        residual_state = torch.cat(
            ((current - mean) / state_scale, (base - mean) / state_scale, (base - current) / delta_scale),
            dim=1,
        )
        projected = self.state_proj(residual_state)
        f0 = decoder_features + projected
        return f0 + 0.1 * self.cell_mlp(self.channel_norm(f0))


class StructuredCorrector(_CorrectorFeatures):
    """Shared-edge scalar correction plus full Hodge wind correction."""

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        *,
        state_mean: Tensor | Any | None = None,
        state_std: Tensor | Any | None = None,
        std_dx: Tensor | Any | None = None,
        std_res_zero: Tensor | Any | None = None,
        std_res_mean: Tensor | Any | None = None,
        std_res_wind: Tensor | Any | None = None,
        latitudes: Tensor | Any | None = None,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim, state_mean=state_mean, state_std=state_std, std_dx=std_dx
        )
        self.edge_mlp = _zero_final(nn.Sequential(
            nn.Linear(396, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        ))
        self.global_mlp = _zero_final(nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        ))
        self.wind_head = _zero_final(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1),
        ))
        self.p_sph = nn.Parameter(torch.zeros(16))
        self.p_tor = nn.Parameter(torch.zeros(16))
        self.p_high = nn.Parameter(torch.zeros(1))

        latitude = _latitudes(latitudes)
        area = unit_sphere_cell_areas(latitude.double(), _GRID_WIDTH).float()
        geometry = self._build_geometry(latitude, area)
        self.register_buffer("latitudes", latitude)
        self.register_buffer("cell_area", area)
        for name, value in geometry.items():
            self.register_buffer(name, value)
        self.register_buffer("std_res_zero", _positive_vector(std_res_zero, 3, "std_res_zero"))
        self.register_buffer("std_res_mean", _positive_vector(std_res_mean, 3, "std_res_mean"))
        self.register_buffer("std_res_wind", _positive_vector(std_res_wind, 2, "std_res_wind"))
        self.grid_adapter = GridAdapter(latitude, _GRID_WIDTH)

        self.vector_sht = RealVectorSHT(
            _GRID_HEIGHT, _GRID_WIDTH, lmax=16, mmax=16, grid="equiangular", csphase=False
        )
        self.inverse_vector_sht = InverseRealVectorSHT(
            _GRID_HEIGHT, _GRID_WIDTH, lmax=16, mmax=16, grid="equiangular", csphase=False
        )

    @staticmethod
    def _build_geometry(latitude: Tensor, area: Tensor) -> dict[str, Tensor]:
        longitude = torch.arange(_GRID_WIDTH, dtype=torch.float32) * (2.0 * math.pi / _GRID_WIDTH)
        phi, lam = torch.meshgrid(latitude, longitude, indexing="ij")
        xyz = torch.stack((phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin()), dim=-1)
        flat_xyz = xyz.reshape(-1, 3)

        cell = torch.arange(_GRID_HEIGHT * _GRID_WIDTH, dtype=torch.long).reshape(_GRID_HEIGHT, _GRID_WIDTH)
        east_first = cell.reshape(-1)
        east_second = cell.roll(-1, dims=1).reshape(-1)
        north_first = cell[:-1].reshape(-1)
        north_second = cell[1:].reshape(-1)
        first = torch.cat((east_first, north_first))
        second = torch.cat((east_second, north_second))
        xyz_first = flat_xyz[first]
        xyz_second = flat_xyz[second]
        dot = (xyz_first * xyz_second).sum(-1).clamp(-1.0, 1.0)
        tangent = xyz_second - dot[:, None] * xyz_first
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
        edge_length = torch.acos(dot).unsqueeze(-1)
        directions = torch.cat((
            torch.tensor([1.0, 0.0]).expand(east_first.numel(), -1),
            torch.tensor([0.0, 1.0]).expand(north_first.numel(), -1),
        ))
        return {
            "xyz": flat_xyz,
            "edge_first": first,
            "edge_second": second,
            "edge_tangent": tangent,
            "edge_length": edge_length / area.mean().sqrt(),
            "edge_direction_onehot": directions,
        }
    @staticmethod
    def _era_to_spherical(wind: Tensor) -> Tensor:
        return torch.stack((-wind[:, 1], wind[:, 0]), dim=1)

    @staticmethod
    def _spherical_to_era(vector: Tensor) -> Tensor:
        return torch.stack((vector[:, 1], -vector[:, 0]), dim=1)

    def _edge_inputs(self, features: Tensor) -> Tensor:
        flattened = features.flatten(2).transpose(1, 2)
        first = flattened[:, self.edge_first]
        second = flattened[:, self.edge_second]
        batch = features.shape[0]
        return torch.cat((
            first,
            second,
            second - first,
            self.xyz[self.edge_first][None].expand(batch, -1, -1),
            self.xyz[self.edge_second][None].expand(batch, -1, -1),
            self.edge_tangent[None].expand(batch, -1, -1),
            self.edge_length[None].expand(batch, -1, -1),
            self.edge_direction_onehot[None].expand(batch, -1, -1),
        ), dim=-1)

    def _scalar_correction(self, features: Tensor) -> tuple[Tensor, Tensor]:
        edge_logits = self.edge_mlp(self._edge_inputs(features))
        flat_area = self.cell_area.reshape(-1)
        first_area = flat_area[self.edge_first]
        second_area = flat_area[self.edge_second]
        harmonic_area = (2.0 * first_area * second_area / (first_area + second_area))[:, None]
        transfer = (
            3.0 * self.std_res_zero[None, None, :] * harmonic_area[None] * torch.tanh(edge_logits)
        )
        net = transfer.new_zeros((features.shape[0], _GRID_HEIGHT * _GRID_WIDTH, 3))
        net.index_add_(1, self.edge_first, -transfer)
        net.index_add_(1, self.edge_second, transfer)
        scalar_edge = net.transpose(1, 2).reshape(features.shape[0], 3, _GRID_HEIGHT, _GRID_WIDTH)
        scalar_edge = scalar_edge / self.cell_area[None, None]

        pooled = (features * self.cell_area[None, None]).sum((-2, -1)) / self.cell_area.sum()
        scalar_global = 3.0 * self.std_res_mean[None] * torch.tanh(self.global_mlp(pooled))
        return scalar_edge, scalar_global

    def _wind_correction(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        raw = 3.0 * self.std_res_wind[None, :, None, None] * torch.tanh(self.wind_head(features))
        raw_official = self.grid_adapter.vector_to_official(raw)
        coefficients = self.vector_sht(self._era_to_spherical(raw_official))
        sph_mask = coefficients.new_tensor([1.0, 0.0])[None, :, None, None]
        tor_mask = coefficients.new_tensor([0.0, 1.0])[None, :, None, None]
        sph_coefficients = coefficients * sph_mask
        tor_coefficients = coefficients * tor_mask
        low_sph_official = self._spherical_to_era(self.inverse_vector_sht(sph_coefficients))
        low_tor_official = self._spherical_to_era(self.inverse_vector_sht(tor_coefficients))
        low_sph = self.grid_adapter.vector_to_era(low_sph_official)
        low_tor = self.grid_adapter.vector_to_era(low_tor_official)
        high = raw - low_sph - low_tor

        sph_gain = 0.5 + torch.sigmoid(self.p_sph)
        tor_gain = 0.5 + torch.sigmoid(self.p_tor)
        high_gain = 0.5 + torch.sigmoid(self.p_high)
        gained_sph = self.grid_adapter.vector_to_era(self._spherical_to_era(
            self.inverse_vector_sht(sph_coefficients * sph_gain[None, None, :, None])
        ))
        gained_tor = self.grid_adapter.vector_to_era(self._spherical_to_era(
            self.inverse_vector_sht(tor_coefficients * tor_gain[None, None, :, None])
        ))
        gained_high = high_gain * high
        wind = gained_sph + gained_tor + gained_high
        gains = {
            "spheroidal": sph_gain,
            "toroidal": tor_gain,
            "high_frequency": high_gain,
        }
        return wind, gained_sph, gained_tor, gained_high, gains

    def _regularizer(self, scalar_edge: Tensor, scalar_global: Tensor, wind: Tensor) -> dict[str, Tensor]:
        normalized_area = self.cell_area / self.cell_area.sum()
        scalar_zero = (
            (scalar_edge / self.std_res_zero[None, :, None, None]).square()
            * normalized_area[None, None]
        ).sum((-2, -1, -3)).mean()
        scalar_mean = (scalar_global / self.std_res_mean[None]).square().sum(-1).mean()
        wind_energy = (
            (wind / self.std_res_wind[None, :, None, None]).square()
            * normalized_area[None, None]
        ).sum((-2, -1, -3)).mean()
        return {"scalar_zero": scalar_zero, "scalar_mean": scalar_mean, "wind": wind_energy}

    def forward(self, current: Tensor, base: Tensor, decoder_features: Tensor) -> dict[str, Any]:
        features = self._features(current, base, decoder_features)
        # Physics and all reductions are deliberately outside mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features_fp32 = features.float()
            scalar_edge, scalar_global = self._scalar_correction(features_fp32)
            wind, wind_sph, wind_tor, wind_high, gains = self._wind_correction(features_fp32)
            correction = torch.cat((scalar_edge + scalar_global[:, :, None, None], wind), dim=1)
            regularizer = self._regularizer(scalar_edge, scalar_global, wind)
        return {
            "correction": correction,
            "scalar_edge": scalar_edge,
            "scalar_global": scalar_global,
            "wind_correction": wind,
            "wind_spheroidal": wind_sph,
            "wind_toroidal": wind_tor,
            "wind_high_frequency": wind_high,
            "hodge_gains": gains,
            "correction_regularizer_terms": regularizer,
        }
