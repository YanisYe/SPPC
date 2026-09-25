"""Official torch-harmonics neural architecture and raw physics-head tensors."""
from __future__ import annotations

import math

import torch
from torch import nn
from .physics import GridAdapter

from torch_harmonics.examples.models._layers import SpectralPositionEmbedding
from torch_harmonics.examples.models.s2transformer import (
    DiscreteContinuousDecoder,
    DiscreteContinuousEncoder,
    SphericalAttentionBlock,
)


def _zero_init(module: nn.Module) -> nn.Module:
    final = module[-1]
    nn.init.zeros_(final.weight)
    if final.bias is not None:
        nn.init.zeros_(final.bias)
    return module



class _CellHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = _zero_init(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        )

    @property
    def final_linear(self) -> nn.Module:
        return self.net[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _GlobalHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = _zero_init(
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.GELU(),
                nn.Linear(in_channels, out_channels),
            )
        )

    @property
    def final_linear(self) -> nn.Module:
        return self.net[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.mean(dim=(-2, -1)))


class _EdgeHead(nn.Module):
    """Predict canonical east and north edge logits; incidence is physics code."""

    def __init__(self, feature_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        # endpoints, difference, two 3D positions, and a 3D edge direction
        edge_channels = 3 * feature_channels + 9
        self.net = _zero_init(
            nn.Sequential(
                nn.Conv2d(edge_channels, feature_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(feature_channels, out_channels, kernel_size=1),
            )
        )
        lat = torch.linspace(-math.pi / 2, math.pi / 2, 32)
        lon = torch.arange(64) * (2 * math.pi / 64)
        phi, lam = torch.meshgrid(lat, lon, indexing="ij")
        xyz = torch.stack(
            (torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam), torch.sin(phi))
        )
        self.register_buffer("xyz", xyz, persistent=False)

    @property
    def final_linear(self) -> nn.Module:
        return self.net[-1]

    def _edge_features(
        self, first: torch.Tensor, second: torch.Tensor, xyz_first: torch.Tensor, xyz_second: torch.Tensor
    ) -> torch.Tensor:
        batch = first.shape[0]
        first_xyz = xyz_first.unsqueeze(0).expand(batch, -1, -1, -1)
        second_xyz = xyz_second.unsqueeze(0).expand(batch, -1, -1, -1)
        direction = second_xyz - first_xyz
        return torch.cat((first, second, second - first, first_xyz, second_xyz, direction), dim=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        east_second = features.roll(shifts=-1, dims=-1)
        east_xyz = self.xyz.roll(shifts=-1, dims=-1)
        east = self.net(self._edge_features(features, east_second, self.xyz, east_xyz))
        north = self.net(
            self._edge_features(features[..., :-1, :], features[..., 1:, :], self.xyz[..., :-1, :], self.xyz[..., 1:, :])
        )
        return east, north


class SphericalBackbone(nn.Module):
    """Fixed 32x64 spherical backbone; physics is implemented separately."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 14
        self.embed_dim = 256
        self.decoder_out_channels = 128
        self.attention_pattern = ("neighborhood", "global") * 4
        self.drop_path_rates = tuple(index / 70 for index in range(8))
        internal_shape = (16, 32)
        internal_grid = "legendre-gauss"
        era_lat = torch.linspace(-87.1875, 87.1875, 32).deg2rad()
        self.grid_adapter = GridAdapter(era_lat, 64)

        self.encoder = DiscreteContinuousEncoder(
            in_shape=(32, 64),
            out_shape=internal_shape,
            grid_in="equiangular",
            grid_out=internal_grid,
            in_chans=self.in_channels,
            out_chans=self.embed_dim,
        )
        self.position_embedding = SpectralPositionEmbedding(
            internal_shape, grid=internal_grid, num_chans=self.embed_dim
        )
        self.processor = nn.ModuleList(
            [
                SphericalAttentionBlock(
                    in_shape=internal_shape,
                    out_shape=internal_shape,
                    grid_in=internal_grid,
                    grid_out=internal_grid,
                    in_chans=self.embed_dim,
                    out_chans=self.embed_dim,
                    num_heads=8,
                    mlp_ratio=2.0,
                    drop_rate=0.0,
                    drop_path=self.drop_path_rates[index],
                    norm_layer="layer_norm",
                    attention_mode=mode,
                )
                for index, mode in enumerate(self.attention_pattern)
            ]
        )
        self.decoder = DiscreteContinuousDecoder(
            in_shape=internal_shape,
            out_shape=(32, 64),
            grid_in=internal_grid,
            grid_out="equiangular",
            in_chans=self.embed_dim,
            out_chans=self.decoder_out_channels,
        )
        self.wind_head = _CellHead(self.decoder_out_channels, 4)
        self.surface_residual_head = _CellHead(self.decoder_out_channels, 2)
        self.edge_head = _EdgeHead(self.decoder_out_channels, 3)
        self.global_source_head = _GlobalHead(self.decoder_out_channels, 3)
        self.output_head_names = {
            "wind_head",
            "surface_residual_head",
            "edge_head",
            "global_source_head",
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (14, 32, 64):
            raise ValueError("inputs must have shape [B, 14, 32, 64]")
        official = self.grid_adapter.scalar_to_official(inputs)
        latent = self.position_embedding(self.encoder(official))
        for block in self.processor:
            latent = block(latent)
        return self.grid_adapter.scalar_to_era(self.decoder(latent))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(inputs)
        output = {
            "features": features,
            "raw_wind": self.wind_head(features),
            "raw_surface_residual": self.surface_residual_head(features),
        }
        raw_edge_east, raw_edge_north = self.edge_head(features)
        output.update(
            raw_edge_east=raw_edge_east,
            raw_edge_north=raw_edge_north,
            raw_global_source=self.global_source_head(features),
        )
        return output
