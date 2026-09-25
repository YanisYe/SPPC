"""Differentiable construction of the 14-channel SSP input."""
from __future__ import annotations

import math
import torch


def make_model_input(
    current: torch.Tensor,
    current_climatology: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    target_calendar: torch.Tensor,
) -> torch.Tensor:
    """Construct model input while retaining current-state gradients."""
    channels, height, width = current.shape[-3:]
    mean = torch.as_tensor(state_mean, device=current.device, dtype=torch.float32).view(1, channels, 1, 1)
    std = torch.as_tensor(state_std, device=current.device, dtype=torch.float32).clamp_min(1e-6).view(1, channels, 1, 1)
    calendar = target_calendar.to(device=current.device, dtype=torch.float32)
    day, slot = calendar[:, 2], calendar[:, 3]
    phase = torch.stack((
        torch.sin(2 * math.pi * day / 365),
        torch.cos(2 * math.pi * day / 365),
        torch.sin(2 * math.pi * slot / 4),
        torch.cos(2 * math.pi * slot / 4),
    ), dim=1)
    phase = phase[:, :, None, None].expand(-1, -1, height, width)
    climate = current_climatology.float()
    return torch.cat(((current.float() - climate) / std, (climate - mean) / std, phase), dim=1)
