"""Exact v2 Phase-A/Phase-B optimization infrastructure for correctors."""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .common import optimizer_parameter_groups
from ..models.input import make_model_input


@dataclass(frozen=True)
class PhaseAProtocol:
    epochs: int = 15
    effective_global_batch_size: int = 128
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    minimum_learning_rate: float = 3e-6
    gradient_clip_norm: float = 1.0
    precision: str = "bf16"
    ema_decay: float = 0.999


@dataclass(frozen=True)
class PhaseBProtocol:
    rollout_steps: int = 4
    epochs: int = 15
    effective_global_batch_size: int = 108
    corrector_learning_rate: float = 3e-5
    predictor_tail_learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    minimum_lr_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    precision: str = "bf16"
    ema_decay: float = 0.999
    lead_weights: tuple[float, float, float, float] = (0.3, 0.2, 0.2, 0.3)
    channel_weights: tuple[float, float, float, float, float] = (1., 1., 1., 2., 2.)


def accumulation_plan(target_effective_batch: int, micro_batch: int, world_size: int = 1) -> tuple[int, int]:
    """Largest realizable effective batch no greater than the protocol target."""
    if min(target_effective_batch, micro_batch, world_size) < 1:
        raise ValueError("batch dimensions must be positive")
    per_update = micro_batch * world_size
    accumulation = target_effective_batch // per_update
    if accumulation < 1:
        raise ValueError("micro_batch * world_size exceeds target effective batch")
    return accumulation, accumulation * per_update


def gradients_are_finite(parameters: Iterable[nn.Parameter]) -> bool:
    """Return false before an optimizer can consume any NaN/Inf gradient."""
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def correction_regularizer(terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
    required = {"scalar_zero", "scalar_mean", "wind"}
    if set(terms) != required:
        raise ValueError(f"regularizer terms must be exactly {sorted(required)}")
    return (terms["scalar_zero"] + terms["scalar_mean"] + terms["wind"]) / 5.0


def weather_mse(prediction: torch.Tensor, target: torch.Tensor, area: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    residual = (prediction.float() - target.float()) / torch.as_tensor(
        scale, device=prediction.device, dtype=torch.float32
    )[None, :, None, None].clamp_min(1e-6)
    weights = torch.as_tensor(area, device=prediction.device, dtype=torch.float32)
    return (residual.square() * weights).sum((-2, -1)).div(weights.sum()).mean()


def weighted_normalized_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    area: torch.Tensor,
    scale: torch.Tensor,
    channel_weights: Iterable[float] = PhaseBProtocol().channel_weights,
) -> torch.Tensor:
    alpha = torch.as_tensor(tuple(channel_weights), device=prediction.device, dtype=torch.float32)
    if alpha.shape != (5,) or tuple(alpha.tolist()) != PhaseBProtocol().channel_weights:
        raise ValueError("channel order must be [T2m,T850,Z500,U10,V10] with [1,1,1,2,2]")
    residual = (prediction.float() - target.float()).abs() / torch.as_tensor(
        scale, device=prediction.device, dtype=torch.float32
    )[None, :, None, None].clamp_min(1e-6)
    area_weights = torch.as_tensor(area, device=prediction.device, dtype=torch.float32)
    per_channel = (residual * area_weights).sum((-2, -1)).div(area_weights.sum())
    return (per_channel * alpha[None]).sum(1).div(alpha.sum()).mean()


def phase_a_objective(
    final: torch.Tensor,
    target: torch.Tensor,
    terms: Mapping[str, torch.Tensor],
    area: torch.Tensor,
    std_dx: torch.Tensor,
) -> torch.Tensor:
    return weather_mse(final, target, area, std_dx) + 0.01 * correction_regularizer(terms)


def _predictor_model(predictor: nn.Module) -> nn.Module:
    return getattr(predictor, "predictor", predictor)


def _neural_model(predictor: nn.Module) -> nn.Module:
    model = _predictor_model(predictor)
    return getattr(model, "neural", model)


def set_phase_a_trainable(predictor: nn.Module, corrector: nn.Module) -> None:
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    corrector.train()
    for parameter in corrector.parameters():
        parameter.requires_grad_(True)


def set_phase_b_trainable(predictor: nn.Module, corrector: nn.Module) -> None:
    """Freeze encoder/first six blocks; train last2, decoder, original heads, corrector."""
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    neural = _neural_model(predictor)
    if not hasattr(neural, "processor") or len(neural.processor) != 8:
        raise ValueError("Stage-1 must expose exactly eight processor blocks")
    train_modules = list(neural.processor[-2:]) + [neural.decoder]
    head_names = getattr(neural, "output_head_names", None)
    if head_names is None:
        head_names = {
            name for name in ("wind_head", "surface_residual_head", "edge_head", "global_source_head", "scalar_residual_head")
            if hasattr(neural, name)
        }
    if not head_names:
        raise ValueError("Stage-1 physics output heads are unavailable")
    train_modules.extend(getattr(neural, name) for name in head_names)
    for module in train_modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for module in neural.modules():
        if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            module.eval()
    corrector.train()
    for parameter in corrector.parameters():
        parameter.requires_grad_(True)


def _groups_with_lr(model: nn.Module, weight_decay: float, lr: float) -> list[dict[str, Any]]:
    groups = optimizer_parameter_groups(model, weight_decay)
    for group in groups:
        group["lr"] = lr
    return groups


def build_phase_b_optimizer(
    predictor: nn.Module,
    corrector: nn.Module,
    *,
    device: str | torch.device,
    protocol: PhaseBProtocol = PhaseBProtocol(),
) -> torch.optim.AdamW:
    predictor_groups = _groups_with_lr(predictor, protocol.weight_decay, protocol.predictor_tail_learning_rate)
    corrector_groups = _groups_with_lr(corrector, protocol.weight_decay, protocol.corrector_learning_rate)
    for group in predictor_groups + corrector_groups:
        group["initial_lr"] = float(group["lr"])
    use_fused = torch.device(device).type == "cuda"
    if use_fused and not torch.cuda.is_available():
        raise RuntimeError("fused AdamW requires CUDA")
    optimizer = torch.optim.AdamW(
        predictor_groups + corrector_groups,
        betas=protocol.betas,
        eps=protocol.eps,
        fused=use_fused,
    )
    return optimizer


def phase_b_rollout_objective(
    predictor: nn.Module,
    corrector: nn.Module,
    initial_x: torch.Tensor,
    targets: torch.Tensor,
    current_climatologies: torch.Tensor,
    target_climatologies: torch.Tensor,
    calendars: torch.Tensor,
    *,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    std_dx: torch.Tensor,
    area: torch.Tensor,
    protocol: PhaseBProtocol = PhaseBProtocol(),
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if targets.ndim != 5 or targets.shape[1] != protocol.rollout_steps:
        raise ValueError("Phase B targets must have shape [B,4,5,H,W]")
    state = initial_x
    records: list[dict[str, Any]] = []
    weighted_terms: list[torch.Tensor] = []
    for lead in range(protocol.rollout_steps):
        model_input = make_model_input(
            state, current_climatologies[:, lead], state_mean, state_std, calendars[:, lead]
        )
        base_out = predictor(
            model_input, state, current_climatologies[:, lead], target_climatologies[:, lead]
        )
        base = base_out["base_prediction"]
        corr_out = corrector(state, base, base_out["decoder_features"])
        final = base + corr_out["correction"]
        final_l1 = weighted_normalized_l1(
            final, targets[:, lead], area, std_dx, protocol.channel_weights
        )
        base_mse = weather_mse(base, targets[:, lead], area, std_dx)
        regularizer = correction_regularizer(corr_out["correction_regularizer_terms"])
        term = final_l1 + 0.2 * base_mse + 0.01 * regularizer
        weighted_terms.append(protocol.lead_weights[lead] * term)
        records.append({
            "base": base,
            "correction": corr_out["correction"],
            "final": final,
            "target": targets[:, lead],
            "losses": {
                "final_weighted_l1": final_l1,
                "base_equal_mse": base_mse,
                "correction_regularizer": regularizer,
            },
        })
        state = final  # no detach and no teacher forcing
    return sum(weighted_terms), records


class CorrectorEMA:
    """FP32 EMA over one or more named modules, including buffers."""

    def __init__(self, modules: Mapping[str, nn.Module], decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            f"{prefix}.{name}": value.detach().float().cpu().clone()
            for prefix, module in modules.items()
            for name, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, modules: Mapping[str, nn.Module]) -> None:
        for prefix, module in modules.items():
            for name, value in module.state_dict().items():
                key = f"{prefix}.{name}"
                self.shadow[key].lerp_(value.detach().float().cpu(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name.replace("stage1.", "predictor."):
            value.detach().float().cpu().clone()
            for name, value in state["shadow"].items()
        }

    @contextmanager
    def apply(self, modules: Mapping[str, nn.Module]):
        originals = {
            prefix: {name: value.detach().clone() for name, value in module.state_dict().items()}
            for prefix, module in modules.items()
        }
        try:
            for prefix, module in modules.items():
                state = {
                    name: self.shadow[f"{prefix}.{name}"].to(device=value.device, dtype=value.dtype)
                    for name, value in module.state_dict().items()
                }
                module.load_state_dict(state, strict=True)
            yield modules
        finally:
            for prefix, module in modules.items():
                module.load_state_dict(originals[prefix], strict=True)
