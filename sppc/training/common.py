"""Shared Stage A optimization, checkpoint, and run-state utilities."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def atomic_json_write(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _normalization_parameter_ids(model: nn.Module) -> set[int]:
    normalization_types = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, normalization_types)
        for parameter in module.parameters(recurse=False)
    }


def optimizer_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split AdamW params; norms, biases, and scalar params never decay."""
    norm_ids = _normalization_parameter_ids(model)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = parameter.numel() == 1 or name.endswith(".bias") or id(parameter) in norm_ids
        (no_decay if excluded else decay).append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_adamw(
    model: nn.Module,
    *,
    device: str | torch.device,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    use_fused = torch.device(device).type == "cuda"
    if use_fused and not torch.cuda.is_available():
        raise RuntimeError("fused AdamW requires an available CUDA device")
    return torch.optim.AdamW(
        optimizer_parameter_groups(model, weight_decay),
        lr=lr,
        betas=betas,
        eps=eps,
        fused=use_fused,
    )


def build_optimizer_bundle(model: nn.Module, config: Mapping[str, Any], *, device: str | torch.device):
    """Build the final AdamW optimizer and its parameter audit."""
    if config.get("optimizer", "fused_adamw") != "fused_adamw":
        raise ValueError("SPPC uses fused_adamw")
    opt = build_adamw(model, device=device, lr=float(config["learning_rate"]),
                      betas=tuple(config.get("betas", (0.9, 0.95))),
                      eps=float(config.get("epsilon", 1e-8)),
                      weight_decay=float(config.get("weight_decay", 0.05)))
    return [opt], [{"name": n, "shape": list(p.shape), "group": "adamw", "parameters": p.numel()}
                   for n,p in model.named_parameters() if p.requires_grad]


def learning_rate_for_update(
    update_index: int,
    total_updates: int,
    warmup_updates: int,
    peak_lr: float,
    minimum_lr: float,
) -> float:
    """LR for zero-based update: linear warmup, then endpoint-inclusive cosine."""
    if not (0 <= update_index < total_updates):
        raise ValueError("update_index must lie inside the training budget")
    if not (0 <= warmup_updates < total_updates):
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if warmup_updates and update_index < warmup_updates:
        return peak_lr * (update_index + 1) / warmup_updates
    cosine_updates = total_updates - warmup_updates
    progress = (update_index - warmup_updates + 1) / cosine_updates
    return minimum_lr + 0.5 * (peak_lr - minimum_lr) * (1.0 + math.cos(math.pi * progress))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)


class TopKCheckpointManager:
    """Retain exactly the best available K validation model checkpoints."""

    def __init__(self, directory: str | Path, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.index_path = self.directory / "index.json"
        self.records: list[dict[str, Any]] = (
            json.loads(self.index_path.read_text()) if self.index_path.exists() else []
        )
        self._normalize()

    def _normalize(self) -> None:
        self.records = sorted(self.records, key=lambda record: (float(record["score"]), int(record["epoch"])))[:self.k]
        keep = {record["checkpoint"] for record in self.records}
        for checkpoint in self.directory.glob("epoch_*.pt"):
            if checkpoint.name not in keep:
                checkpoint.unlink()
        atomic_json_write(self.index_path, self.records)

    @torch.inference_mode()
    def consider(
        self,
        *,
        epoch: int,
        score: float,
        model_state: Mapping[str, torch.Tensor],
        metrics: Mapping[str, Any],
    ) -> bool:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("nonfinite validation score")
        if any(int(record["epoch"]) == epoch for record in self.records):
            raise ValueError(f"epoch {epoch} already ranked")
        qualifies = len(self.records) < self.k or score < max(float(record["score"]) for record in self.records)
        if not qualifies:
            return False
        filename = f"epoch_{epoch:04d}.pt"
        atomic_torch_save(
            self.directory / filename,
            {
                "epoch": epoch,
                "score": score,
                "model": {name: value.detach().cpu() for name, value in model_state.items()},
                "validation_metrics": dict(metrics),
                "selection_weight_source": "raw_model",
            },
        )
        self.records.append(
            {
                "epoch": epoch,
                "score": score,
                "checkpoint": filename,
                "selection_weight_source": "raw_model",
                "validation_metrics": dict(metrics),
            }
        )
        self._normalize()
        return any(int(record["epoch"]) == epoch for record in self.records)


def validate_predictor_config(config: Mapping[str, Any]) -> None:
    expected = {
        "batch_size": 128,
        "gradient_accumulation": 1,
        "epochs": 500,
        "optimizer": "fused_adamw",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.05,
        "warmup_epochs": 6,
        "minimum_learning_rate": 3e-6,
        "precision": "bf16",
        "drop_last": False,
        "gradient_clip_norm": 1.0,
        "validate_every_epochs": 1,
        "validation_split": "2016",
        "selection_metric": "mean_5var_normalized_rmse_6h",
        "selection_weight_source": "raw_model",
        "top_k": 3,
        "early_stopping": False,
        "ema": False,
    }
    if config.get("variant") != "full":
        raise ValueError("SPPC only supports the final full SSP variant")
    for key, value in expected.items():
        actual = config.get(key)
        valid = actual in value if isinstance(value, set) else actual == value
        if not valid:
            raise ValueError(f"Stage A config {key!r} must be {value!r}, got {actual!r}")
