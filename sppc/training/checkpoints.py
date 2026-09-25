"""Final corrector phase budgets and status records."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


PHASE_NAMES = ("phase_a", "phase_b")


def configured_epochs(config: Mapping[str, Any], phase: str) -> int:
    """Return a positive configured epoch budget for one training phase."""
    if phase not in PHASE_NAMES:
        raise ValueError(f"unknown phase {phase!r}")
    epochs = config.get(phase, {}).get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"{phase}.epochs must be a positive integer")
    return epochs


def phase_status_payload(
    config: Mapping[str, Any],
    phase: str,
    *,
    status: str,
    completed_epoch: int,
    effective_batch: int,
    records: Sequence[Mapping[str, Any]],
    config_path: str | Path,
    selected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detailed, machine-readable per-phase status document."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "phase": phase,
        "status": status,
        "budget_epochs": configured_epochs(config, phase),
        "completed_epoch": int(completed_epoch),
        "effective_batch": int(effective_batch),
        "config": str(config_path),
        "records": list(records),
    }
    if selected is not None:
        payload["selected"] = dict(selected)
    return payload


def phase_a_checkpoint_path(config: Mapping[str, Any], root: str | Path) -> Path:
    """Resolve Phase A provenance: local output unless an explicit source is set."""
    outputs = config["outputs"]
    path = outputs.get("phase_a_source_checkpoint")
    if path is None:
        path = Path(outputs["run_dir"]) / outputs.get("phase_a_checkpoint", "phase_a_best_ema.pt")
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path
