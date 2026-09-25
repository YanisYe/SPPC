"""Validation-only checkpoint selection and test evaluation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .metrics import evaluate_forecast
from ..models.input import make_model_input

VARIABLES = ("T2m", "T850", "Z500", "U10", "V10")
FINAL_LEADS = (6, 12, 18, 24, 30, 36)


def _rollout_starts(dataset: Any, steps: int) -> list[int]:
    years = np.asarray(dataset.arrays["year"])
    return [i for i in range(len(dataset) - steps + 1) if np.all(years[i:i + steps] == years[i])]


@torch.inference_mode()
def evaluate_corrector(
    predictor: torch.nn.Module,
    corrector: torch.nn.Module,
    dataset: Any,
    stats: Mapping[str, Any],
    latitudes: torch.Tensor,
    *,
    split: str,
    leads: Sequence[int] = FINAL_LEADS,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Evaluate the final Stage-1+corrector autoregressive path at requested leads."""
    requested = tuple(int(x) for x in leads)
    if not requested or any(x % 6 or x < 6 for x in requested):
        raise ValueError("leads must be positive six-hour multiples")
    steps = max(requested) // 6
    starts = _rollout_starts(dataset, steps)
    if not starts:
        raise ValueError("no complete rollout windows")
    predictor.eval(); corrector.eval()
    dev = torch.device(device)
    mean = torch.as_tensor(stats["state_mean"], device=dev)
    std = torch.as_tensor(stats["state_std"], device=dev)
    keep: dict[int, tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]] = {
        lead: ([], [], []) for lead in requested
    }
    years = np.asarray(dataset.arrays["year"])
    annual = {}
    for year in np.unique(years):
        ids = np.flatnonzero(years == year)
        sequence = np.concatenate((np.asarray(dataset.arrays["current"][ids]),
                                   np.asarray(dataset.arrays["target"][ids[-1]])[None]))
        annual[int(year)] = sequence.mean(0, dtype=np.float64).astype(np.float32)
    for left in range(0, len(starts), batch_size):
        index = np.asarray(starts[left:left + batch_size])
        state = torch.from_numpy(np.asarray(dataset.arrays["current"][index])).to(dev)
        climate = torch.from_numpy(np.asarray(dataset.arrays["current_climatology"][index])).to(dev)
        climates = torch.from_numpy(np.stack([annual[int(years[i])] for i in index]))
        for step in range(steps):
            rows = index + step
            target_climate = torch.from_numpy(np.asarray(dataset.arrays["target_climatology"][rows])).to(dev)
            calendar = torch.from_numpy(np.asarray(dataset.arrays["calendar"][rows])).to(dev)
            model_input = make_model_input(state, climate, mean, std, calendar)
            base_out = predictor(model_input, state, climate, target_climate)
            corr_out = corrector(state, base_out["base_prediction"], base_out["decoder_features"])
            state = base_out["base_prediction"] + corr_out["correction"]
            climate = target_climate
            hour = (step + 1) * 6
            if hour in keep:
                keep[hour][0].append(state.float().cpu())
                keep[hour][1].append(torch.from_numpy(np.asarray(dataset.arrays["target"][rows])).clone())
                keep[hour][2].append(climates)
    metrics = {
        str(lead): evaluate_forecast(torch.cat(pred), torch.cat(truth), torch.cat(climate),
                                    latitudes.cpu(), torch.as_tensor(stats["state_std"]))
        for lead, (pred, truth, climate) in keep.items()
    }
    return {"split": split, "n_windows": len(starts), "metrics": metrics}


def _values(metrics: Mapping[str, Any], key: str, names: Sequence[str] = VARIABLES) -> list[float]:
    values = metrics.get(key, {})
    if set(values) != set(VARIABLES):
        raise ValueError(f"metrics.{key} must contain exactly {VARIABLES}")
    result = [float(values[name]) for name in names]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("selection metrics must be finite")
    return result


def phase_a_constraints(metrics: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, bool]:
    wind = sum(_values(metrics, "rmse", VARIABLES[3:])) / 2
    base_wind = sum(_values(base, "rmse", VARIABLES[3:])) / 2
    scalar = sum(_values(metrics, "normalized_rmse", VARIABLES[:3])) / 3
    base_scalar = sum(_values(base, "normalized_rmse", VARIABLES[:3])) / 3
    return {
        "wind_6h_improved": wind < base_wind,
        "scalar_6h_within_1pct": scalar <= 1.01 * base_scalar,
    }


def choose_phase_a_checkpoint(candidates: Sequence[Mapping[str, Any]], base_6h: Mapping[str, Any]) -> dict[str, Any]:
    eligible = []
    for candidate in candidates:
        constraints = phase_a_constraints(candidate["metrics"], base_6h)
        row = dict(candidate)
        row["constraints"] = constraints
        if all(constraints.values()):
            eligible.append(row)
    if not eligible:
        raise ValueError("no Phase-A EMA checkpoint satisfies mandatory constraints")
    return min(eligible, key=lambda item: (float(item["score"]), str(item["checkpoint"])))


def validation_score(metrics: Mapping[str, Any], base_metrics: Mapping[str, Any]) -> float:
    ratios = []
    for lead in (6, 24, 36):
        current = metrics[str(lead)]
        base = base_metrics[str(lead)]
        for variable in VARIABLES:
            denominator = float(base["rmse"][variable])
            if denominator <= 0 or not math.isfinite(denominator):
                raise ValueError("Stage-1 validation RMSE must be positive finite")
            ratios.append(float(current["rmse"][variable]) / denominator)
    return sum(ratios) / 15.0


def phase_b_constraints(candidate: Mapping[str, Any], base_metrics: Mapping[str, Any]) -> dict[str, bool]:
    metrics = candidate["metrics"]
    six = phase_a_constraints(metrics["6"], base_metrics["6"])
    return {
        **six,
        "nrmse_24h_improved": float(metrics["24"]["mean_normalized_rmse"]) < float(base_metrics["24"]["mean_normalized_rmse"]),
        "nrmse_36h_improved": float(metrics["36"]["mean_normalized_rmse"]) < float(base_metrics["36"]["mean_normalized_rmse"]),
        "finite_rollout": bool(candidate.get("finite", False)),
        "correction_stable": bool(candidate.get("correction_stable", False)),
    }


def choose_phase_b_checkpoint(candidates: Sequence[Mapping[str, Any]], base_metrics: Mapping[str, Any]) -> dict[str, Any]:
    if len(candidates) != 3:
        raise ValueError("Phase B selection requires exactly the top three 24h EMA checkpoints")
    rows = []
    for candidate in candidates:
        row = dict(candidate)
        row["score"] = validation_score(candidate["metrics"], base_metrics)
        row["constraints"] = phase_b_constraints(candidate, base_metrics)
        rows.append(row)
    eligible = [row for row in rows if all(row["constraints"].values())]
    selected = min(eligible or rows, key=lambda item: (float(item["score"]), str(item["checkpoint"])))
    selected["selection_fallback"] = not bool(eligible)
    selected["violated_constraints"] = [name for name, passed in selected["constraints"].items() if not passed]
    return selected


def final_test_once(
    checkpoint: str | Path,
    evaluator: Callable[..., Mapping[str, Any]],
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate exactly one validation-selected checkpoint at all six test leads."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"test output already exists: {destination}")
    result = dict(evaluator(Path(checkpoint), split="test", leads=FINAL_LEADS))
    if set(result.get("metrics", {})) != {str(lead) for lead in FINAL_LEADS}:
        raise ValueError("test evaluator did not return all 6..36h leads")
    result.update(
        checkpoint=str(checkpoint),
        split="test",
        leads_hours=list(FINAL_LEADS),
        selection="validation_only_test_once",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return result
