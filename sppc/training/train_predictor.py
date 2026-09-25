"""Epoch-based single-step SSP training."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.cache import CachedWeatherDataset
from .common import (
    TopKCheckpointManager,
    atomic_json_write,
    atomic_torch_save,
    build_optimizer_bundle,
    capture_rng_state,
    learning_rate_for_update,
    load_config,
    restore_rng_state,
    set_reproducibility,
    validate_predictor_config,
)
from ..evaluation.metrics import VARIABLES, evaluate_forecast

if TYPE_CHECKING:
    from ..models.predictor import StructuredPredictor


def collate_cached(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.stack([sample[key] for sample in batch])) for key in batch[0]}


def make_epoch_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    num_workers: int,
    collate_fn=None,
) -> DataLoader:
    """Create a reproducibly shuffled loader whose permutation depends on epoch."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    if num_workers:
        # Six concurrent runs can exhaust the low per-process FD limit when
        # every tensor is transferred through a distinct descriptor.
        torch.multiprocessing.set_sharing_strategy("file_system")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": num_workers,
        "generator": generator,
        "collate_fn": collate_fn,
    }
    if num_workers:
        kwargs.update(pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def validation_score(metrics: Mapping[str, Any]) -> float:
    normalized = metrics.get("normalized_rmse", {})
    if set(normalized) != set(VARIABLES) or len(normalized) != 5:
        raise ValueError("validation score requires exactly five variable normalized RMSE values")
    values = [float(normalized[name]) for name in VARIABLES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite validation metric")
    return float(sum(values) / 5.0)


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def backward_with_fp32_retry(
    model: torch.nn.Module,
    closure,
) -> tuple[torch.Tensor, bool]:
    """Retry a numerically failed BF16 backward once in FP32.

    The optimizer step is owned by the caller. This keeps the same batch and
    does not silently skip data or change the learning-rate schedule.
    """
    model.zero_grad(set_to_none=True)
    loss = closure(True)
    if torch.isfinite(loss):
        loss.backward()
    gradients_finite = torch.isfinite(loss) and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if gradients_finite:
        return loss, False
    model.zero_grad(set_to_none=True)
    loss = closure(False)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss after FP32 retry")
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError("nonfinite gradient after FP32 retry")
    return loss, True


def _loss(model: StructuredPredictor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    prediction, parts = model(
        batch["model_input"].float(),
        batch["current"].float(),
        batch["current_climatology"].float(),
        batch["target_climatology"].float(),
        True,
    )
    residual = (prediction.float() - batch["target"].float()) / model.delta_std[None, :, None, None]
    primary = (residual.square() * model.areas.float()).sum((-2, -1)).div(model.areas.float().sum()).mean()
    auxiliary = batch["aux_wind_mid"].float()
    teacher = torch.cat((model.project_era_wind(auxiliary[:, :2]), model.project_era_wind(auxiliary[:, 2:])), 1)
    wind = (parts["wind850"] - teacher[:, :2]).square().mean() + (parts["wind500"] - teacher[:, 2:]).square().mean()
    return primary + 0.1 * wind


@torch.inference_mode()
def validate_full_2016(
    model: StructuredPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    latitudes: torch.Tensor,
    state_std: torch.Tensor,
) -> dict[str, Any]:
    """Score every 2016 one-step pair with ClimODE physical metrics."""
    model.eval()
    predictions: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    climatology_sum: torch.Tensor | None = None
    last_target: torch.Tensor | None = None
    samples = 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            prediction = model(
                batch["model_input"].float(),
                batch["current"].float(),
                batch["current_climatology"].float(),
                batch["target_climatology"].float(),
            )
            predictions.append(prediction.float().cpu())
            truths.append(batch["target"].float().cpu())
            current_sum = batch["current"].double().sum(0).cpu()
            climatology_sum = current_sum if climatology_sum is None else climatology_sum + current_sum
            samples += int(batch["target"].shape[0])
            last_target = batch["target"][-1].double().cpu()
    if samples != len(loader.dataset):
        raise RuntimeError(f"validation did not cover full dataset: {samples} != {len(loader.dataset)}")
    if climatology_sum is None or last_target is None:
        raise ValueError("empty validation dataset")
    annual_climatology = ((climatology_sum + last_target) / (samples + 1)).float()
    metrics = evaluate_forecast(
        torch.cat(predictions), torch.cat(truths), annual_climatology,
        latitudes.cpu(), state_std.cpu(),
    )
    if metrics["n_samples"] != len(loader.dataset):
        raise RuntimeError("metric aggregation omitted validation samples")
    metrics["score"] = validation_score(metrics)
    metrics["lead_hours"] = 6
    metrics["split_year"] = 2016
    metrics["protocol"] = "ClimODE physical-unit cosine-latitude RMSE/ACC; sample-first aggregation"
    return metrics


def _resume_payload(
    model: StructuredPredictor,
    optimizers: list[torch.optim.Optimizer],
    optimizer_audit: list[dict[str, Any]],
    *,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "optimizer_audit": optimizer_audit,
        "completed_epoch": epoch,
        "global_step": global_step,
        "history": history,
        "rng": capture_rng_state(),
        "selection_weight_source": "raw_model",
        "ema": None,
    }


def run(config_path: str | Path, *, resume: bool = True) -> None:
    config = load_config(config_path)
    validate_predictor_config(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("predictor training requires exactly one visible CUDA GPU")

    device = torch.device("cuda")
    set_reproducibility(int(config["seed"]))
    torch.set_float32_matmul_precision("high")

    root = Path(__file__).parents[2]
    inherited = Path(config["inherited_root"])
    run_dir = root / config["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "status.json"
    metrics_path = run_dir / "epoch_metrics.jsonl"
    resume_path = run_dir / "resume.pt"

    stats = json.loads((inherited / "results/data_audit.json").read_text())["normalization"]
    latitude = torch.tensor(json.loads((inherited / "results/data_audit.json").read_text())["grid"]["latitude"])
    from ..models.predictor import StructuredPredictor

    train_data = CachedWeatherDataset(inherited / config["cache_train"])
    validation_data = CachedWeatherDataset(inherited / config["cache_validation"])
    if len(validation_data) != int(config["validation_samples"]):
        raise RuntimeError(f"expected {config['validation_samples']} validation pairs, found {len(validation_data)}")

    model = StructuredPredictor(
        latitude.deg2rad(), stats["state_std"], stats["delta_std"]
    ).to(device)
    optimizers, optimizer_audit = build_optimizer_bundle(model, config, device=device)
    steps_per_epoch = math.ceil(len(train_data) / int(config["batch_size"]))
    total_updates = int(config["epochs"]) * steps_per_epoch
    warmup_updates = int(config["warmup_epochs"]) * steps_per_epoch
    start_epoch, global_step, history = 1, 0, []
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(state["completed_epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state["history"])
        restore_rng_state(state["rng"])

    validation_loader = DataLoader(
        validation_data,
        batch_size=int(config["validation_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(config["num_workers"]),
        collate_fn=collate_cached,
    )
    top3 = TopKCheckpointManager(run_dir / "top3", int(config["top_k"]))
    atomic_json_write(ledger_path, {
        "status": "running", "variant": config["variant"], "next_epoch": start_epoch,
        "epochs": config["epochs"], "global_step": global_step,
        "selection_weight_source": "raw_model", "ema_enabled": False,
    })

    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            epoch_start = time.perf_counter()
            loader = make_epoch_loader(
                train_data, int(config["batch_size"]), seed=int(config["seed"]), epoch=epoch,
                num_workers=int(config["num_workers"]), collate_fn=collate_cached,
            )
            model.train()
            loss_sum = 0.0
            sample_count = 0
            grad_norm_sum = 0.0
            for cpu_batch in loader:
                batch = _move(cpu_batch, device)
                lr = learning_rate_for_update(
                    global_step, total_updates, warmup_updates,
                    float(config["learning_rate"]), float(config["minimum_learning_rate"]),
                )
                for optimizer in optimizers:
                    for group in optimizer.param_groups: group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                def loss_closure(use_autocast: bool) -> torch.Tensor:
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return _loss(model, batch)
                    with torch.autocast(device_type="cuda", enabled=False):
                        return _loss(model, batch)

                loss, fp32_retry = backward_with_fp32_retry(model, loss_closure)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                for optimizer in optimizers: optimizer.step()
                batch_size = int(batch["target"].shape[0])
                loss_sum += float(loss.detach()) * batch_size
                grad_norm_sum += float(grad_norm)
                sample_count += batch_size
                global_step += 1

            validation = validate_full_2016(
                model, validation_loader, device=device, latitudes=latitude,
                state_std=torch.tensor(stats["state_std"]),
            )
            elapsed = time.perf_counter() - epoch_start
            record = {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "validation": validation,
                "learning_rates": [optimizer.param_groups[0]["lr"] for optimizer in optimizers],
                "mean_gradient_norm": grad_norm_sum / len(loader),
                "seconds": elapsed,
                "global_step": global_step,
                "train_samples": sample_count,
            }
            history.append(record)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            top3.consider(
                epoch=epoch, score=validation["score"], model_state=model.state_dict(), metrics=validation
            )
            atomic_torch_save(
                resume_path,
                _resume_payload(model, optimizers, optimizer_audit, epoch=epoch, global_step=global_step, history=history),
            )
            atomic_json_write(ledger_path, {
                "status": "running" if epoch < int(config["epochs"]) else "completed",
                "variant": config["variant"], "completed_epoch": epoch,
                "epochs": config["epochs"], "global_step": global_step,
                "latest": record, "top3": top3.records,
                "selection_weight_source": "raw_model", "ema_enabled": False,
                "updated_at_unix": time.time(),
            })
            print(json.dumps(record, allow_nan=False), flush=True)
    except BaseException as error:
        atomic_json_write(ledger_path, {
            "status": "failed", "variant": config["variant"], "completed_epoch": start_epoch - 1,
            "global_step": global_step, "error": repr(error), "updated_at_unix": time.time(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, resume=not arguments.no_resume)


if __name__ == "__main__":
    main()
