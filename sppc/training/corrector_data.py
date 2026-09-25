"""Aligned Phase-A caches, Phase-B windows, and FP32 residual statistics."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


_CACHE_FIELDS = (
    "model_input", "current", "target", "current_climatology", "target_climatology",
    "base_prediction", "decoder_features", "calendar", "year",
)


def _copy_tensor(value: Any) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _stack_samples(samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.stack([np.asarray(sample[key]) for sample in samples]))
        for key in (
            "model_input", "current", "target", "current_climatology",
            "target_climatology", "calendar", "year",
        )
    }


def write_phase_a_cache(
    dataset: Any,
    cache_root: str | Path,
    split: str,
    infer: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
) -> Path:
    """Run frozen Stage-1 once and atomically write index-aligned float32 arrays."""
    if split not in {"train", "val"}:
        raise ValueError("Phase-A cache split must be train or val")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(cache_root)
    destination = root / split
    if destination.exists():
        raise FileExistsError(destination)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{split}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    arrays: dict[str, np.memmap] = {}
    try:
        count = len(dataset)
        if count < 1:
            raise ValueError("cannot cache an empty dataset")
        first = dataset[0]
        shape = np.asarray(first["current"]).shape
        if len(shape) != 3 or shape[0] != 5:
            raise ValueError("current samples must have shape [5,H,W]")
        h, w = shape[-2:]
        specifications = {
            "model_input": ((count, 14, h, w), "float32"),
            "current": ((count, 5, h, w), "float32"),
            "target": ((count, 5, h, w), "float32"),
            "current_climatology": ((count, 5, h, w), "float32"),
            "target_climatology": ((count, 5, h, w), "float32"),
            "base_prediction": ((count, 5, h, w), "float32"),
            "decoder_features": ((count, 128, h, w), "float32"),
            "calendar": ((count, 4), "int16"),
            "year": ((count,), "int16"),
        }
        for name, (array_shape, dtype) in specifications.items():
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy", mode="w+", dtype=dtype, shape=array_shape
            )
        for left in range(0, count, batch_size):
            right = min(count, left + batch_size)
            samples = [dataset[index] for index in range(left, right)]
            batch = _stack_samples(samples)
            with torch.inference_mode():
                output = infer(batch)
            base = output["base_prediction"].detach().float().cpu().numpy()
            features = output["decoder_features"].detach().float().cpu().numpy()
            if base.shape != (right - left, 5, h, w):
                raise ValueError("cached base_prediction shape mismatch")
            if features.shape != (right - left, 128, h, w):
                raise ValueError("cached decoder_features shape mismatch")
            arrays["current"][left:right] = batch["current"].float().numpy()
            arrays["target"][left:right] = batch["target"].float().numpy()
            arrays["model_input"][left:right] = batch["model_input"].float().numpy()
            arrays["current_climatology"][left:right] = batch["current_climatology"].float().numpy()
            arrays["target_climatology"][left:right] = batch["target_climatology"].float().numpy()
            arrays["calendar"][left:right] = batch["calendar"].short().numpy()
            arrays["year"][left:right] = batch["year"].short().numpy()
            arrays["base_prediction"][left:right] = base
            arrays["decoder_features"][left:right] = features
        for array in arrays.values():
            array.flush()
        manifest = {
            "version": 2,
            "split": split,
            "samples": count,
            "alignment": "source dataset index; no shuffle",
            "dtypes": {name: dtype for name, (_, dtype) in specifications.items()},
            "arrays": {name: f"{name}.npy" for name in specifications},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


class CorrectorCacheDataset(Dataset):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("version") != 2:
            raise ValueError("corrector cache must be version 2")
        self.arrays = {
            name: np.load(self.directory / filename, mmap_mode="r")
            for name, filename in self.manifest["arrays"].items()
        }
        if set(self.arrays) != set(_CACHE_FIELDS):
            raise ValueError("corrector cache fields are incomplete")
        count = int(self.manifest["samples"])
        if any(len(array) != count for array in self.arrays.values()):
            raise ValueError("corrector cache arrays are not aligned")

    def __len__(self) -> int:
        return int(self.manifest["samples"])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {name: np.asarray(array[index]) for name, array in self.arrays.items()}


class FourStepCorrectorDataset(Dataset):
    """Contiguous four-transition windows with calendar data for every target."""

    def __init__(self, dataset: Any, steps: int = 4) -> None:
        if steps != 4:
            raise ValueError("Phase B requires exactly four rollout steps")
        self.dataset = dataset
        years = np.asarray(dataset.arrays["year"])
        self.starts = tuple(
            index for index in range(len(dataset) - steps + 1)
            if np.all(years[index:index + steps] == years[index])
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | np.ndarray]:
        start = self.starts[item]
        rows = slice(start, start + 4)
        arrays = self.dataset.arrays
        calendars = np.array(arrays["calendar"][rows], copy=True)
        return {
            "initial_x": _copy_tensor(arrays["current"][start]),
            "targets": _copy_tensor(arrays["target"][rows]),
            "current_climatologies": _copy_tensor(arrays["current_climatology"][rows]),
            "target_climatologies": _copy_tensor(arrays["target_climatology"][rows]),
            "calendars": torch.from_numpy(calendars),
            "target_times": calendars[:, 2:].copy(),
            "start": torch.tensor(start, dtype=torch.int64),
        }


class ResidualStatsAccumulator:
    """Streaming FP32 protocol statistics for target minus Stage-1 proposal."""

    def __init__(self, cell_area: torch.Tensor | np.ndarray) -> None:
        area = torch.as_tensor(cell_area, dtype=torch.float32)
        if area.ndim != 2 or not torch.isfinite(area).all() or (area <= 0).any():
            raise ValueError("cell_area must be positive finite [H,W]")
        self.area = area / area.sum()
        self.count = 0
        self.zero_square = torch.zeros(3, dtype=torch.float32)
        self.mean_sum = torch.zeros(3, dtype=torch.float32)
        self.mean_square = torch.zeros(3, dtype=torch.float32)
        self.total_square = torch.zeros(3, dtype=torch.float32)
        self.wind_square = torch.zeros(2, dtype=torch.float32)

    @torch.no_grad()
    def update(self, residual: torch.Tensor | np.ndarray) -> None:
        values = torch.as_tensor(residual, dtype=torch.float32, device="cpu")
        if values.ndim != 4 or values.shape[1] != 5 or values.shape[-2:] != self.area.shape:
            raise ValueError("residual must have shape [B,5,H,W]")
        if not torch.isfinite(values).all():
            raise ValueError("residual contains nonfinite values")
        scalar = values[:, :3]
        means = (scalar * self.area).sum((-2, -1))
        zero = scalar - means[..., None, None]
        batch = values.shape[0]
        self.zero_square += (zero.square() * self.area).sum((-2, -1)).sum(0)
        self.mean_sum += means.sum(0)
        self.mean_square += means.square().sum(0)
        self.total_square += (scalar.square() * self.area).sum((-2, -1)).sum(0)
        self.wind_square += (values[:, 3:].square() * self.area).sum((-2, -1)).sum(0)
        self.count += batch

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count < 1:
            raise ValueError("no residuals accumulated")
        n = float(self.count)
        mean_variance = (self.mean_square / n - (self.mean_sum / n).square()).clamp_min(0)
        result = {
            "std_res_zero": (self.zero_square / n).sqrt().clamp_min(1e-6),
            "std_res_mean": mean_variance.sqrt().clamp_min(1e-6),
            "std_res_total": (self.total_square / n).sqrt().clamp_min(1e-6),
            "std_res_wind": (self.wind_square / n).sqrt().clamp_min(1e-6),
        }
        return {key: value.numpy().astype(np.float32, copy=False) for key, value in result.items()}

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, **self.finalize(), accumulation_dtype=np.array("float32"), count=np.int64(self.count))
        return hashlib.sha256(destination.read_bytes()).hexdigest()
