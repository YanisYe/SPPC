"""Evaluate the bundled final SPPC checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data.cache import CachedWeatherDataset
from .evaluation.rollout import FINAL_LEADS, evaluate_corrector
from .models.corrector import StructuredCorrector
from .models.predictor import StructuredPredictor
from .models.predictor_adapter import PredictorAdapter
from .training.corrector_objectives import CorrectorEMA

ROOT = Path(__file__).parents[1]
CHECKPOINT = ROOT / "checkpoints/structured_corrector_a300_b200/final_ema.pt"
TEST_CACHE = ROOT / "cache/era5/test"


def evaluate(output: str | Path = "results/sppc_test_6_36.json", batch_size: int = 32) -> dict:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    device = torch.device("cuda")
    latitudes = torch.tensor(metadata["latitude"], dtype=torch.float32)
    stats = {
        "state_mean": metadata["state_mean"],
        "state_std": metadata["state_std"],
        "delta_std": metadata["delta_std"],
    }
    predictor = PredictorAdapter(
        StructuredPredictor(
            latitudes.deg2rad(), metadata["state_std"], metadata["delta_std"]
        ).to(device)
    )
    corrector = StructuredCorrector(
        state_mean=metadata["state_mean"],
        state_std=metadata["state_std"],
        std_dx=metadata["delta_std"],
        std_res_zero=metadata["std_res_zero"],
        std_res_mean=metadata["std_res_mean"],
        std_res_wind=metadata["std_res_wind"],
        latitudes=latitudes.deg2rad(),
    ).to(device)
    ema = CorrectorEMA({"predictor": predictor, "corrector": corrector})
    ema.load_state_dict(state["ema"])
    test = CachedWeatherDataset(TEST_CACHE)
    with ema.apply({"predictor": predictor, "corrector": corrector}):
        result = evaluate_corrector(
            predictor, corrector, test, stats, latitudes,
            split="test", leads=FINAL_LEADS, batch_size=batch_size, device=device,
        )
    result.update(status="test", checkpoint=str(CHECKPOINT), years=[2017, 2018])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/sppc_test_6_36.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.output, args.batch_size)


if __name__ == "__main__":
    main()
