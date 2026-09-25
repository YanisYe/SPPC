"""Expose the Stage-1 forecast to v2 correctors without changing its architecture."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class PredictorAdapter(nn.Module):
    """Expose predictor outputs required by the corrector."""

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        model_input: torch.Tensor,
        current: torch.Tensor,
        current_climatology: torch.Tensor,
        target_climatology: torch.Tensor,
    ) -> dict[str, Any]:
        result = self.predictor(
            model_input, current, current_climatology, target_climatology, True
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("predictor must support return_parts=True")
        base, diagnostics = result
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("predictor diagnostics must be a mapping")
        features = diagnostics.get("features")
        raw = diagnostics.get("raw")
        if features is None and isinstance(raw, Mapping):
            features = raw.get("features")
        if not isinstance(base, torch.Tensor) or not isinstance(features, torch.Tensor):
            raise RuntimeError("predictor did not expose prediction and decoder features")
        if base.ndim != 4 or base.shape[1] != 5:
            raise ValueError("base_prediction must have shape [B,5,H,W]")
        if features.shape != (base.shape[0], 128, base.shape[2], base.shape[3]):
            raise ValueError("decoder_features must have shape [B,128,H,W]")
        return {
            "base_prediction": base,
            "decoder_features": features,
            "base_diagnostics": dict(diagnostics),
        }
