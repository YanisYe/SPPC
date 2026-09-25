"""Independent, physical-unit forecast scoring on [B, C, H, W] tensors.

ACC follows ClimODE (Verma et al., ICLR 2024), third_party/ClimODE/utils.py,
``evaluation_acc_mm`` and evaluation_global.py: subtract a climatological
field, remove each sample's *unweighted spatial* anomaly mean, then compute
cos(latitude)-weighted spatial correlation. Scores are averaged over samples
*after* the per-sample ratio (not a global correlation). ClimODE constructs
its climatology with ``Final_test_data.mean(dim=0)`` (one field per year) and
averages lead/sample scores; callers must supply the appropriate field, never
silently estimate climatology from the evaluated predictions or targets.

Unlike ClimODE's min/max intermediate, inputs here are already physical values;
a common affine inverse-transform cancels inside the anomaly-centering step.
The upstream ``weights_lat.reshape(H,1).repeat(W,1)`` produces [H*W,1]
for NumPy, not [H,W], and cannot multiply a [H,W] anomaly. We broadcast
[H,1] across longitude instead; this preserves its stated latitude weighting
while correcting that dimensional error. This is not bitwise execution of the
broken upstream function.
Undefined ACC (zero spatial variance) is NaN and must not be silently counted
as zero or one. No training or evaluation dataset is imported here.
"""
from __future__ import annotations

import torch

VARIABLES = ("T2m", "T850", "Z500", "U10", "V10")
LEADS_HOURS = (6, 12, 18, 24, 36, 72, 144)


def _check(pred: torch.Tensor, truth: torch.Tensor, latitudes: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4 or pred.shape != truth.shape:
        raise ValueError("prediction and truth must have identical [B,C,H,W] shape")
    if latitudes.ndim != 1 or latitudes.numel() != pred.shape[-2]:
        raise ValueError("latitudes must have one degree-valued entry per latitude row")
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError("prediction/truth contain nonfinite values")
    # Evaluate in float64 to keep centered geopotential anomalies accurate.
    lat = latitudes.to(device=pred.device, dtype=torch.float64)
    weight = torch.cos(torch.deg2rad(lat))
    if not torch.isfinite(weight).all() or (weight < -1e-12).any() or weight.mean() <= 0:
        raise ValueError("invalid latitude values/weights")
    return (weight / weight.mean()).clamp_min(0).view(1, 1, -1, 1)


def latitude_weighted_rmse(pred: torch.Tensor, truth: torch.Tensor,
                           latitudes: torch.Tensor) -> torch.Tensor:
    """Return [B,C] physical RMSE; square-root spatial mean *before* batch mean.

    Matches ``evaluation_rmsd_mm`` in ClimODE's utils.py (cosine weights
    normalized by their latitude mean, then mean over lat/lon).
    """
    w = _check(pred, truth, latitudes)
    return (((pred.double() - truth.double()).square() * w).mean(dim=(-2, -1))).sqrt()


def climode_acc(pred: torch.Tensor, truth: torch.Tensor, climatology: torch.Tensor,
                latitudes: torch.Tensor) -> torch.Tensor:
    """Return [B,C] ClimODE anomaly correlation; undefined entries become NaN.

    ``climatology`` is a matching [B,C,H,W] field or a shared [C,H,W] field.
    Do not replace the unweighted spatial anomaly centering with a weighted
    mean: ClimODE's official ``evaluation_acc_mm`` uses ``np.mean`` there.
    """
    w = _check(pred, truth, latitudes)
    if climatology.shape not in (pred.shape, pred.shape[1:]):
        raise ValueError("climatology must be [C,H,W] or [B,C,H,W]")
    clim = climatology.to(device=pred.device, dtype=torch.float64)
    if not torch.isfinite(clim).all():
        raise ValueError("climatology contains nonfinite values")
    p = pred.double() - clim
    t = truth.double() - clim
    p = p - p.mean(dim=(-2, -1), keepdim=True)
    t = t - t.mean(dim=(-2, -1), keepdim=True)
    numerator = (w * p * t).sum(dim=(-2, -1))
    denominator = ((w * p.square()).sum(dim=(-2, -1)) *
                   (w * t.square()).sum(dim=(-2, -1))).sqrt()
    return torch.where(denominator > 0, numerator / denominator,
                       torch.full_like(numerator, float("nan")))


def evaluate_forecast(pred: torch.Tensor, truth: torch.Tensor, climatology: torch.Tensor,
                      latitudes: torch.Tensor, state_std: torch.Tensor,
                      variables: tuple[str, ...] = VARIABLES) -> dict:
    """Aggregate sample-first metrics for one lead, physical [B,C,H,W].

    ``state_std`` is train-set physical standard deviation [C]; normalized
    RMSE divides each *per-sample* physical RMSE by this quantity. Mean scores
    are simple arithmetic means over variables. ACC cannot be aggregated when
    any per-sample denominator is zero (NaN propagates intentionally).
    """
    if len(variables) != pred.shape[1]:
        raise ValueError("variables must match channel count")
    scale = torch.as_tensor(state_std, dtype=torch.float64, device=pred.device)
    if scale.shape != (pred.shape[1],) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("state_std must be positive finite [C]")
    rmse = latitude_weighted_rmse(pred, truth, latitudes).mean(0)
    acc = climode_acc(pred, truth, climatology, latitudes).mean(0)
    normalized = rmse / scale
    return {
        "rmse": dict(zip(variables, rmse.tolist())),
        "acc": dict(zip(variables, acc.tolist())),
        "normalized_rmse": dict(zip(variables, normalized.tolist())),
        "mean_normalized_rmse": normalized.mean().item(),
        "mean_acc": acc.mean().item(),
        "n_samples": pred.shape[0],
    }


def evaluate_leads(predictions: torch.Tensor, truths: torch.Tensor,
                   climatology: torch.Tensor, latitudes: torch.Tensor,
                   state_std: torch.Tensor, lead_hours: tuple[int, ...] = LEADS_HOURS,
                   variables: tuple[str, ...] = VARIABLES) -> dict[int, dict]:
    """Score [B,L,C,H,W] physical rollouts at explicitly supplied lead hours.

    Climatology may be shared [C,H,W], per-sample [B,C,H,W], or
    lead-dependent [B,L,C,H,W]. Never infer that indexing a 24-step rollout by
    position corresponds to 36/72/144 hours: pass precisely selected leads.
    """
    if predictions.ndim != 5 or truths.shape != predictions.shape or predictions.shape[1] != len(lead_hours):
        raise ValueError("predictions/truths must match [B,L,C,H,W] and lead_hours")
    if len(set(lead_hours)) != len(lead_hours) or any(h <= 0 for h in lead_hours):
        raise ValueError("lead_hours must be unique and positive")
    if climatology.shape not in (predictions.shape[2:],
                                 (predictions.shape[0], *predictions.shape[2:]),
                                 predictions.shape):
        raise ValueError("climatology shape incompatible with rollout")
    return {hour: evaluate_forecast(predictions[:, i], truths[:, i],
                                   climatology[:, i] if climatology.ndim == 5 else climatology,
                                   latitudes, state_std, variables)
            for i, hour in enumerate(lead_hours)}
