#!/usr/bin/env python3
"""Leakage-safe metrics and event thresholds for continuous-subgraph runs."""

from __future__ import annotations

import numpy as np


def fit_absolute_change_thresholds(
    train_changes: np.ndarray,
    quantile: float = 0.9,
) -> np.ndarray:
    """Fit per-feature event thresholds from an explicitly supplied train array."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    values = np.asarray(train_changes, dtype=float)
    if values.ndim < 2:
        raise ValueError("train_changes must include a feature dimension")
    return np.nanquantile(np.abs(values), quantile, axis=tuple(range(values.ndim - 1)))


def masked_regression_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(actual)
    if not valid.any():
        return {"valid_points": 0, "mae": None, "rmse": None, "nse": None}
    error = pred[valid] - actual[valid]
    actual_valid = actual[valid]
    denominator = float(np.sum((actual_valid - actual_valid.mean()) ** 2))
    return {
        "valid_points": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nse": None
        if denominator <= np.finfo(float).eps
        else float(1.0 - np.sum(error**2) / denominator),
    }
