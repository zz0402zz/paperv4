#!/usr/bin/env python3
"""Shared helpers for water-quality forecasting experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.baselines import gat_gru_baseline as base

def save_json(path: str | Path, payload: dict) -> None:
    """Save a UTF-8 JSON file with parent directories created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def station_samples(split: dict[str, np.ndarray], station_count: int) -> dict[str, np.ndarray]:
    """Flatten graph windows W,T,N,F into station-level samples W*N,T,F."""
    x = split["x"]
    y = split["y"]
    mask = split["y_mask"]
    windows, steps, _, features = x.shape
    output_steps, target_dim = y.shape[1], y.shape[-1]
    x_flat = x.transpose(0, 2, 1, 3).reshape(windows * station_count, steps, features)
    y_flat = y.transpose(0, 2, 1, 3).reshape(windows * station_count, output_steps, target_dim)
    mask_flat = mask.transpose(0, 2, 1, 3).reshape(windows * station_count, output_steps, target_dim)
    station_id = np.tile(np.arange(station_count, dtype=np.int64), windows)
    return {
        "x": x_flat.astype(np.float32),
        "y": y_flat.astype(np.float32),
        "mask": mask_flat.astype(bool),
        "station_id": station_id,
        "train_keep": mask_flat.any(axis=(1, 2)),
        "windows": windows,
        "station_count": station_count,
    }


def one_hot_station(station_id: np.ndarray, station_count: int) -> np.ndarray:
    """Build station one-hot features for linear/tabular baselines."""
    out = np.zeros((len(station_id), station_count), dtype=np.float32)
    out[np.arange(len(station_id)), station_id] = 1.0
    return out


def tabular_features(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten temporal features and append station identity."""
    x = samples["x"].reshape(len(samples["x"]), -1)
    station = one_hot_station(samples["station_id"], samples["station_count"])
    bias = np.ones((len(x), 1), dtype=np.float32)
    return np.concatenate([bias, x, station], axis=1).astype(np.float64)


class StationDataset:
    """PyTorch-compatible station-level dataset with target masks."""

    def __init__(self, samples: dict[str, np.ndarray], train_only_valid: bool = False) -> None:
        keep = samples["train_keep"] if train_only_valid else np.ones(len(samples["x"]), dtype=bool)
        self.x = samples["x"][keep]
        self.y = samples["y"][keep]
        self.mask = samples["mask"][keep]
        self.station_id = samples["station_id"][keep]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.station_id[idx], self.y[idx], self.mask[idx]


def make_station_loader(torch, samples: dict[str, np.ndarray], batch_size: int, shuffle: bool, train_only_valid: bool):
    """Create a station-level DataLoader."""
    dataset = StationDataset(samples, train_only_valid=train_only_valid)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def masked_l1_loss(torch, pred, target, mask):
    """Masked L1 loss for partially observed multi-target labels."""
    weights = mask.to(dtype=pred.dtype)
    return (torch.abs(pred - target) * weights).sum() / weights.sum().clamp_min(1.0)


def scaled_flat_rmse(pred: np.ndarray, samples: dict[str, np.ndarray]) -> float:
    """RMSE in scaled target space for early stopping."""
    mask = samples["mask"] & np.isfinite(pred) & np.isfinite(samples["y"])
    if not mask.any():
        return float("inf")
    error = pred - samples["y"]
    return float(np.sqrt(np.mean(error[mask] ** 2)))


def target_input_indices(input_feature_columns: tuple[str, ...], target_feature_columns: tuple[str, ...]) -> tuple[int, ...]:
    """Find target feature positions inside the input feature list."""
    missing = [feature for feature in target_feature_columns if feature not in input_feature_columns]
    if missing:
        raise ValueError(f"Persistence baseline needs target features in inputs: {missing}")
    return tuple(input_feature_columns.index(feature) for feature in target_feature_columns)


def evaluate_persistence(
    raw_splits: dict[str, dict[str, np.ndarray]],
    target_feature_columns: tuple[str, ...],
    stations,
    input_feature_columns: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Persistence baseline: future target equals the last observed input target value."""
    target_dim = len(target_feature_columns)
    indices = (
        tuple(range(target_dim))
        if input_feature_columns is None
        else target_input_indices(input_feature_columns, target_feature_columns)
    )
    metrics = {}
    for split_name, split in raw_splits.items():
        if len(split["x"]) == 0:
            empty = np.empty((0, split["y"].shape[1], len(stations), target_dim))
            metrics[split_name] = base.masked_error_metrics(empty, empty.astype(bool), target_feature_columns, stations)
            continue
        pred = np.repeat(split["x"][:, -1:, :, indices], split["y"].shape[1], axis=1)
        valid = split["y_mask"] & np.isfinite(pred)
        metrics[split_name] = base.masked_error_metrics(pred - split["y"], valid, target_feature_columns, stations, truth=split["y"])
    return metrics


def prediction_row(model_name: str, metrics: dict, best_epoch: dict | None = None) -> dict[str, object]:
    """Build one overall metrics row from train/val/test metrics."""
    test = metrics["test"]
    return {
        "model": model_name,
        "best_epoch": None if best_epoch is None else best_epoch.get("epoch"),
        "val_rmse": metrics["val"].get("rmse"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_nse": test.get("nse"),
        "valid_points": test.get("valid_points"),
    }


def feature_rows(model_name: str, metrics: dict, target_feature_columns: tuple[str, ...]) -> list[dict[str, object]]:
    """Build test-set feature metric rows."""
    test = metrics["test"]
    return [
        {
            "model": model_name,
            "feature": feature,
            "valid_points": test["feature_valid_points"].get(feature, 0),
            "test_mae": test["feature_mae"].get(feature),
            "test_rmse": test["feature_rmse"].get(feature),
            "test_nse": test["feature_nse"].get(feature),
        }
        for feature in target_feature_columns
    ]


def station_rows(model_name: str, metrics: dict, stations) -> list[dict[str, object]]:
    """Build test-set station metric rows."""
    rows = []
    for station in stations:
        item = metrics["test"]["station_metrics"].get(station, {})
        rows.append(
            {
                "model": model_name,
                "station": station,
                "valid_points": item.get("valid_points", 0),
                "test_mae": item.get("mae"),
                "test_rmse": item.get("rmse"),
                "test_nse": item.get("nse"),
            }
        )
    return rows
