#!/usr/bin/env python3
"""Leakage-safe joint samples for the five-node continuous subgraph."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import FEATURE_COLUMNS


TARGET_FEATURES = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)


def _stack_station_values(
    panel: pd.DataFrame,
    station_order: tuple[str, ...],
    index: pd.DatetimeIndex,
) -> np.ndarray:
    station_arrays = []
    for station in station_order:
        frame = panel[panel["station"].eq(station)].copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.set_index("time").reindex(index)
        station_arrays.append(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    return np.stack(station_arrays, axis=1)


def _stack_target_mask(
    quality: pd.DataFrame,
    station_order: tuple[str, ...],
    index: pd.DatetimeIndex,
) -> np.ndarray:
    columns = [f"{feature}__target_ok" for feature in TARGET_FEATURES]
    station_arrays = []
    for station in station_order:
        frame = quality[quality["station"].eq(station)].copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.set_index("time").reindex(index)
        station_arrays.append(frame[columns].fillna(False).to_numpy(dtype=bool))
    return np.stack(station_arrays, axis=1)


def _empty_split(
    input_steps: int,
    output_steps: int,
    node_count: int,
) -> dict[str, np.ndarray]:
    return {
        "history_diffs": np.empty((0, input_steps, node_count, len(FEATURE_COLUMNS))),
        "current_targets": np.empty((0, node_count, len(TARGET_FEATURES))),
        "target_levels": np.empty((0, output_steps, node_count, len(TARGET_FEATURES))),
        "target_mask": np.empty((0, output_steps, node_count, len(TARGET_FEATURES)), dtype=bool),
        "origin_time": np.empty(0, dtype="datetime64[ns]"),
        "history_end_time": np.empty(0, dtype="datetime64[ns]"),
        "target_start": np.empty(0, dtype="datetime64[ns]"),
        "target_end": np.empty(0, dtype="datetime64[ns]"),
    }


def build_joint_samples(
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    station_order: tuple[str, ...],
    input_steps: int = 6,
    output_steps: int = 9,
    train_end: str = "2024-01-01",
    val_end: str = "2025-01-01",
) -> dict[str, dict[str, np.ndarray]]:
    """Create direct multi-step samples without crossing split boundaries."""
    times = pd.to_datetime(panel["time"])
    index = pd.date_range(times.min(), times.max(), freq="4h")
    values = _stack_station_values(panel, station_order, index)
    target_mask = _stack_target_mask(quality, station_order, index)
    target_indices = [FEATURE_COLUMNS.index(feature) for feature in TARGET_FEATURES]
    target_values = values[:, :, target_indices]
    diffs = np.full_like(values, np.nan)
    diffs[1:] = values[1:] - values[:-1]

    rows: dict[str, list[dict[str, np.ndarray | np.datetime64]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    train_boundary = pd.Timestamp(train_end)
    val_boundary = pd.Timestamp(val_end)
    for origin in range(input_steps, len(index) - output_steps):
        history = diffs[origin - input_steps + 1 : origin + 1]
        current = target_values[origin]
        future = target_values[origin + 1 : origin + output_steps + 1]
        future_mask = target_mask[origin + 1 : origin + output_steps + 1] & np.isfinite(future)
        if not np.isfinite(history).all() or not np.isfinite(current).all() or not future_mask.any():
            continue
        target_start = index[origin + 1]
        target_end = index[origin + output_steps]
        if target_end < train_boundary:
            split = "train"
        elif target_start >= train_boundary and target_end < val_boundary:
            split = "val"
        elif target_start >= val_boundary:
            split = "test"
        else:
            continue
        rows[split].append(
            {
                "history_diffs": history,
                "current_targets": current,
                "target_levels": future,
                "target_mask": future_mask,
                "origin_time": index[origin].to_datetime64(),
                "history_end_time": index[origin].to_datetime64(),
                "target_start": target_start.to_datetime64(),
                "target_end": target_end.to_datetime64(),
            }
        )

    output: dict[str, dict[str, np.ndarray]] = {}
    for split, split_rows in rows.items():
        if not split_rows:
            output[split] = _empty_split(input_steps, output_steps, len(station_order))
            continue
        output[split] = {
            key: np.stack([row[key] for row in split_rows])
            if key in {"history_diffs", "current_targets", "target_levels", "target_mask"}
            else np.asarray([row[key] for row in split_rows])
            for key in split_rows[0]
        }
    return output


def fit_train_scalers(train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    history = np.asarray(train["history_diffs"], dtype=float)
    current = np.asarray(train["current_targets"], dtype=float)
    history_mean = np.nanmean(history, axis=(0, 1, 2))
    history_std = np.nanstd(history, axis=(0, 1, 2))
    level_mean = np.nanmean(current, axis=(0, 1))
    level_std = np.nanstd(current, axis=(0, 1))
    history_std = np.where(history_std > 1e-8, history_std, 1.0)
    level_std = np.where(level_std > 1e-8, level_std, 1.0)
    return {
        "history_mean": history_mean,
        "history_std": history_std,
        "level_mean": level_mean,
        "level_std": level_std,
    }


def scale_split(
    split: dict[str, np.ndarray],
    scalers: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in split.items()}
    output["history_diffs"] = (
        split["history_diffs"] - scalers["history_mean"]
    ) / scalers["history_std"]
    output["current_targets"] = (
        split["current_targets"] - scalers["level_mean"]
    ) / scalers["level_std"]
    output["target_levels"] = (
        split["target_levels"] - scalers["level_mean"]
    ) / scalers["level_std"]
    return output


def inverse_levels(values: np.ndarray, scalers: dict[str, np.ndarray]) -> np.ndarray:
    return values * scalers["level_std"] + scalers["level_mean"]
