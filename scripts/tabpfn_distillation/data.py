"""Causal single-station windows for direct 4--72 hour forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import load_processed_4h_data, target_ok_column
from scripts.tabpfn_distillation import config


def load_v2_panel() -> pd.DataFrame:
    panel = load_processed_4h_data(config.OBSERVED_DATA_PATH)
    panel = panel.loc[pd.to_datetime(panel["time"]) >= pd.Timestamp(config.START_DATE)].copy()
    panel["station"] = panel["station"].astype(str)
    panel["time"] = pd.to_datetime(panel["time"])
    return panel.sort_values(["station", "time"]).reset_index(drop=True)


def available_stations(panel: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(panel["station"].dropna().astype(str).unique()))


def _station_grid(
    panel: pd.DataFrame, station: str, target: str
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if target not in config.TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    frame = panel.loc[panel["station"].astype(str).eq(str(station))].copy()
    if frame.empty:
        raise ValueError(f"Unknown station: {station}")
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    start = pd.Timestamp(frame["time"].min()).ceil(f"{config.STEP_HOURS}h")
    end = pd.Timestamp(frame["time"].max()).floor(f"{config.STEP_HOURS}h")
    times = pd.date_range(start, end, freq=f"{config.STEP_HOURS}h")
    frame = frame.set_index("time").reindex(times)
    values = frame.loc[:, config.INPUT_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target_values = values[:, config.INPUT_FEATURES.index(target)]
    quality_column = target_ok_column(target)
    if quality_column in frame:
        target_ok = frame[quality_column].fillna(False).to_numpy(bool)
    else:
        target_ok = np.zeros(len(frame), dtype=bool)
    target_ok &= np.isfinite(target_values)
    return times, values, target_ok


def empty_dataset() -> dict[str, np.ndarray]:
    feature_count = len(config.INPUT_FEATURES)
    return {
        "x_raw": np.empty((0, config.INPUT_STEPS, feature_count), dtype=float),
        "x_diff": np.empty((0, config.INPUT_STEPS, feature_count), dtype=float),
        "x_raw_mask": np.empty((0, config.INPUT_STEPS, feature_count), dtype=bool),
        "x_diff_mask": np.empty((0, config.INPUT_STEPS, feature_count), dtype=bool),
        "current": np.empty((0, 1), dtype=float),
        "current_mask": np.empty((0, 1), dtype=bool),
        "y_delta": np.empty((0, config.OUTPUT_STEPS), dtype=float),
        "y_abs": np.empty((0, config.OUTPUT_STEPS), dtype=float),
        "y_mask": np.empty((0, config.OUTPUT_STEPS), dtype=bool),
        "target_start": np.asarray([], dtype="datetime64[ns]"),
        "target_end": np.asarray([], dtype="datetime64[ns]"),
    }


def build_station_target_dataset(
    panel: pd.DataFrame, station: str, target: str
) -> dict[str, np.ndarray]:
    """Build one 24-hour input and 18 direct future labels per origin."""
    times, values, target_ok = _station_grid(panel, station, target)
    total_steps = config.INPUT_STEPS + config.OUTPUT_STEPS
    if len(times) < total_steps:
        return empty_dataset()

    diffs = np.full_like(values, np.nan, dtype=float)
    diff_valid = np.zeros_like(values, dtype=bool)
    if len(values) > 1:
        candidate = values[1:] - values[:-1]
        valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        diffs[1:] = np.where(valid, candidate, np.nan)
        diff_valid[1:] = valid

    target_index = config.INPUT_FEATURES.index(target)
    rows: dict[str, list[object]] = {
        "x_raw": [],
        "x_diff": [],
        "x_raw_mask": [],
        "x_diff_mask": [],
        "current": [],
        "current_mask": [],
        "y_delta": [],
        "y_abs": [],
        "y_mask": [],
        "target_start": [],
        "target_end": [],
    }
    for begin in range(len(times) - total_steps + 1):
        current_index = begin + config.INPUT_STEPS - 1
        future_slice = slice(current_index + 1, current_index + 1 + config.OUTPUT_STEPS)
        current = values[current_index, target_index]
        future = values[future_slice, target_index]
        current_valid = bool(target_ok[current_index])
        future_valid = target_ok[future_slice]

        rows["x_raw"].append(values[begin : current_index + 1])
        rows["x_diff"].append(diffs[begin : current_index + 1])
        rows["x_raw_mask"].append(np.isfinite(values[begin : current_index + 1]))
        rows["x_diff_mask"].append(diff_valid[begin : current_index + 1])
        rows["current"].append([current])
        rows["current_mask"].append([current_valid])
        rows["y_delta"].append(future - current)
        rows["y_abs"].append(future)
        rows["y_mask"].append(current_valid & future_valid & np.isfinite(future))
        rows["target_start"].append(times[current_index + 1].to_datetime64())
        rows["target_end"].append(times[current_index + config.OUTPUT_STEPS].to_datetime64())

    return {
        "x_raw": np.asarray(rows["x_raw"], dtype=float),
        "x_diff": np.asarray(rows["x_diff"], dtype=float),
        "x_raw_mask": np.asarray(rows["x_raw_mask"], dtype=bool),
        "x_diff_mask": np.asarray(rows["x_diff_mask"], dtype=bool),
        "current": np.asarray(rows["current"], dtype=float),
        "current_mask": np.asarray(rows["current_mask"], dtype=bool),
        "y_delta": np.asarray(rows["y_delta"], dtype=float),
        "y_abs": np.asarray(rows["y_abs"], dtype=float),
        "y_mask": np.asarray(rows["y_mask"], dtype=bool),
        "target_start": np.asarray(rows["target_start"], dtype="datetime64[ns]"),
        "target_end": np.asarray(rows["target_end"], dtype="datetime64[ns]"),
    }


def subset(dataset: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    return {key: np.asarray(value)[mask] for key, value in dataset.items()}


def split_by_time(dataset: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Use the full 72-hour label extent when enforcing split boundaries."""
    target_start = pd.to_datetime(dataset["target_start"])
    target_end = pd.to_datetime(dataset["target_end"])
    masks = {
        "train": target_end < pd.Timestamp(config.TRAIN_END),
        "val": (target_start >= pd.Timestamp(config.TRAIN_END))
        & (target_end < pd.Timestamp(config.VAL_END)),
        "test": target_start >= pd.Timestamp(config.VAL_END),
    }
    return {name: subset(dataset, np.asarray(mask)) for name, mask in masks.items()}


def join_splits(*splits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not splits:
        return empty_dataset()
    return {
        key: np.concatenate([np.asarray(split[key]) for split in splits], axis=0)
        for key in splits[0]
    }


def tabpfn_features(split: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.asarray(split["x_raw"], dtype=float).reshape(len(split["x_raw"]), -1)
    diffs = np.asarray(split["x_diff"], dtype=float).reshape(len(split["x_diff"]), -1)
    raw_mask = np.asarray(split["x_raw_mask"], dtype=bool).reshape(len(raw), -1).astype(float)
    diff_mask = np.asarray(split["x_diff_mask"], dtype=bool).reshape(len(raw), -1).astype(float)
    current = np.asarray(split["current"], dtype=float)
    current_mask = np.asarray(split["current_mask"], dtype=bool).astype(float)
    return np.concatenate((raw, diffs, raw_mask, diff_mask, current, current_mask), axis=1)


def target_values(split: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.asarray(split["y_abs"], dtype=float)
    if mode == "delta":
        return np.asarray(split["y_delta"], dtype=float)
    raise ValueError(f"Unknown target mode: {mode}")


def to_absolute(values: np.ndarray, current: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "absolute":
        return values
    if mode == "delta":
        return values + np.repeat(np.asarray(current, dtype=float), config.OUTPUT_STEPS, axis=1)
    raise ValueError(f"Unknown target mode: {mode}")


def causal_oof_folds(train: dict[str, np.ndarray]) -> tuple[dict[str, object], ...]:
    """Return expanding-window OOF masks with a strict label-time gap."""
    target_start = pd.to_datetime(train["target_start"])
    target_end = pd.to_datetime(train["target_end"])
    folds = []
    for index, (name, prediction_start, prediction_end) in enumerate(config.OOF_FOLDS):
        start = pd.Timestamp(prediction_start)
        end = pd.Timestamp(prediction_end)
        fit_mask = np.asarray(target_end < start)
        prediction_mask = np.asarray((target_start >= start) & (target_end < end))
        folds.append(
            {
                "index": index,
                "name": name,
                "prediction_start": start,
                "prediction_end": end,
                "fit_mask": fit_mask,
                "prediction_mask": prediction_mask,
            }
        )
    return tuple(folds)


def prediction_arrays(split: dict[str, np.ndarray], pred: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pred": np.asarray(pred, dtype=float),
        "true": np.asarray(split["y_abs"], dtype=float),
        "mask": np.asarray(split["y_mask"], dtype=bool),
        "current": np.asarray(split["current"], dtype=float),
        "target_start": np.asarray(split["target_start"], dtype="datetime64[ns]"),
    }
