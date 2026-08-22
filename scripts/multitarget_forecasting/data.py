"""Leakage-safe single-station windows with five simultaneous targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import load_processed_4h_data, target_ok_column
from scripts.multitarget_forecasting import config


def load_panel() -> pd.DataFrame:
    panel = load_processed_4h_data(config.OBSERVED_DATA_PATH)
    panel = panel.loc[
        pd.to_datetime(panel["time"]) >= pd.Timestamp(config.START_DATE)
    ].copy()
    panel["station"] = panel["station"].astype(str)
    panel["time"] = pd.to_datetime(panel["time"])
    return panel.sort_values(["station", "time"]).reset_index(drop=True)


def load_development_panel() -> pd.DataFrame:
    """Load only train and validation dates; never materialize test labels."""

    panel = load_panel()
    return panel.loc[panel["time"] < pd.Timestamp(config.VAL_END)].copy()


def available_stations(panel: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(panel["station"].dropna().astype(str).unique()))


def _station_grid(
    panel: pd.DataFrame, station: str
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = panel.loc[panel["station"].astype(str).eq(str(station))].copy()
    if frame.empty:
        raise ValueError(f"未知站点: {station}")
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    start = pd.Timestamp(frame["time"].min()).ceil(f"{config.STEP_HOURS}h")
    end = pd.Timestamp(frame["time"].max()).floor(f"{config.STEP_HOURS}h")
    times = pd.date_range(start, end, freq=f"{config.STEP_HOURS}h")
    frame = frame.set_index("time").reindex(times)
    values = (
        frame.loc[:, config.INPUT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    target_ok = np.zeros((len(frame), len(config.TARGETS)), dtype=bool)
    for target_index, target in enumerate(config.TARGETS):
        quality_column = target_ok_column(target)
        if quality_column in frame:
            target_ok[:, target_index] = frame[quality_column].fillna(False).to_numpy(
                bool
            )
        feature_index = config.INPUT_FEATURES.index(target)
        target_ok[:, target_index] &= np.isfinite(values[:, feature_index])
    return times, values, target_ok


def _calendar_features(times: pd.DatetimeIndex) -> tuple[np.ndarray, tuple[str, ...]]:
    hour = times.hour.to_numpy(float) + times.minute.to_numpy(float) / 60.0
    day = times.dayofyear.to_numpy(float) - 1.0 + hour / 24.0
    values = np.column_stack(
        (
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day / 365.2425),
            np.cos(2.0 * np.pi * day / 365.2425),
        )
    )
    return values, ("日周期正弦", "日周期余弦", "年周期正弦", "年周期余弦")


def _lag_features(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    rows, feature_count = values.shape
    blocks = []
    names = []
    row_index = np.arange(rows)
    for lag_hours in config.LAG_HOURS:
        lag_steps = lag_hours // config.STEP_HOURS
        source = row_index - lag_steps
        lagged = np.full_like(values, np.nan, dtype=float)
        valid_rows = source >= 0
        lagged[valid_rows] = values[source[valid_rows]]
        mask = np.isfinite(lagged).astype(float)
        blocks.extend((lagged, mask))
        names.extend(f"滞后{lag_hours}小时_{name}" for name in config.INPUT_FEATURES)
        names.extend(
            f"滞后{lag_hours}小时_{name}_可用" for name in config.INPUT_FEATURES
        )
    return np.concatenate(blocks, axis=1), tuple(names)


def _rolling_features(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    frame = pd.DataFrame(values, columns=config.INPUT_FEATURES)
    blocks = []
    names = []
    for window_hours in config.ROLLING_HOURS:
        steps = window_hours // config.STEP_HOURS
        rolling = frame.rolling(window=steps, min_periods=1)
        mean = rolling.mean().to_numpy(float)
        std = rolling.std(ddof=0).to_numpy(float)
        minimum = rolling.min().to_numpy(float)
        maximum = rolling.max().to_numpy(float)
        coverage = rolling.count().to_numpy(float) / float(steps)
        trend = np.full_like(values, np.nan, dtype=float)
        if steps > 1:
            trend[steps - 1 :] = (
                values[steps - 1 :] - values[: -(steps - 1)]
            ) / float(steps - 1)
        statistic_blocks = (
            ("均值", mean),
            ("标准差", std),
            ("最小值", minimum),
            ("最大值", maximum),
            ("端点趋势", trend),
            ("覆盖率", coverage),
        )
        for statistic, block in statistic_blocks:
            blocks.append(block)
            names.extend(
                f"过去{window_hours}小时_{name}_{statistic}"
                for name in config.INPUT_FEATURES
            )
    return np.concatenate(blocks, axis=1), tuple(names)


def multiscale_feature_names() -> tuple[str, ...]:
    dummy_times = pd.date_range("2022-01-01", periods=2, freq="4h")
    dummy = np.full((2, len(config.INPUT_FEATURES)), np.nan, dtype=float)
    _, calendar_names = _calendar_features(dummy_times)
    _, lag_names = _lag_features(dummy)
    _, rolling_names = _rolling_features(dummy)
    return calendar_names + lag_names + rolling_names


def build_station_dataset(panel: pd.DataFrame, station: str) -> dict[str, np.ndarray]:
    """Build paired seven-day inputs and 18 by 5 future labels."""

    times, values, target_ok = _station_grid(panel, station)
    minimum_steps = config.MAX_SEQUENCE_STEPS + config.OUTPUT_STEPS
    if len(times) < minimum_steps:
        raise ValueError(f"站点数据不足，无法建立联合预测窗口: {station}")

    diffs = np.full_like(values, np.nan, dtype=float)
    if len(values) > 1:
        valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        candidate = values[1:] - values[:-1]
        diffs[1:] = np.where(valid, candidate, np.nan)

    origins = np.arange(
        config.MAX_SEQUENCE_STEPS - 1,
        len(times) - config.OUTPUT_STEPS,
        dtype=np.int64,
    )
    sequence_offsets = np.arange(
        config.MAX_SEQUENCE_STEPS - 1, -1, -1, dtype=np.int64
    )
    sequence_rows = origins[:, None] - sequence_offsets[None, :]
    future_rows = origins[:, None] + np.arange(
        1, config.OUTPUT_STEPS + 1, dtype=np.int64
    )[None, :]
    target_feature_indices = np.asarray(
        [config.INPUT_FEATURES.index(target) for target in config.TARGETS],
        dtype=np.int64,
    )

    current = values[origins][:, target_feature_indices]
    future = values[future_rows][:, :, target_feature_indices]
    current_mask = target_ok[origins] & np.isfinite(current)
    future_mask = target_ok[future_rows] & np.isfinite(future)
    label_mask = current_mask[:, None, :] & future_mask

    calendar, _ = _calendar_features(times)
    lags, _ = _lag_features(values)
    rolling, _ = _rolling_features(values)
    multiscale = np.concatenate((calendar, lags, rolling), axis=1)[origins]

    return {
        "x_raw": values[sequence_rows],
        "x_diff": diffs[sequence_rows],
        "x_raw_mask": np.isfinite(values[sequence_rows]),
        "x_diff_mask": np.isfinite(diffs[sequence_rows]),
        "x_multiscale": multiscale,
        "current": current,
        "current_mask": current_mask,
        "y_abs": future,
        "y_delta": future - current[:, None, :],
        "y_mask": label_mask,
        "target_start": times[future_rows[:, 0]].to_numpy(dtype="datetime64[ns]"),
        "target_end": times[future_rows[:, -1]].to_numpy(dtype="datetime64[ns]"),
    }


def subset(dataset: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    return {key: np.asarray(value)[mask] for key, value in dataset.items()}


def split_by_time(dataset: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    target_start = pd.to_datetime(dataset["target_start"])
    target_end = pd.to_datetime(dataset["target_end"])
    masks = {
        "train": target_end < pd.Timestamp(config.TRAIN_END),
        "val": (target_start >= pd.Timestamp(config.TRAIN_END))
        & (target_end < pd.Timestamp(config.VAL_END)),
        "test": target_start >= pd.Timestamp(config.VAL_END),
    }
    return {name: subset(dataset, np.asarray(mask)) for name, mask in masks.items()}


def internal_time_split(
    train: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split the train block causally for epoch selection without using 2024."""

    target_start = pd.to_datetime(train["target_start"])
    target_end = pd.to_datetime(train["target_end"])
    boundary = pd.Timestamp(config.INTERNAL_VAL_START)
    fit_mask = np.asarray(target_end < boundary)
    validation_mask = np.asarray(
        (target_start >= boundary) & (target_end < pd.Timestamp(config.TRAIN_END))
    )
    return subset(train, fit_mask), subset(train, validation_mask)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray | None = None) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        approved = np.where(valid, values, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(approved, axis=0)
            scale = np.nanstd(approved, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def prepare_inputs(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if context not in config.CONTEXTS:
        raise ValueError(f"未知输入尺度: {context}")
    steps = config.CONTEXT_STEPS[context]
    raw_scaler = Standardizer.fit(
        np.asarray(train["x_raw"], dtype=float).reshape(-1, len(config.INPUT_FEATURES))
    )
    diff_scaler = Standardizer.fit(
        np.asarray(train["x_diff"], dtype=float).reshape(-1, len(config.INPUT_FEATURES))
    )

    def sequence(split: dict[str, np.ndarray]) -> np.ndarray:
        raw = raw_scaler.transform(split["x_raw"][:, -steps:])
        diff = diff_scaler.transform(split["x_diff"][:, -steps:])
        raw_mask = np.asarray(split["x_raw_mask"][:, -steps:], dtype=float)
        diff_mask = np.asarray(split["x_diff_mask"][:, -steps:], dtype=float)
        return np.nan_to_num(
            np.concatenate((raw, diff, raw_mask, diff_mask), axis=2),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    target_feature_indices = np.asarray(
        [config.INPUT_FEATURES.index(target) for target in config.TARGETS]
    )

    def current_context(split: dict[str, np.ndarray]) -> np.ndarray:
        current = (
            np.asarray(split["current"], dtype=float)
            - raw_scaler.mean[target_feature_indices]
        ) / raw_scaler.scale[target_feature_indices]
        return np.nan_to_num(
            np.concatenate(
                (current, np.asarray(split["current_mask"], dtype=float)), axis=1
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    train_context = current_context(train)
    evaluation_context = current_context(evaluation)
    auxiliary_scaler = None
    if context == "multiscale":
        auxiliary_scaler = Standardizer.fit(train["x_multiscale"])
        train_auxiliary = np.nan_to_num(
            auxiliary_scaler.transform(train["x_multiscale"]), nan=0.0
        )
        evaluation_auxiliary = np.nan_to_num(
            auxiliary_scaler.transform(evaluation["x_multiscale"]), nan=0.0
        )
        train_context = np.concatenate((train_context, train_auxiliary), axis=1)
        evaluation_context = np.concatenate(
            (evaluation_context, evaluation_auxiliary), axis=1
        )

    scaler_arrays = {
        "raw_mean": raw_scaler.mean,
        "raw_scale": raw_scaler.scale,
        "diff_mean": diff_scaler.mean,
        "diff_scale": diff_scaler.scale,
    }
    if auxiliary_scaler is not None:
        scaler_arrays["auxiliary_mean"] = auxiliary_scaler.mean
        scaler_arrays["auxiliary_scale"] = auxiliary_scaler.scale
    return (
        sequence(train),
        train_context.astype(np.float32),
        sequence(evaluation),
        evaluation_context.astype(np.float32),
        scaler_arrays,
    )


def warning_thresholds(train: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(config.TARGETS), np.nan, dtype=float)
    upper = np.full(len(config.TARGETS), np.nan, dtype=float)
    current = np.asarray(train["current"], dtype=float)
    mask = np.asarray(train["current_mask"], dtype=bool) & np.isfinite(current)
    q = config.WARNING_QUANTILE
    for target_index, target in enumerate(config.TARGETS):
        values = current[mask[:, target_index], target_index]
        if not len(values):
            continue
        direction = config.WARNING_DIRECTIONS[target]
        if direction in {"lower", "two_sided"}:
            lower[target_index] = float(np.quantile(values, q))
        if direction in {"upper", "two_sided"}:
            upper[target_index] = float(np.quantile(values, 1.0 - q))
    return lower, upper
