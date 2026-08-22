"""Quality sidecars and leakage-safe transformations for preprocessing ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data as base_data
from scripts.multitarget_forecasting import preprocessing_ablation_config as config


def enrich_station_dataset(
    panel: pd.DataFrame, station: str
) -> dict[str, np.ndarray]:
    """Attach soft-suspect flags without changing the frozen base dataset."""

    dataset = base_data.build_station_dataset(panel, station)
    frame = panel.loc[panel["station"].astype(str).eq(str(station))].copy()
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    start = pd.Timestamp(frame["time"].min()).ceil(f"{base_config.STEP_HOURS}h")
    end = pd.Timestamp(frame["time"].max()).floor(f"{base_config.STEP_HOURS}h")
    times = pd.date_range(start, end, freq=f"{base_config.STEP_HOURS}h")
    indexed = frame.set_index("time").reindex(times)

    soft = np.zeros((len(times), len(base_config.INPUT_FEATURES)), dtype=bool)
    for feature_index, feature in enumerate(base_config.INPUT_FEATURES):
        column = f"{feature}__soft_suspect"
        if column in indexed:
            soft[:, feature_index] = indexed[column].fillna(False).to_numpy(bool)

    target_start = pd.to_datetime(dataset["target_start"])
    current_times = target_start - pd.Timedelta(hours=base_config.STEP_HOURS)
    origins = times.get_indexer(current_times)
    if (origins < 0).any():
        raise ValueError(f"质量标记无法与联合预测窗口对齐: {station}")
    sequence_offsets = np.arange(
        base_config.MAX_SEQUENCE_STEPS - 1, -1, -1, dtype=np.int64
    )
    sequence_rows = origins[:, None] - sequence_offsets[None, :]
    future_rows = origins[:, None] + np.arange(
        1, base_config.OUTPUT_STEPS + 1, dtype=np.int64
    )[None, :]
    target_indices = np.asarray(
        [base_config.INPUT_FEATURES.index(target) for target in base_config.TARGETS],
        dtype=np.int64,
    )
    dataset["x_soft_suspect"] = soft[sequence_rows]
    dataset["current_soft_suspect"] = soft[origins][:, target_indices]
    dataset["future_soft_suspect"] = soft[future_rows][:, :, target_indices]
    dataset["quality_y_mask"] = (
        np.asarray(dataset["y_mask"], dtype=bool)
        & ~dataset["current_soft_suspect"][:, None, :]
        & ~dataset["future_soft_suspect"]
    )
    return dataset


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls, values: np.ndarray, mask: np.ndarray | None = None
    ) -> "RobustScaler":
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        approved = np.where(valid, values, np.nan)
        with np.errstate(all="ignore"):
            center = np.nanmedian(approved, axis=0)
            lower = np.nanquantile(approved, 0.25, axis=0)
            upper = np.nanquantile(approved, 0.75, axis=0)
        scale = upper - lower
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(center=center, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.center) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.center


@dataclass(frozen=True)
class LogFeatureTransform:
    scales: np.ndarray
    enabled: bool

    @classmethod
    def fit(cls, train_raw: np.ndarray, enabled: bool) -> "LogFeatureTransform":
        scales = np.ones(len(base_config.INPUT_FEATURES), dtype=float)
        if enabled:
            flattened = np.asarray(train_raw, dtype=float).reshape(
                -1, len(base_config.INPUT_FEATURES)
            )
            for target in config.LOG_TARGETS:
                index = base_config.INPUT_FEATURES.index(target)
                values = flattened[:, index]
                positive = values[np.isfinite(values) & (values > 0)]
                if len(positive):
                    candidate = float(np.median(positive))
                    scales[index] = candidate if candidate > 0 else 1.0
        return cls(scales=scales, enabled=enabled)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        result = np.asarray(raw, dtype=float).copy()
        if not self.enabled:
            return result
        for target in config.LOG_TARGETS:
            index = base_config.INPUT_FEATURES.index(target)
            valid = np.isfinite(result[..., index]) & (result[..., index] >= 0)
            result[..., index] = np.where(
                valid,
                np.log1p(np.maximum(result[..., index], 0.0) / self.scales[index]),
                np.nan,
            )
        return result


@dataclass(frozen=True)
class TargetTransform:
    log_scales: np.ndarray
    log_enabled: bool

    @classmethod
    def fit(cls, split: dict[str, np.ndarray], enabled: bool) -> "TargetTransform":
        scales = np.ones(len(config.TARGETS), dtype=float)
        if enabled:
            current = np.asarray(split["current"], dtype=float)
            future = np.asarray(split["y_abs"], dtype=float)
            current_mask = np.asarray(split["current_mask"], dtype=bool)
            future_mask = np.asarray(split["y_mask"], dtype=bool)
            for target in config.LOG_TARGETS:
                index = config.TARGETS.index(target)
                values = np.concatenate(
                    (
                        current[current_mask[:, index], index],
                        future[:, :, index][future_mask[:, :, index]],
                    )
                )
                positive = values[np.isfinite(values) & (values > 0)]
                if len(positive):
                    candidate = float(np.median(positive))
                    scales[index] = candidate if candidate > 0 else 1.0
        return cls(log_scales=scales, log_enabled=enabled)

    def _absolute(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        if not self.log_enabled:
            return result
        for target in config.LOG_TARGETS:
            index = config.TARGETS.index(target)
            valid = np.isfinite(result[..., index]) & (result[..., index] >= 0)
            result[..., index] = np.where(
                valid,
                np.log1p(np.maximum(result[..., index], 0.0) / self.log_scales[index]),
                np.nan,
            )
        return result

    def training_values(
        self, split: dict[str, np.ndarray], quality_aware: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        absolute = self._absolute(split["y_abs"])
        current = self._absolute(split["current"])
        values = np.empty_like(absolute)
        for index, target in enumerate(config.TARGETS):
            if config.TARGET_OUTPUT_MODES[target] == "delta":
                values[:, :, index] = absolute[:, :, index] - current[:, None, index]
            else:
                values[:, :, index] = absolute[:, :, index]
        mask_key = "quality_y_mask" if quality_aware else "y_mask"
        mask = np.asarray(split[mask_key], dtype=bool) & np.isfinite(values)
        return values, mask

    def to_absolute(
        self, transformed_prediction: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        prediction = np.asarray(transformed_prediction, dtype=float).copy()
        transformed_current = self._absolute(current)
        for index, target in enumerate(config.TARGETS):
            if config.TARGET_OUTPUT_MODES[target] == "delta":
                prediction[:, :, index] += transformed_current[:, None, index]
            if self.log_enabled and target in config.LOG_TARGETS:
                prediction[:, :, index] = (
                    np.expm1(np.clip(prediction[:, :, index], -30.0, 30.0))
                    * self.log_scales[index]
                )
                prediction[:, :, index] = np.maximum(prediction[:, :, index], 0.0)
        return prediction


def _elapsed_missing(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    elapsed = np.zeros(mask.shape, dtype=float)
    for step in range(mask.shape[1]):
        if step == 0:
            elapsed[:, step, :] = (~mask[:, step, :]).astype(float)
        else:
            elapsed[:, step, :] = np.where(
                mask[:, step, :], 0.0, elapsed[:, step - 1, :] + 1.0
            )
    return np.minimum(elapsed, base_config.MAX_SEQUENCE_STEPS) / float(
        base_config.MAX_SEQUENCE_STEPS
    )


def prepare_inputs(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    *,
    log_targets: bool,
    quality_aware: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    steps = base_config.CONTEXT_STEPS[config.CONTEXT]
    feature_transform = LogFeatureTransform.fit(train["x_raw"], log_targets)
    train_raw = feature_transform.transform(train["x_raw"])
    evaluation_raw = feature_transform.transform(evaluation["x_raw"])

    def differences(raw: np.ndarray) -> np.ndarray:
        result = np.full_like(raw, np.nan, dtype=float)
        if raw.shape[1] > 1:
            valid = np.isfinite(raw[:, 1:]) & np.isfinite(raw[:, :-1])
            result[:, 1:] = np.where(valid, raw[:, 1:] - raw[:, :-1], np.nan)
        return result

    train_diff = differences(train_raw)
    evaluation_diff = differences(evaluation_raw)
    raw_scaler = RobustScaler.fit(
        train_raw.reshape(-1, len(base_config.INPUT_FEATURES))
    )
    diff_scaler = RobustScaler.fit(
        train_diff.reshape(-1, len(base_config.INPUT_FEATURES))
    )

    def sequence(split: dict[str, np.ndarray], raw: np.ndarray, diff: np.ndarray) -> np.ndarray:
        raw_mask = np.isfinite(raw)
        diff_mask = np.isfinite(diff)
        blocks = [
            raw_scaler.transform(raw)[:, -steps:],
            diff_scaler.transform(diff)[:, -steps:],
            raw_mask[:, -steps:].astype(float),
            diff_mask[:, -steps:].astype(float),
        ]
        if quality_aware:
            blocks.extend(
                (
                    np.asarray(split["x_soft_suspect"][:, -steps:], dtype=float),
                    _elapsed_missing(raw_mask)[:, -steps:],
                )
            )
        return np.nan_to_num(
            np.concatenate(blocks, axis=2), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    target_feature_indices = np.asarray(
        [base_config.INPUT_FEATURES.index(target) for target in config.TARGETS]
    )

    def current_context(split: dict[str, np.ndarray]) -> np.ndarray:
        current_full = np.asarray(split["current"], dtype=float).copy()
        if log_targets:
            for target in config.LOG_TARGETS:
                target_index = config.TARGETS.index(target)
                feature_index = base_config.INPUT_FEATURES.index(target)
                current_full[:, target_index] = np.log1p(
                    np.maximum(current_full[:, target_index], 0.0)
                    / feature_transform.scales[feature_index]
                )
        current = (
            current_full - raw_scaler.center[target_feature_indices]
        ) / raw_scaler.scale[target_feature_indices]
        blocks = [current, np.asarray(split["current_mask"], dtype=float)]
        if quality_aware:
            blocks.append(np.asarray(split["current_soft_suspect"], dtype=float))
        return np.nan_to_num(
            np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    scalers = {
        "raw_center": raw_scaler.center,
        "raw_scale": raw_scaler.scale,
        "diff_center": diff_scaler.center,
        "diff_scale": diff_scaler.scale,
        "input_log_scales": feature_transform.scales,
    }
    return (
        sequence(train, train_raw, train_diff),
        current_context(train),
        sequence(evaluation, evaluation_raw, evaluation_diff),
        current_context(evaluation),
        scalers,
    )
