"""Leakage-safe data preparation shared by all teacher candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import data as base_data
from scripts.multitarget_forecasting import preprocessing_ablation_data as prep_data
from scripts.teacher_screening import config


@dataclass(frozen=True)
class PreparedFold:
    fit_flat: np.ndarray
    prediction_flat: np.ndarray
    fit_sequence: np.ndarray
    prediction_sequence: np.ndarray
    fit_context: np.ndarray
    prediction_context: np.ndarray
    fit_scaled_target: np.ndarray
    fit_target_mask: np.ndarray
    target_transform: prep_data.TargetTransform
    target_scaler: prep_data.RobustScaler
    prediction_current: np.ndarray

    def to_absolute(self, prediction_scaled: np.ndarray) -> np.ndarray:
        transformed = self.target_scaler.inverse_transform(prediction_scaled)
        return self.target_transform.to_absolute(
            transformed, self.prediction_current
        )


def load_training_panel() -> pd.DataFrame:
    """Load development data, while downstream code only materializes train rows."""

    return base_data.load_development_panel()


def build_training_dataset(panel: pd.DataFrame, station: str) -> dict[str, np.ndarray]:
    enriched = prep_data.enrich_station_dataset(panel, station)
    return base_data.split_by_time(enriched)["train"]


def causal_oof_folds(train: dict[str, np.ndarray]) -> tuple[dict[str, object], ...]:
    """Expanding folds separated by the complete 72-hour label extent."""

    target_start = pd.to_datetime(train["target_start"])
    target_end = pd.to_datetime(train["target_end"])
    folds: list[dict[str, object]] = []
    for index, (name, prediction_start, prediction_end) in enumerate(config.OOF_FOLDS):
        start = pd.Timestamp(prediction_start)
        end = pd.Timestamp(prediction_end)
        fit_mask = np.asarray(target_end < start)
        prediction_mask = np.asarray(
            (target_start >= start) & (target_end < end)
        )
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


def prepare_fold(
    train: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    prediction_mask: np.ndarray,
    horizon_indices: tuple[int, ...],
) -> PreparedFold:
    """Fit every scaler only on the causal fit block of the current fold."""

    fit = base_data.subset(train, np.asarray(fit_mask, dtype=bool))
    prediction = base_data.subset(train, np.asarray(prediction_mask, dtype=bool))
    if not len(fit["target_start"]) or not len(prediction["target_start"]):
        raise ValueError("教师OOF折的拟合区间或预测区间为空。")

    fit_sequence, fit_context, pred_sequence, pred_context, _ = (
        prep_data.prepare_inputs(
            fit,
            prediction,
            log_targets=True,
            quality_aware=False,
        )
    )
    transformer = prep_data.TargetTransform.fit(fit, enabled=True)
    fit_values, fit_value_mask = transformer.training_values(
        fit, quality_aware=False
    )
    fit_values = fit_values[:, horizon_indices, :]
    fit_value_mask = fit_value_mask[:, horizon_indices, :]
    target_scaler = prep_data.RobustScaler.fit(fit_values, fit_value_mask)
    fit_scaled = np.nan_to_num(
        target_scaler.transform(fit_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    def flatten(sequence: np.ndarray, context: np.ndarray) -> np.ndarray:
        return np.concatenate((sequence.reshape(len(sequence), -1), context), axis=1).astype(
            np.float32
        )

    return PreparedFold(
        fit_flat=flatten(fit_sequence, fit_context),
        prediction_flat=flatten(pred_sequence, pred_context),
        fit_sequence=fit_sequence,
        prediction_sequence=pred_sequence,
        fit_context=fit_context,
        prediction_context=pred_context,
        fit_scaled_target=fit_scaled,
        fit_target_mask=np.asarray(fit_value_mask, dtype=bool),
        target_transform=transformer,
        target_scaler=target_scaler,
        prediction_current=np.asarray(prediction["current"], dtype=float),
    )


def horizon_indices(hours: tuple[int, ...]) -> tuple[int, ...]:
    unknown = set(hours).difference(config.ALL_HOURS)
    if unknown:
        raise ValueError(f"时距必须来自4、8、…、72小时: {sorted(unknown)}")
    return tuple(config.ALL_HOURS.index(hour) for hour in hours)


def oof_truth(
    train: dict[str, np.ndarray], horizon_indices_: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(train["y_abs"], dtype=float)[:, horizon_indices_, :],
        np.asarray(train["y_mask"], dtype=bool)[:, horizon_indices_, :],
        np.asarray(train["current"], dtype=float),
    )
