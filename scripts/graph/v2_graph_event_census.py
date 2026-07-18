#!/usr/bin/env python3
"""Causal event definitions for the V2 full-graph census."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventThresholds:
    source_shock_abs: float
    target_quiet_abs: float
    target_response_abs: float
    source_control_abs: float = 0.0


def fit_event_thresholds(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    train_idx: np.ndarray,
) -> EventThresholds:
    """Fit all event thresholds from finite training rows only."""
    source = np.abs(np.asarray(source_delta, dtype=float)[train_idx])
    target = np.abs(np.asarray(target_delta, dtype=float)[train_idx])
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]
    if len(source) < 20 or len(target) < 20:
        raise ValueError("At least 20 finite training changes are required")
    return EventThresholds(
        source_shock_abs=float(np.quantile(source, 0.95)),
        target_quiet_abs=float(np.quantile(target, 0.50)),
        target_response_abs=float(np.quantile(target, 0.75)),
        source_control_abs=float(np.quantile(source, 0.50)),
    )


def event_flags(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    thresholds: EventThresholds,
) -> np.ndarray:
    source = np.asarray(source_delta, dtype=float)
    target = np.asarray(target_delta, dtype=float)
    return (
        np.isfinite(source)
        & np.isfinite(target)
        & (np.abs(source) > thresholds.source_shock_abs)
        & (np.abs(target) <= thresholds.target_quiet_abs)
    )


def control_flags(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    thresholds: EventThresholds,
) -> np.ndarray:
    """Select moderate source changes under the same quiet downstream condition."""
    source = np.asarray(source_delta, dtype=float)
    target = np.asarray(target_delta, dtype=float)
    source_abs = np.abs(source)
    return (
        np.isfinite(source)
        & np.isfinite(target)
        & (source_abs >= thresholds.source_control_abs)
        & (source_abs <= thresholds.source_shock_abs)
        & (np.abs(target) <= thresholds.target_quiet_abs)
    )


def delayed_response(
    future_step_delta: np.ndarray,
    source_delta: np.ndarray,
    response_threshold: float,
    support: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect same-direction downstream responses in a fixed physical lag window."""
    future = np.asarray(future_step_delta, dtype=float)
    source = np.asarray(source_delta, dtype=float)
    if future.ndim != 2 or source.shape != (len(future),):
        raise ValueError("future steps must be 2D and source_delta must match rows")
    steps = np.asarray(support, dtype=int)
    if len(steps) == 0 or np.any(steps < 1) or np.any(steps > future.shape[1]):
        raise ValueError("support must reference available one-indexed future steps")
    selected = future[:, steps - 1]
    same_direction = np.sign(selected) == np.sign(source)[:, None]
    qualifies = np.isfinite(selected) & same_direction & (np.abs(selected) >= float(response_threshold))
    response = qualifies.any(axis=1)
    first_step = np.full(len(future), np.nan, dtype=float)
    max_signed = np.full(len(future), np.nan, dtype=float)
    for row_idx in range(len(future)):
        positions = np.flatnonzero(qualifies[row_idx])
        if positions.size:
            first_step[row_idx] = float(steps[positions[0]])
            values = selected[row_idx, positions]
            max_signed[row_idx] = float(values[np.argmax(np.abs(values))])
    return response, first_step, max_signed


def causal_flow_features(daily_flow: pd.Series, origin_times) -> dict[str, np.ndarray]:
    """Use only the two complete calendar days before each forecast origin."""
    series = pd.to_numeric(daily_flow, errors="coerce").copy()
    series.index = pd.to_datetime(series.index).normalize()
    series = series.sort_index()
    dates = pd.DatetimeIndex(pd.to_datetime(origin_times)).normalize()
    previous = series.reindex(dates - pd.Timedelta(days=1)).to_numpy(dtype=float)
    previous_two = series.reindex(dates - pd.Timedelta(days=2)).to_numpy(dtype=float)
    rise = np.log1p(previous) - np.log1p(previous_two)
    return {"flow_previous_day": previous, "flow_log_rise": rise}


def blind_test_response(
    response: np.ndarray,
    first_step: np.ndarray,
    max_signed: np.ndarray,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(split_name) != "test":
        return response, first_step, max_signed
    count = len(response)
    hidden = np.full(count, np.nan, dtype=float)
    return hidden.copy(), hidden.copy(), hidden.copy()


def positive_uplift_pvalue(
    event_success: int,
    event_count: int,
    control_success: int,
    control_count: int,
) -> float:
    """One-sided pooled two-proportion z-test for event response uplift."""
    n_event = int(event_count)
    n_control = int(control_count)
    if n_event <= 0 or n_control <= 0:
        return float("nan")
    p_event = float(event_success) / n_event
    p_control = float(control_success) / n_control
    pooled = float(event_success + control_success) / (n_event + n_control)
    variance = pooled * (1.0 - pooled) * (1.0 / n_event + 1.0 / n_control)
    if variance <= 0.0:
        return 1.0
    z_value = (p_event - p_control) / np.sqrt(variance)
    return float(0.5 * erfc(z_value / sqrt(2.0)))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return false-discovery-rate adjusted q-values while preserving NaNs."""
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if not len(finite_idx):
        return output
    finite_values = values[finite_idx]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(count, dtype=float)
    restored[order] = adjusted
    output[finite_idx] = restored
    return output


def thin_event_origins(
    mask: np.ndarray,
    origin_times,
    split_labels: np.ndarray,
    min_gap_hours: int = 24,
) -> np.ndarray:
    """Keep the first eligible origin in each split-local event episode."""
    selected = np.asarray(mask, dtype=bool)
    times = pd.DatetimeIndex(pd.to_datetime(origin_times))
    labels = np.asarray(split_labels).astype(str)
    if len(selected) != len(times) or len(labels) != len(times):
        raise ValueError("mask, times, and split labels must have equal rows")
    output = np.zeros(len(selected), dtype=bool)
    gap = pd.Timedelta(hours=int(min_gap_hours))
    for split_name in np.unique(labels):
        last_kept = None
        candidates = np.flatnonzero(selected & (labels == split_name))
        for idx in candidates:
            if last_kept is None or times[idx] - last_kept >= gap:
                output[idx] = True
                last_kept = times[idx]
    return output
