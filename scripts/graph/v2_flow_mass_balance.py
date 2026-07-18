#!/usr/bin/env python3
"""Causal flow weights for V2 graph-message experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.graph import v2_delayed_step_graph as delayed_graph

WEIGHT_MODES = ("unweighted", "branch_normalized", "mass_balance")


@dataclass(frozen=True)
class FlowWeightResult:
    weights: np.ndarray
    unobserved_fraction: np.ndarray
    valid: np.ndarray


def compute_flow_weights(
    source_flow: np.ndarray,
    downstream_flow: np.ndarray,
    mode: str,
) -> FlowWeightResult:
    """Compute edge weights while preserving invalid-row diagnostics."""
    source = np.asarray(source_flow, dtype=float)
    downstream = np.asarray(downstream_flow, dtype=float)
    if source.ndim != 2 or downstream.shape != (len(source),):
        raise ValueError("source_flow must be 2D and downstream_flow must match its rows")
    if mode not in WEIGHT_MODES:
        raise ValueError(f"Unsupported flow-weight mode: {mode}")

    valid = (
        np.isfinite(source).all(axis=1)
        & (source >= 0.0).all(axis=1)
        & np.isfinite(downstream)
        & (downstream > 0.0)
        & (source.sum(axis=1) > 0.0)
    )
    weights = np.zeros_like(source, dtype=float)
    unobserved = np.zeros(len(source), dtype=float)
    if not np.any(valid):
        return FlowWeightResult(weights, unobserved, valid)

    if mode == "unweighted":
        weights[valid] = 1.0
    elif mode == "branch_normalized":
        denominator = source[valid].sum(axis=1)
        weights[valid] = source[valid] / denominator[:, None]
    else:
        source_sum = source[valid].sum(axis=1)
        denominator = np.maximum(downstream[valid], source_sum)
        weights[valid] = source[valid] / denominator[:, None]
        unobserved[valid] = np.maximum(0.0, 1.0 - weights[valid].sum(axis=1))
    return FlowWeightResult(weights, unobserved, valid)


def causal_daily_values(
    daily: pd.DataFrame,
    origin_times,
    columns: tuple[str, ...],
    lag_days: int = 1,
    smooth_days: int = 3,
) -> np.ndarray:
    """Look up rolling daily values ending strictly before each forecast origin."""
    if int(lag_days) < 1 or int(smooth_days) < 1:
        raise ValueError("lag_days and smooth_days must be positive")
    missing = [column for column in columns if column not in daily.columns]
    if missing:
        raise ValueError(f"Missing daily-flow columns: {missing}")

    frame = daily.loc[:, list(columns)].copy()
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame.sort_index().rolling(int(smooth_days), min_periods=int(smooth_days)).median()
    lookup_dates = pd.DatetimeIndex(pd.to_datetime(origin_times)).normalize() - pd.Timedelta(days=int(lag_days))
    return frame.reindex(lookup_dates).to_numpy(dtype=float)


def make_multi_source_delayed_mapper(
    torch,
    source_dim: int,
    target_dim: int,
    lag_counts: tuple[int, ...],
):
    """Create edge-specific delayed maps whose step messages receive flow weights."""

    class _MultiSourceDelayedMapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.edge_mappers = torch.nn.ModuleList(
                [
                    delayed_graph.make_delayed_step_mapper(
                        torch,
                        source_dim=int(source_dim),
                        target_dim=int(target_dim),
                        lag_count=int(lag_count),
                    )
                    for lag_count in lag_counts
                ]
            )

        def forward(self, aligned_sources, edge_weights):
            if len(aligned_sources) != len(self.edge_mappers):
                raise ValueError("aligned_sources must contain one tensor per edge")
            if edge_weights.ndim != 2 or edge_weights.shape[1] != len(self.edge_mappers):
                raise ValueError("edge_weights must have shape (batch, edge_count)")
            weighted_steps = []
            for edge_idx, (mapper, aligned) in enumerate(zip(self.edge_mappers, aligned_sources)):
                step, _ = mapper(aligned, accumulate=False)
                weighted_steps.append(step * edge_weights[:, edge_idx, None, None])
            combined_step = torch.stack(weighted_steps, dim=0).sum(dim=0)
            return combined_step, delayed_graph.cumulative_step_correction(combined_step)

    return _MultiSourceDelayedMapper()
