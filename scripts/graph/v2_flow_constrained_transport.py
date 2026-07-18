#!/usr/bin/env python3
"""Physically constrained same-feature transport for V2 flow graph experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.graph import v2_delayed_step_graph as delayed_graph

def make_constrained_transport(
    torch,
    target_input_indices: tuple[int, ...],
    target_scale: np.ndarray,
    lag_counts: tuple[int, ...],
    initial_retention: float = 0.05,
):
    """Create edge-specific lag kernels and bounded same-feature retention."""
    target_indices = np.asarray(target_input_indices, dtype=np.int64)
    scale = np.asarray(target_scale, dtype=np.float32)
    if target_indices.ndim != 1 or scale.shape != target_indices.shape:
        raise ValueError("target indices and target scale must be aligned vectors")
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("target scale must be finite and positive")
    if not lag_counts or min(lag_counts) < 1:
        raise ValueError("lag_counts must be positive")
    if not 0.0 < float(initial_retention) < 1.0:
        raise ValueError("initial_retention must be between zero and one")

    class _ConstrainedTransport(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("target_indices", torch.as_tensor(target_indices, dtype=torch.long))
            self.register_buffer("target_scale", torch.as_tensor(scale, dtype=torch.float32))
            self.lag_logits = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(int(count))) for count in lag_counts]
            )
            initial_logit = float(np.log(initial_retention / (1.0 - initial_retention)))
            self.retention_logits = torch.nn.Parameter(
                torch.full((len(lag_counts), len(target_indices)), initial_logit)
            )

        def lag_weights(self, edge_idx: int):
            return torch.softmax(self.lag_logits[int(edge_idx)], dim=0)

        def retention(self):
            return torch.sigmoid(self.retention_logits)

        def forward(self, aligned_sources, edge_weights):
            if len(aligned_sources) != len(self.lag_logits):
                raise ValueError("aligned_sources must contain one tensor per edge")
            if edge_weights.ndim != 2 or edge_weights.shape[1] != len(self.lag_logits):
                raise ValueError("edge_weights must have shape (batch, edge_count)")
            steps = []
            retention = self.retention()
            for edge_idx, aligned in enumerate(aligned_sources):
                if aligned.shape[-2] != self.lag_logits[edge_idx].numel():
                    raise ValueError("aligned lag axis does not match the edge lag kernel")
                mixed = (
                    aligned * self.lag_weights(edge_idx)[None, None, :, None]
                ).sum(dim=-2)
                target_change = mixed.index_select(-1, self.target_indices)
                raw_step = target_change * retention[edge_idx][None, None, :]
                raw_step = raw_step * edge_weights[:, edge_idx, None, None]
                steps.append(raw_step)
            combined_raw = torch.stack(steps, dim=0).sum(dim=0)
            scaled_step = combined_raw / self.target_scale[None, None, :]
            return scaled_step, delayed_graph.cumulative_step_correction(scaled_step)

    return _ConstrainedTransport()


def static_train_weights(weights: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Repeat edge-weight medians estimated from training rows only."""
    values = np.asarray(weights, dtype=float)
    selected = values[np.asarray(train_idx, dtype=int)]
    if values.ndim != 2 or selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("weights and training indices must contain finite rows")
    median = np.median(selected, axis=0)
    return np.tile(median, (len(values), 1)).astype(np.float32)


def shuffle_daily_weights(
    weights: np.ndarray,
    origin_times,
    split: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    """Shuffle complete daily weight profiles within each data split."""
    values = np.asarray(weights, dtype=float)
    times = pd.DatetimeIndex(pd.to_datetime(origin_times))
    if values.ndim != 2 or len(values) != len(times):
        raise ValueError("weights and origin_times must have equal rows")
    shuffled = values.copy()
    for offset, name in enumerate(("train", "val", "test")):
        if name not in split:
            continue
        idx = np.asarray(split[name], dtype=int)
        if len(idx) == 0:
            continue
        dates = times[idx].normalize()
        unique_dates = pd.Index(dates.unique())
        rng = np.random.default_rng(int(seed) + offset)
        permuted_dates = unique_dates[rng.permutation(len(unique_dates))]
        source_by_date = {
            date: values[idx[np.flatnonzero(dates == date)[0]]]
            for date in unique_dates
        }
        for target_date, source_date in zip(unique_dates, permuted_dates):
            target_rows = idx[np.flatnonzero(dates == target_date)]
            shuffled[target_rows] = source_by_date[source_date]
    return shuffled.astype(np.float32)
