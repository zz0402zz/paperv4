#!/usr/bin/env python3
"""Event-gated delayed graph messages for the focused Fushidu experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.graph import v2_delayed_step_graph as delayed_graph

@dataclass(frozen=True)
class EventGateThresholds:
    source_q95: np.ndarray
    target_q50: np.ndarray


def fit_event_thresholds(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    train_idx: np.ndarray,
) -> EventGateThresholds:
    """Fit feature-wise shock and quiet thresholds from training rows only."""
    source = np.abs(np.asarray(source_delta, dtype=float)[train_idx])
    target = np.abs(np.asarray(target_delta, dtype=float)[train_idx])
    if source.ndim != 2 or target.shape != source.shape:
        raise ValueError("source_delta and target_delta must have matching 2-D shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("event-threshold training values must be finite")
    return EventGateThresholds(
        source_q95=np.quantile(source, 0.95, axis=0),
        target_q50=np.quantile(target, 0.50, axis=0),
    )


def event_gate(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    thresholds: EventGateThresholds,
) -> np.ndarray:
    """Open a target-feature gate for an upstream shock unseen downstream."""
    source = np.asarray(source_delta, dtype=float)
    target = np.asarray(target_delta, dtype=float)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source_delta and target_delta must have matching 2-D shapes")
    return (
        np.isfinite(source)
        & np.isfinite(target)
        & (np.abs(source) > thresholds.source_q95[None, :])
        & (np.abs(target) <= thresholds.target_q50[None, :])
    )


def shuffle_by_split(
    values: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
    block_steps: int = 6,
) -> np.ndarray:
    """Shuffle blocks independently inside train/validation/test partitions."""
    output = np.asarray(values).copy()
    rng = np.random.default_rng(int(seed))
    for name in ("train", "val", "test"):
        idx = np.asarray(split[name], dtype=int)
        if len(idx) <= 1:
            continue
        blocks = [idx[start : start + int(block_steps)] for start in range(0, len(idx), int(block_steps))]
        order = rng.permutation(len(blocks))
        shuffled_idx = np.concatenate([blocks[position] for position in order])
        output[idx] = values[shuffled_idx]
    return output


def arrival_feature_gates(
    origin_gates: np.ndarray,
    supports: tuple[tuple[int, ...], ...],
    output_steps: int,
) -> np.ndarray:
    """Place each origin-time event only at its candidate arrival horizons."""
    origin = np.asarray(origin_gates, dtype=bool)
    if origin.ndim != 3 or origin.shape[1] != len(supports):
        raise ValueError("origin_gates must have shape (sample, edge, target_feature)")
    output = np.zeros((origin.shape[0], origin.shape[1], int(output_steps), origin.shape[2]), dtype=bool)
    for edge_idx, support in enumerate(supports):
        for lag in support:
            if 1 <= int(lag) <= int(output_steps):
                output[:, edge_idx, int(lag) - 1, :] = origin[:, edge_idx, :]
    return output


def current_event_pulse(aligned_source: np.ndarray, support: tuple[int, ...]) -> np.ndarray:
    """Keep the origin-time source change at each candidate arrival only."""
    aligned = np.asarray(aligned_source)
    if aligned.ndim != 4 or aligned.shape[2] != len(support):
        raise ValueError("aligned_source must have shape (sample, horizon, lag, feature)")
    output = np.zeros_like(aligned)
    for lag_idx, lag in enumerate(support):
        if 1 <= int(lag) <= aligned.shape[1]:
            output[:, int(lag) - 1, lag_idx, :] = aligned[:, int(lag) - 1, lag_idx, :]
    return output


def make_event_gated_multi_source_mapper(
    torch,
    source_dim: int,
    target_dim: int,
    lag_counts: tuple[int, ...],
):
    """Create edge-specific delayed maps switched by causal feature gates."""

    class _EventGatedMultiSourceMapper(torch.nn.Module):
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

        def forward(self, aligned_sources, feature_gates):
            if len(aligned_sources) != len(self.edge_mappers):
                raise ValueError("aligned_sources must contain one tensor per edge")
            if feature_gates.ndim != 4:
                raise ValueError(
                    "feature_gates must have shape (batch, edge, horizon, target_feature)"
                )
            if feature_gates.shape[1] != len(self.edge_mappers):
                raise ValueError("feature_gates edge axis does not match the graph")
            edge_steps = []
            for edge_idx, (mapper, aligned) in enumerate(zip(self.edge_mappers, aligned_sources)):
                step, _ = mapper(aligned, accumulate=False)
                edge_steps.append(step * feature_gates[:, edge_idx, :, :])
            combined_step = torch.stack(edge_steps, dim=0).sum(dim=0)
            return combined_step, delayed_graph.cumulative_step_correction(combined_step)

    return _EventGatedMultiSourceMapper()
