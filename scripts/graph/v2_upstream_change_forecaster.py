#!/usr/bin/env python3
"""Upstream change forecaster and causal trajectory alignment."""

from __future__ import annotations

import numpy as np

from scripts.graph import v2_direct_pair_graph_config as cfg

def align_source_trajectory(
    observed_history: np.ndarray,
    predicted_future: np.ndarray,
    lag_steps: int,
    output_steps: int = cfg.OUTPUT_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Align upstream changes to downstream horizons without true-future leakage.

    `observed_history` is indexed as past-to-present and must end at forecast origin
    t. `predicted_future[0]` is t+1. For downstream horizon h, source step is
    h - lag_steps.
    """
    if lag_steps < 1:
        raise ValueError("lag_steps must be >= 1")
    observed = np.asarray(observed_history, dtype=float)
    predicted = np.asarray(predicted_future, dtype=float)
    if observed.ndim != 2 or predicted.ndim != 2:
        raise ValueError("observed_history and predicted_future must be 2-D arrays")
    if observed.shape[1] != predicted.shape[1]:
        raise ValueError("observed and predicted feature dimensions must match")

    aligned = []
    origins = []
    present_index = observed.shape[0] - 1
    for horizon in range(1, int(output_steps) + 1):
        source_step = horizon - int(lag_steps)
        if source_step <= 0:
            index = present_index + source_step
            if index < 0:
                raise ValueError("observed_history is too short for this lag/output setting")
            aligned.append(observed[index])
            origins.append("observed")
        else:
            index = source_step - 1
            if index >= predicted.shape[0]:
                raise ValueError("predicted_future is too short for this lag/output setting")
            aligned.append(predicted[index])
            origins.append("predicted")
    return np.stack(aligned, axis=0), np.asarray(origins, dtype=object)


def make_upstream_change_forecaster(
    torch,
    feature_count: int = len(cfg.INPUT_FEATURE_COLUMNS),
    hidden_size: int = 64,
    output_steps: int = cfg.OUTPUT_STEPS,
):
    """Create a GRU that predicts named upstream feature changes."""

    class UpstreamChangeForecaster(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_steps = int(output_steps)
            self.feature_count = int(feature_count)
            self.gru = torch.nn.GRU(feature_count, hidden_size, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, output_steps * feature_count)

        def forward(self, diff_history):
            _, hidden = self.gru(diff_history)
            return self.head(hidden[-1]).view(-1, self.output_steps, self.feature_count)

    return UpstreamChangeForecaster()
