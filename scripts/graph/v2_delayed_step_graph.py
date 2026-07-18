#!/usr/bin/env python3
"""Causal delayed step-change messages for V2 graph experiments."""

from __future__ import annotations

import numpy as np


def lag_support(primary_steps: int, radius: int = 1) -> tuple[int, ...]:
    """Return a positive lag window centered on the physical estimate."""
    primary = int(primary_steps)
    radius = int(radius)
    if primary < 1 or radius < 0:
        raise ValueError("primary_steps must be positive and radius nonnegative")
    return tuple(range(max(1, primary - radius), primary + radius + 1))


def align_lag_support(
    observed_history: np.ndarray,
    predicted_future: np.ndarray,
    support: tuple[int, ...],
    output_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align source one-step changes to downstream horizons without look-ahead."""
    history = np.asarray(observed_history, dtype=float)
    future = np.asarray(predicted_future, dtype=float)
    if history.ndim != 2 or future.ndim != 2 or history.shape[1] != future.shape[1]:
        raise ValueError("history and future must be two-dimensional with equal feature counts")
    lags = tuple(int(value) for value in support)
    if not lags or min(lags) < 1:
        raise ValueError("support must contain positive lag steps")
    if history.shape[0] < max(lags):
        raise ValueError("history is too short for the requested lag support")
    needed_future = max(0, int(output_steps) - min(lags))
    if future.shape[0] < needed_future:
        raise ValueError("predicted future is too short for the requested horizons")

    aligned = np.empty((int(output_steps), len(lags), history.shape[1]), dtype=float)
    known = np.empty((int(output_steps), len(lags)), dtype=bool)
    for horizon_idx in range(int(output_steps)):
        horizon = horizon_idx + 1
        for lag_idx, lag in enumerate(lags):
            source_step = horizon - lag
            if source_step <= 0:
                aligned[horizon_idx, lag_idx] = history[source_step - 1]
                known[horizon_idx, lag_idx] = True
            else:
                aligned[horizon_idx, lag_idx] = future[source_step - 1]
                known[horizon_idx, lag_idx] = False
    return aligned, known


def observed_only(aligned: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Remove positions that require a forecast of future upstream changes."""
    values = np.asarray(aligned, dtype=float)
    mask = np.asarray(known, dtype=bool)
    if values.shape[:-1] != mask.shape:
        raise ValueError("known mask must match aligned values except for the feature axis")
    return np.where(mask[..., None], values, 0.0)


def cumulative_step_correction(step_correction):
    """Convert downstream one-step corrections to anchor-relative corrections."""
    return step_correction.cumsum(dim=-2)


def make_delayed_step_mapper(torch, source_dim: int, target_dim: int, lag_count: int):
    """Create a zero-preserving lag kernel and sparse linear feature transfer."""

    class _DelayedStepMapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lag_logits = torch.nn.Parameter(torch.zeros(int(lag_count)))
            self.transfer = torch.nn.Linear(int(source_dim), int(target_dim), bias=False)
            torch.nn.init.zeros_(self.transfer.weight)

        def lag_weights(self):
            return torch.softmax(self.lag_logits, dim=0)

        def forward(self, aligned_source, accumulate: bool = True):
            if aligned_source.shape[-2] != self.lag_logits.numel():
                raise ValueError("aligned lag axis does not match lag_count")
            mixed_source = (aligned_source * self.lag_weights()[None, None, :, None]).sum(dim=-2)
            step = self.transfer(mixed_source)
            correction = cumulative_step_correction(step) if accumulate else step
            return step, correction

    return _DelayedStepMapper()
