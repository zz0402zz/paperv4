"""Prediction persistence and exact resume checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.common import forecasting
from scripts.tabpfn_comparison import config


ARRAY_KEYS = ("pred", "true", "mask", "current", "target_start")


def prediction_path(model: str, seed: int, target: str, station: str) -> Path:
    return (
        config.OUTPUT_DIR
        / "predictions"
        / model
        / f"seed_{seed}"
        / forecasting.safe_filename(target)
        / f"{forecasting.safe_filename(station)}.npz"
    )


def partial_prediction_path(path: Path) -> Path:
    return path.with_suffix(".partial.npz")


def save_prediction(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    missing = set(ARRAY_KEYS).difference(arrays)
    if missing:
        raise ValueError(f"Prediction arrays missing keys: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        key: np.asarray(arrays[key])
        for key in ARRAY_KEYS
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_prediction(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ARRAY_KEYS}
        metadata = json.loads(str(saved["metadata_json"].item()))
    return arrays, metadata


def is_complete(path: Path, expected_metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        arrays, metadata = load_prediction(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if metadata != expected_metadata:
        return False
    count = len(arrays["target_start"])
    return (
        arrays["pred"].shape == (count, config.OUTPUT_STEPS)
        and arrays["true"].shape == (count, config.OUTPUT_STEPS)
        and arrays["mask"].shape == (count, config.OUTPUT_STEPS)
        and arrays["current"].shape == (count, 1)
    )


def should_skip(path: Path, expected_metadata: dict, *, force: bool) -> bool:
    """Resume an exact result; refuse silent overwrite of anything else."""
    if force or not path.exists():
        return False
    if is_complete(path, expected_metadata):
        return True
    raise RuntimeError(
        f"Existing result does not match the frozen protocol: {path}. "
        "Inspect it, then pass --force only if replacement is intentional."
    )


def load_prediction_prefix(
    path: Path,
    expected_metadata: dict,
    base: dict[str, np.ndarray],
) -> np.ndarray:
    """Load and audit a native rolling-prediction checkpoint prefix."""
    arrays, metadata = load_prediction(path)
    if metadata != expected_metadata:
        raise RuntimeError(f"Partial result metadata mismatch: {path}")
    count = len(arrays["target_start"])
    if count > len(base["target_start"]):
        raise RuntimeError(f"Partial result is longer than the current task: {path}")
    expected_shapes = {
        "pred": (count, config.OUTPUT_STEPS),
        "true": (count, config.OUTPUT_STEPS),
        "mask": (count, config.OUTPUT_STEPS),
        "current": (count, 1),
        "target_start": (count,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise RuntimeError(
                f"Partial result shape mismatch for {key}: "
                f"{arrays[key].shape} != {shape}"
            )
    for key in ("true", "mask", "current", "target_start"):
        if not np.array_equal(
            arrays[key], np.asarray(base[key])[:count], equal_nan=True
        ):
            raise RuntimeError(f"Partial result prefix mismatch for {key}: {path}")
    return np.asarray(arrays["pred"], dtype=float)
