"""Prediction persistence and exact resume checks for the V2 comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from scripts.tabpfn_comparison import config


ARRAY_KEYS = ("pred", "true", "mask", "current", "target_start")

# These hashes identify the predictions produced before the output-only Chinese
# path migration. The numerical protocol did not change, so those files remain
# resumable only when every non-code metadata field and this entire legacy hash
# bundle match exactly.
LEGACY_PREDICTION_CODE_SHA256 = {
    "scripts/tabpfn_comparison/config.py": "0c4b7a91042eac927d012c71fe743fe338538254f8eaf645a7c946bdf6c42379",
    "scripts/tabpfn_comparison/data.py": "56018e931f3e1b4eded189a1e4121ccb3dba2ba4c5c85e120b3550b207aeda76",
    "scripts/tabpfn_comparison/io.py": "77ba8c37826d02f49367ce36c10313065166e6b4894deaea0226adc89b3f71b4",
    "scripts/tabpfn_comparison/models.py": "f8a84f3d46974d2c1086aac443b1519cb8fca63d49ae5f5a7518fa5451cb68a2",
    "scripts/tabpfn_comparison/run.py": "6bc37a9014fc969e8a1e51eea56ffd7e7d0340f9f610291b56a6625d16cce926",
}


def safe_filename(value: str) -> str:
    """Create a filesystem-safe directory component."""
    # Keep Chinese station and indicator names: stripping non-ASCII text would
    # collapse several official targets to the same directory name.
    normalized = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", str(value))
    normalized = re.sub(r"\s+", "_", normalized).strip("._")
    return normalized or "unnamed"


def prediction_path(
    evaluation_split: str, model: str, seed: int, target: str, station: str
) -> Path:
    model_label = config.MODEL_FILE_LABELS.get(model, model)
    filename = "__".join(
        (
            safe_filename(model_label),
            f"种子{seed}",
            safe_filename(station),
            safe_filename(target),
        )
    )
    return config.output_dir_for_split(evaluation_split) / "预测结果" / f"{filename}.npz"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_prediction(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    missing = set(ARRAY_KEYS).difference(arrays)
    if missing:
        raise ValueError(f"Prediction arrays missing keys: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {key: np.asarray(arrays[key]) for key in ARRAY_KEYS}
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


def _normalized_code_hashes(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {str(path).replace("\\", "/"): str(digest) for path, digest in value.items()}


def _metadata_matches(saved: dict, expected: dict) -> bool:
    if saved == expected:
        return True

    saved_code = _normalized_code_hashes(saved.get("code_sha256"))
    expected_code = _normalized_code_hashes(expected.get("code_sha256"))
    if saved_code not in (expected_code, LEGACY_PREDICTION_CODE_SHA256):
        return False

    saved_protocol = {key: value for key, value in saved.items() if key != "code_sha256"}
    expected_protocol = {key: value for key, value in expected.items() if key != "code_sha256"}
    for key in ("observed_data_path", "quality_data_path"):
        if key in saved_protocol:
            saved_protocol[key] = str(saved_protocol[key]).replace("\\", "/")
        if key in expected_protocol:
            expected_protocol[key] = str(expected_protocol[key]).replace("\\", "/")
    return saved_protocol == expected_protocol


def is_complete(path: Path, expected_metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        arrays, metadata = load_prediction(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if not _metadata_matches(metadata, expected_metadata):
        return False
    count = len(arrays["target_start"])
    return (
        arrays["pred"].shape == (count, config.OUTPUT_STEPS)
        and arrays["true"].shape == (count, config.OUTPUT_STEPS)
        and arrays["mask"].shape == (count, config.OUTPUT_STEPS)
        and arrays["current"].shape == (count, 1)
        and arrays["target_start"].shape == (count,)
    )


def should_skip(path: Path, expected_metadata: dict, *, force: bool) -> bool:
    """Resume an exact result; reject silent replacement of another protocol."""
    if force or not path.exists():
        return False
    if is_complete(path, expected_metadata):
        return True
    raise RuntimeError(
        f"Existing result does not match this frozen protocol: {path}. "
        "Inspect it, then pass --force only if replacement is intentional."
    )
