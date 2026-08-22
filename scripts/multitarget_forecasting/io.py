"""Small auditable IO helpers for the joint-target experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts.multitarget_forecasting import config


def safe_filename(value: str) -> str:
    normalized = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", str(value))
    normalized = re.sub(r"\s+", "_", normalized).strip("._")
    return normalized or "未命名"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_identity() -> dict[str, str]:
    return {
        "observed_data_path": str(config.OBSERVED_DATA_PATH),
        "observed_data_sha256": file_sha256(config.OBSERVED_DATA_PATH),
        "quality_data_path": str(config.QUALITY_DATA_PATH),
        "quality_data_sha256": file_sha256(config.QUALITY_DATA_PATH),
    }


def code_identity() -> dict[str, str]:
    root = Path("scripts/multitarget_forecasting")
    scientific_files = ("config.py", "data.py", "model.py")
    return {
        str(root / name): file_sha256(root / name) for name in scientific_files
    }


def save_archive(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files if key != "metadata_json"}
        metadata = json.loads(str(saved["metadata_json"].item()))
    return arrays, metadata


def load_exact(path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    arrays, metadata = load_archive(path)
    if metadata != expected and not _scientifically_compatible(metadata, expected):
        raise RuntimeError(
            f"已有文件与当前联合预测协议不一致: {path}。"
            "请审阅后显式使用 --force。"
        )
    return arrays


def _scientifically_compatible(actual: dict, expected: dict) -> bool:
    """Ignore legacy hashes of CLI/printing files that cannot alter predictions."""

    actual = dict(actual)
    expected = dict(expected)
    scientific_names = {"config.py", "data.py", "model.py"}
    for payload in (actual, expected):
        hashes = payload.get("code_sha256")
        if isinstance(hashes, dict):
            payload["code_sha256"] = {
                name: digest
                for name, digest in hashes.items()
                if Path(name).name in scientific_names
            }
    return actual == expected
