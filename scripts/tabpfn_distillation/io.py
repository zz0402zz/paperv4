"""Auditable, resumable files for long-horizon teacher and student outputs."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from scripts.tabpfn_distillation import config


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


@lru_cache(maxsize=None)
def code_sha256(names: tuple[str, ...]) -> dict[str, str]:
    root = Path("scripts/tabpfn_distillation")
    paths = tuple(root / name for name in names)
    return {str(path): file_sha256(path) for path in paths}


@lru_cache(maxsize=1)
def data_identity() -> dict[str, str]:
    return {
        "observed_data_path": str(config.OBSERVED_DATA_PATH),
        "observed_data_sha256": file_sha256(config.OBSERVED_DATA_PATH),
        "quality_data_path": str(config.QUALITY_DATA_PATH),
        "quality_data_sha256": file_sha256(config.QUALITY_DATA_PATH),
    }


def teacher_cache_path(kind: str, station: str, target: str) -> Path:
    if kind not in {"训练OOF", "验证集"}:
        raise ValueError(f"Unknown teacher cache kind: {kind}")
    filename = "__".join(
        (
            safe_filename(station),
            safe_filename(target),
            f"种子{config.TEACHER_SEED}",
            kind,
        )
    )
    return config.OUTPUT_DIR / "教师缓存" / f"{filename}.npz"


def student_prediction_path(
    evaluation_split: str, variant: str, seed: int, station: str, target: str
) -> Path:
    label = config.STUDENT_FILE_LABELS[variant]
    filename = "__".join(
        (safe_filename(label), f"种子{seed}", safe_filename(station), safe_filename(target))
    )
    return config.output_dir_for_split(evaluation_split) / "预测结果" / f"{filename}.npz"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


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


def load_exact(path: Path, expected_metadata: dict) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        arrays, metadata = load_archive(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if metadata != expected_metadata:
        raise RuntimeError(
            f"已有文件与当前冻结协议不一致: {path}。请先审阅，再显式使用 --force。"
        )
    return arrays
