#!/usr/bin/env python3
"""Canonical paths, split dates, targets, and provenance for V2 experiments."""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.common.wq_gru_data import FEATURE_COLUMNS


OBSERVED_DATA_PATH = Path("data/processed/v2/quantity_4h_observed.csv")
QUALITY_DATA_PATH = Path("data/processed/v2/quantity_4h_quality.csv")
PREPROCESSING_METADATA_PATH = Path("data/processed/v2/preprocessing_metadata.json")
STRICT_EDGES_PATH = Path("data/metadata/station_edges_verified_strict.csv")
OUTPUT_ROOT = Path("outputs/experiments/v2_reprocessed_20260710")
PROTOCOL_OUTPUT_ROOT = OUTPUT_ROOT / "protocol"
BASELINE_OUTPUT_ROOT = OUTPUT_ROOT / "baselines"
GRU_OUTPUT_ROOT = OUTPUT_ROOT / "gru"
GRAPH_OUTPUT_ROOT = OUTPUT_ROOT / "graph"
REPORT_OUTPUT_ROOT = OUTPUT_ROOT / "reports"

START_DATE = "2022-01-01"
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
RESAMPLE_RULE = "4h"
PILOT_SEED = 42
FORMAL_SEEDS = (17, 42, 73, 101, 202)

TARGET_FEATURE_COLUMNS = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
INPUT_FEATURE_COLUMNS = FEATURE_COLUMNS


def file_provenance(path: Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }


def build_run_manifest(
    experiment: str,
    output_dir: Path,
    seed: int = PILOT_SEED,
    observed_path: Path = OBSERVED_DATA_PATH,
    quality_path: Path = QUALITY_DATA_PATH,
    edge_path: Path = STRICT_EDGES_PATH,
    code_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Build a JSON-safe provenance record shared by every V2 experiment."""
    return {
        "protocol_version": "v2_reprocessed_20260710",
        "experiment": experiment,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": runtime_versions(),
        "splits": {
            "start_date": START_DATE,
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "resample_rule": RESAMPLE_RULE,
        },
        "input_features": list(INPUT_FEATURE_COLUMNS),
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "target_policy": "approved_original_observations_only",
        "review_reconstruction_is_model_input": False,
        "inputs": {
            "observed": file_provenance(observed_path),
            "quality": file_provenance(quality_path),
            "edges": file_provenance(edge_path),
        },
        "code": {str(path): file_provenance(path)["sha256"] for path in code_paths},
    }
