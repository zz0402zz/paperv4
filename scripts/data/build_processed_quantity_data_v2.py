#!/usr/bin/env python3
"""Build the canonical causal, quality-aware 4-hour water-quality dataset."""

from __future__ import annotations

from scripts.common.terminal_output import console

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

from scripts.baselines.gat_gru_baseline import TARGET_FEATURE_COLUMNS
from scripts.data.wq_preprocessing_v2 import (
    FEATURE_COLUMNS,
    aggregate_causal_4h,
    apply_quality_rules,
    reconstruct_review_values,
    resolve_duplicate_timestamps,
    sha256_file,
    write_json,
)


RAW_DATA_DIR = Path("data/quantity")
OUTPUT_DIR = Path("data/processed/v2")
QUALITY_OUTPUT_DIR = Path("outputs/quality/preprocessing_v2")
OBSERVED_DATA_PATH = OUTPUT_DIR / "quantity_4h_observed.csv"
QUALITY_DATA_PATH = OUTPUT_DIR / "quantity_4h_quality.csv"
RECONSTRUCTED_DATA_PATH = OUTPUT_DIR / "quantity_4h_reconstructed_review.csv"
RECONSTRUCTION_FLAGS_PATH = OUTPUT_DIR / "quantity_4h_reconstruction_flags.csv"
METADATA_PATH = OUTPUT_DIR / "preprocessing_metadata.json"
QUALITY_SUMMARY_PATH = QUALITY_OUTPUT_DIR / "station_feature_quality_summary.csv"
DUPLICATE_SUMMARY_PATH = QUALITY_OUTPUT_DIR / "duplicate_resolution_summary.csv"
SPLIT_COVERAGE_PATH = QUALITY_OUTPUT_DIR / "split_target_coverage.csv"

START_DATE = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")
RECONSTRUCTION_LIMIT_STEPS = 3
RULE_VERSION = "hard-v2-user-confirmed_soft-v1"
TIME_ALIGNMENT = "right_edge_trailing_(t-4h,t]_hourly_latest_native4h"
CODE_PATHS = (
    Path("scripts/data/build_processed_quantity_data_v2.py"),
    Path("scripts/data/wq_preprocessing_v2.py"),
    Path("scripts/common/wq_gru_data.py"),
)


def preprocessing_config() -> dict[str, object]:
    return {
        "start_date": str(START_DATE),
        "resample_rule": "4h",
        "time_alignment": TIME_ALIGNMENT,
        "rule_version": RULE_VERSION,
        "reconstruction_is_model_input": False,
        "reconstruction_limit_steps": RECONSTRUCTION_LIMIT_STEPS,
        "reconstruction_limit_hours": RECONSTRUCTION_LIMIT_STEPS * 4,
        "hourly_target_min_observations": 3,
        "native_4h_target_min_observations": 1,
    }


def stable_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xlrd": xlrd.__version__,
    }


def station_name(path: Path) -> str:
    return path.name.split("2015年")[0]


def read_station_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse one source workbook and return observed values, quality, and duplicate audit."""
    raw = pd.read_excel(path, sheet_name=0, engine="xlrd")
    raw = raw[raw["监测时间"].notna()].copy()
    raw = raw[~raw["监测时间"].astype(str).str.contains("标准", na=False)].copy()
    raw["time"] = pd.to_datetime(
        raw["监测时间"].astype(str).str.strip(),
        format="%Y-%m-%d %H",
        errors="coerce",
    )
    raw = raw[raw["time"].notna() & (raw["time"] >= START_DATE - pd.Timedelta(hours=4))].copy()
    raw["_source_order"] = np.arange(len(raw), dtype=int)
    for column in FEATURE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce") if column in raw else np.nan

    resolved = resolve_duplicate_timestamps(raw)
    duplicate_audit = resolved.loc[
        resolved["duplicate_row_count"].gt(1),
        ["time", "duplicate_row_count", "duplicate_conflict", "审核状态"],
    ].copy()
    duplicate_audit.insert(0, "station", station_name(path))

    cleaned = apply_quality_rules(resolved)
    values, quality = aggregate_causal_4h(cleaned, station=station_name(path))
    keep = values["time"] >= START_DATE
    return values.loc[keep].reset_index(drop=True), quality.loc[keep].reset_index(drop=True), duplicate_audit


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    if "time" in work:
        work["time"] = pd.to_datetime(work["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in work.select_dtypes(include="bool"):
        work[column] = work[column].astype("uint8")
    work.to_csv(path, index=False, encoding="utf-8-sig")


def station_feature_summary(values: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    joined = values.merge(quality, on=["station", "time"], how="left", validate="one_to_one")
    rows = []
    for station, group in joined.groupby("station", sort=True):
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "station": station,
                    "feature": feature,
                    "rows": int(len(group)),
                    "finite_values": int(group[feature].notna().sum()),
                    "target_ok_values": int(group[f"{feature}__target_ok"].fillna(False).sum()),
                    "hard_invalid_raw_values": int(group[f"{feature}__hard_invalid_count"].fillna(0).sum()),
                    "soft_suspect_bins": int(group[f"{feature}__soft_suspect"].fillna(False).sum()),
                    "partial_or_missing_bins": int((~group[f"{feature}__target_ok"].fillna(False)).sum()),
                }
            )
    return pd.DataFrame(rows)


def split_target_coverage(values: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    stations = tuple(sorted(values["station"].unique()))
    times = pd.date_range(values["time"].min(), values["time"].max(), freq="4h")
    index = pd.MultiIndex.from_product([times, stations], names=["time", "station"])
    target_columns = [f"{feature}__target_ok" for feature in TARGET_FEATURE_COLUMNS]
    panel = (
        quality.set_index(["time", "station"])[target_columns]
        .reindex(index)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    rows = []
    splits = {
        "train": (times < TRAIN_END),
        "val": (times >= TRAIN_END) & (times < VAL_END),
        "test": times >= VAL_END,
    }
    for split, time_mask in splits.items():
        split_times = times[time_mask]
        for station in stations:
            station_panel = panel.loc[(split_times, station), target_columns] if len(split_times) else panel.iloc[:0]
            rows.append(
                {
                    "split": split,
                    "station": station,
                    "expected_time_rows": int(len(split_times)),
                    "valid_target_cells": int(station_panel.to_numpy(bool).sum()),
                    "has_any_target": bool(station_panel.to_numpy(bool).any()),
                }
            )
    return pd.DataFrame(rows)


def build_all() -> dict[str, object]:
    paths = sorted(RAW_DATA_DIR.glob("*.xls"))
    if not paths:
        raise FileNotFoundError(f"No source workbooks found in {RAW_DATA_DIR}")

    value_parts = []
    quality_parts = []
    duplicate_parts = []
    for path in paths:
        console.print(f"processing {path.name}", flush=True)
        values, quality, duplicates = read_station_source(path)
        value_parts.append(values)
        quality_parts.append(quality)
        duplicate_parts.append(duplicates)

    values = pd.concat(value_parts, ignore_index=True).sort_values(["station", "time"]).reset_index(drop=True)
    quality = pd.concat(quality_parts, ignore_index=True).sort_values(["station", "time"]).reset_index(drop=True)
    duplicates = pd.concat(duplicate_parts, ignore_index=True).sort_values(["station", "time"]).reset_index(drop=True)
    if values.duplicated(["station", "time"]).any() or quality.duplicated(["station", "time"]).any():
        raise ValueError("V2 output contains duplicate station/time keys")

    reconstructed, reconstruction_flags = reconstruct_review_values(
        values,
        limit_steps=RECONSTRUCTION_LIMIT_STEPS,
    )
    summary = station_feature_summary(values, quality)
    split_coverage = split_target_coverage(values, quality)

    save_csv(values, OBSERVED_DATA_PATH)
    save_csv(quality, QUALITY_DATA_PATH)
    save_csv(reconstructed, RECONSTRUCTED_DATA_PATH)
    save_csv(reconstruction_flags, RECONSTRUCTION_FLAGS_PATH)
    save_csv(summary, QUALITY_SUMMARY_PATH)
    save_csv(duplicates, DUPLICATE_SUMMARY_PATH)
    save_csv(split_coverage, SPLIT_COVERAGE_PATH)

    config = preprocessing_config()
    metadata = {
        "version": "v2",
        "raw_data_dir": str(RAW_DATA_DIR),
        **config,
        "config_sha256": stable_sha256(config),
        "code_sha256": {str(path): sha256_file(path) for path in CODE_PATHS},
        "runtime_versions": runtime_versions(),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in paths
        ],
        "outputs": {
            "observed_values": str(OBSERVED_DATA_PATH),
            "quality": str(QUALITY_DATA_PATH),
            "review_reconstruction": str(RECONSTRUCTED_DATA_PATH),
            "reconstruction_flags": str(RECONSTRUCTION_FLAGS_PATH),
        },
        "output_sha256": {
            "observed_values": sha256_file(OBSERVED_DATA_PATH),
            "quality": sha256_file(QUALITY_DATA_PATH),
            "review_reconstruction": sha256_file(RECONSTRUCTED_DATA_PATH),
            "reconstruction_flags": sha256_file(RECONSTRUCTION_FLAGS_PATH),
        },
        "counts": {
            "stations": int(values["station"].nunique()),
            "rows": int(len(values)),
            "duplicate_source_groups": int(len(duplicates)),
            "conflicting_duplicate_source_groups": int(duplicates["duplicate_conflict"].sum()) if len(duplicates) else 0,
            "test_stations_with_targets": int(
                split_coverage.loc[split_coverage["split"].eq("test"), "has_any_target"].sum()
            ),
        },
    }
    write_json(METADATA_PATH, metadata)
    return metadata


def main() -> int:
    metadata = build_all()
    console.print(f"saved {OBSERVED_DATA_PATH}", flush=True)
    console.print(metadata["counts"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
