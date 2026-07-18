#!/usr/bin/env python3
"""Build endpoint-aligned inputs for the hourly representation ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import FEATURE_COLUMNS
from scripts.data.wq_preprocessing_v2 import (
    FOUR_HOUR_FEATURE_COLUMNS,
    HOURLY_FEATURE_COLUMNS,
    apply_quality_rules,
    resolve_duplicate_timestamps,
)


RAW_DATA_DIR = Path("data/quantity")
CANONICAL_VALUES_PATH = Path("data/processed/v2/quantity_4h_observed.csv")
CANONICAL_QUALITY_PATH = Path("data/processed/v2/quantity_4h_quality.csv")
OUTPUT_DIR = Path("data/processed/v2/ablation_hourly_representation")
VALUES_PATH = OUTPUT_DIR / "hourly_representation_values.csv"
QUALITY_PATH = OUTPUT_DIR / "hourly_representation_quality.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

START_DATE = pd.Timestamp("2020-01-01")
RESAMPLE_RULE = "4h"
MIN_WINDOW_OBSERVATIONS = 3
STATISTICS = ("mean", "max", "std", "slope")


def station_name(path: Path) -> str:
    return path.name.split("2015年")[0]


def prefixed_feature(prefix: str, feature: str) -> str:
    return f"{prefix}__{feature}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    """Read one station workbook with the same duplicate and quality rules as V2."""
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
    for feature in FEATURE_COLUMNS:
        raw[feature] = pd.to_numeric(raw[feature], errors="coerce") if feature in raw else np.nan
    return apply_quality_rules(resolve_duplicate_timestamps(raw)).sort_values("time")


def _resample(series: pd.Series, method: str) -> pd.Series:
    resampler = series.resample(RESAMPLE_RULE, closed="right", label="right", origin="start_day")
    if method == "mean":
        return resampler.mean()
    if method == "max":
        return resampler.max()
    if method == "std":
        return resampler.std(ddof=0)
    if method == "first":
        return resampler.first()
    if method == "last":
        return resampler.last()
    if method == "count":
        return resampler.count()
    raise ValueError(f"Unsupported resampling method: {method}")


def trailing_statistics(values: pd.Series, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Return causal statistics for (t-4h, t], requiring at least three observations."""
    values = pd.to_numeric(values, errors="coerce").sort_index()
    count = _resample(values, "count").reindex(times).fillna(0).astype(int)
    mean = _resample(values, "mean").reindex(times)
    maximum = _resample(values, "max").reindex(times)
    std = _resample(values, "std").reindex(times)
    first = _resample(values, "first").reindex(times)
    last = _resample(values, "last").reindex(times)

    timestamps = pd.Series(values.index, index=values.index)
    time_hours = ((timestamps - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)).where(values.notna())
    first_time = _resample(time_hours, "first").reindex(times)
    last_time = _resample(time_hours, "last").reindex(times)
    duration = last_time - first_time
    slope = (last - first) / duration.where(duration > 0)

    enough = count >= MIN_WINDOW_OBSERVATIONS
    return pd.DataFrame(
        {
            "observed_count": count,
            "mean": mean.where(enough),
            "max": maximum.where(enough),
            "std": std.where(enough),
            "slope": slope.where(enough),
        },
        index=times,
    )


def build_station_representation(
    station: str,
    cleaned: pd.DataFrame,
    canonical_values: pd.DataFrame,
    canonical_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build common endpoint targets plus mean, endpoint, and window-stat inputs."""
    canonical_values = canonical_values.sort_values("time").copy()
    canonical_quality = canonical_quality.sort_values("time").copy()
    canonical_values["time"] = pd.to_datetime(canonical_values["time"])
    canonical_quality["time"] = pd.to_datetime(canonical_quality["time"])
    times = pd.DatetimeIndex(canonical_values["time"])

    source = cleaned.copy()
    source["time"] = pd.to_datetime(source["time"])
    source = source.set_index("time").sort_index()
    if source.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps remain for {station}")

    values = pd.DataFrame({"station": station, "time": times})
    quality = pd.DataFrame({"station": station, "time": times})
    canonical_quality = canonical_quality.set_index("time").reindex(times)

    for feature in FEATURE_COLUMNS:
        values[prefixed_feature("mean", feature)] = canonical_values[feature].to_numpy(float)

    for feature in HOURLY_FEATURE_COLUMNS:
        exact = pd.to_numeric(source[feature], errors="coerce").reindex(times)
        values[feature] = exact.to_numpy(float)
        stats = trailing_statistics(source[feature], times)
        for statistic in STATISTICS:
            values[prefixed_feature(f"window_{statistic}", feature)] = stats[statistic].to_numpy(float)

        hard = source[f"{feature}__hard_invalid"].astype(bool).reindex(times).fillna(False)
        soft = source[f"{feature}__soft_suspect"].astype(bool).reindex(times).fillna(False)
        quality[f"{feature}__observed_count"] = exact.notna().astype(int).to_numpy()
        quality[f"{feature}__target_ok"] = exact.notna().to_numpy()
        quality[f"{feature}__hard_invalid_count"] = hard.astype(int).to_numpy()
        quality[f"{feature}__soft_suspect"] = soft.to_numpy()

    for feature in FOUR_HOUR_FEATURE_COLUMNS:
        values[feature] = canonical_values[feature].to_numpy(float)
        for suffix in ("observed_count", "target_ok", "hard_invalid_count", "soft_suspect"):
            column = f"{feature}__{suffix}"
            quality[column] = canonical_quality[column].to_numpy()

    duplicate = source.get("duplicate_conflict", pd.Series(False, index=source.index))
    quality["duplicate_conflict"] = duplicate.astype(bool).reindex(times).fillna(False).to_numpy()
    ordered_values = [
        "station",
        "time",
        *FEATURE_COLUMNS,
        *(prefixed_feature("mean", feature) for feature in FEATURE_COLUMNS),
        *(
            prefixed_feature(f"window_{statistic}", feature)
            for feature in HOURLY_FEATURE_COLUMNS
            for statistic in STATISTICS
        ),
    ]
    return values.loc[:, ordered_values], quality


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in work.select_dtypes(include="bool"):
        work[column] = work[column].astype("uint8")
    work.to_csv(path, index=False, encoding="utf-8-sig")


def build_all() -> dict[str, object]:
    canonical_values = pd.read_csv(CANONICAL_VALUES_PATH)
    canonical_quality = pd.read_csv(CANONICAL_QUALITY_PATH)
    canonical_values["time"] = pd.to_datetime(canonical_values["time"])
    canonical_quality["time"] = pd.to_datetime(canonical_quality["time"])

    value_parts = []
    quality_parts = []
    source_paths = sorted(RAW_DATA_DIR.glob("*.xls"))
    for path in source_paths:
        station = station_name(path)
        console.print(f"processing {station}", flush=True)
        station_values = canonical_values[canonical_values["station"].astype(str).eq(station)].copy()
        station_quality = canonical_quality[canonical_quality["station"].astype(str).eq(station)].copy()
        if station_values.empty or station_quality.empty:
            raise ValueError(f"Canonical V2 data are missing station {station}")
        values, quality = build_station_representation(
            station,
            read_source(path),
            station_values,
            station_quality,
        )
        value_parts.append(values)
        quality_parts.append(quality)

    values = pd.concat(value_parts, ignore_index=True).sort_values(["station", "time"]).reset_index(drop=True)
    quality = pd.concat(quality_parts, ignore_index=True).sort_values(["station", "time"]).reset_index(drop=True)
    if values.duplicated(["station", "time"]).any() or quality.duplicated(["station", "time"]).any():
        raise ValueError("Hourly representation outputs contain duplicate station/time keys")

    save_csv(values, VALUES_PATH)
    save_csv(quality, QUALITY_PATH)
    metadata = {
        "version": "hourly-representation-ablation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "common_target": "exact_4h_endpoint_for_hourly_features; canonical_latest_for_native_4h_features",
        "mean_input": "right-edge trailing (t-4h,t] mean from canonical V2",
        "endpoint_input": "exact observation at the 4h endpoint for hourly features",
        "window_statistics": list(STATISTICS),
        "minimum_window_observations": MIN_WINDOW_OBSERVATIONS,
        "causal": True,
        "stations": int(values["station"].nunique()),
        "rows": int(len(values)),
        "inputs": {
            "canonical_values": {"path": str(CANONICAL_VALUES_PATH), "sha256": sha256_file(CANONICAL_VALUES_PATH)},
            "canonical_quality": {"path": str(CANONICAL_QUALITY_PATH), "sha256": sha256_file(CANONICAL_QUALITY_PATH)},
            "raw_sources": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in source_paths
            ],
        },
        "outputs": {
            "values": {"path": str(VALUES_PATH), "sha256": sha256_file(VALUES_PATH)},
            "quality": {"path": str(QUALITY_PATH), "sha256": sha256_file(QUALITY_PATH)},
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    metadata = build_all()
    console.print(json.dumps({key: metadata[key] for key in ("version", "stations", "rows")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
