from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import FEATURE_COLUMNS


HOURLY_FEATURE_COLUMNS = FEATURE_COLUMNS[:5]
FOUR_HOUR_FEATURE_COLUMNS = FEATURE_COLUMNS[5:]
REVIEW_STATUS_RANK = {
    "未审核": 0,
    "已初审": 1,
    "已二审": 2,
    "已三审": 3,
    "已终审": 4,
}


def hard_invalid_mask(column: str, values: pd.Series) -> pd.Series:
    """Return only physically invalid rules approved for automatic removal."""
    rules = {
        "水温(℃)": lambda s: (s <= 0) | (s > 40),
        "pH(无量纲)": lambda s: (s <= 0) | (s >= 14),
        "溶解氧(mg/L)": lambda s: (s < 0) | (s > 25),
        "浊度(NTU)": lambda s: s < 0,
        "电导率(μS/cm)": lambda s: s <= 10,
        "高锰酸盐指数(mg/L)": lambda s: s <= 0,
        "氨氮(mg/L)": lambda s: s < 0,
        "总磷(mg/L)": lambda s: s < 0,
        "总氮(mg/L)": lambda s: s <= 0,
    }
    return rules[column](values).fillna(False)


def soft_suspect_mask(column: str, values: pd.Series) -> pd.Series:
    """Flag unusual but potentially real values without deleting them."""
    rules = {
        "水温(℃)": lambda s: pd.Series(False, index=s.index),
        "pH(无量纲)": lambda s: (s < 3) | (s > 12),
        "溶解氧(mg/L)": lambda s: s == 0,
        "浊度(NTU)": lambda s: s > 5000,
        "电导率(μS/cm)": lambda s: (s <= 20) | (s > 2000),
        "高锰酸盐指数(mg/L)": lambda s: (s <= 0.02) | (s > 20),
        "氨氮(mg/L)": lambda s: s > 20,
        "总磷(mg/L)": lambda s: s > 5,
        "总氮(mg/L)": lambda s: s > 50,
    }
    return rules[column](values).fillna(False)


def apply_quality_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert features to numeric, remove hard-invalid values, and retain soft flags."""
    work = frame.copy()
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(work[column], errors="coerce") if column in work else pd.Series(np.nan, index=work.index)
        hard = hard_invalid_mask(column, values)
        soft = soft_suspect_mask(column, values) & ~hard
        work[column] = values.mask(hard)
        work[f"{column}__hard_invalid"] = hard.astype(bool)
        work[f"{column}__soft_suspect"] = soft.astype(bool)
    return work


def resolve_duplicate_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose one deterministic record per timestamp instead of averaging conflicts."""
    if "time" not in frame:
        raise ValueError("frame must contain time")
    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work[work["time"].notna()].copy()
    if "_source_order" not in work:
        work["_source_order"] = np.arange(len(work), dtype=int)
    status = work["审核状态"].astype(str) if "审核状态" in work else pd.Series("", index=work.index)
    work["_review_rank"] = status.map(REVIEW_STATUS_RANK).fillna(-1).astype(int)
    work["_feature_completeness"] = work.reindex(columns=FEATURE_COLUMNS).notna().sum(axis=1)

    selected = []
    for _, group in work.groupby("time", sort=True):
        chosen = group.sort_values(
            ["_review_rank", "_feature_completeness", "_source_order"],
            kind="stable",
        ).iloc[-1].copy()
        chosen["duplicate_row_count"] = int(len(group))
        chosen["duplicate_conflict"] = bool(
            any(pd.to_numeric(group[column], errors="coerce").dropna().nunique() > 1 for column in FEATURE_COLUMNS)
        )
        selected.append(chosen)

    if not selected:
        return work.iloc[:0].drop(columns=["_review_rank", "_feature_completeness"])
    return (
        pd.DataFrame(selected)
        .drop(columns=["_review_rank", "_feature_completeness"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def _resample(series: pd.Series, method: str) -> pd.Series:
    resampler = series.resample("4h", closed="right", label="right", origin="start_day")
    if method == "mean":
        return resampler.mean()
    if method == "last":
        return resampler.last()
    if method == "count":
        return resampler.count()
    if method == "sum":
        return resampler.sum()
    if method == "max":
        return resampler.max()
    raise ValueError(f"unsupported resample method: {method}")


def aggregate_causal_4h(frame: pd.DataFrame, station: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align all features at the right edge using only observations available by that time."""
    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work[work["time"].notna()].sort_values("time").set_index("time")
    if work.index.has_duplicates:
        raise ValueError("duplicate timestamps must be resolved before 4-hour aggregation")

    values_by_feature: dict[str, pd.Series] = {}
    quality_by_column: dict[str, pd.Series] = {}
    for column in FEATURE_COLUMNS:
        method = "mean" if column in HOURLY_FEATURE_COLUMNS else "last"
        values = pd.to_numeric(work[column], errors="coerce")
        values_by_feature[column] = _resample(values, method)
        count = _resample(values, "count").fillna(0).astype(int)
        minimum_count = 3 if column in HOURLY_FEATURE_COLUMNS else 1
        quality_by_column[f"{column}__observed_count"] = count
        quality_by_column[f"{column}__target_ok"] = (count >= minimum_count) & values_by_feature[column].notna()

        hard_column = f"{column}__hard_invalid"
        soft_column = f"{column}__soft_suspect"
        hard = work[hard_column].astype(int) if hard_column in work else pd.Series(0, index=work.index)
        soft = work[soft_column].astype(int) if soft_column in work else pd.Series(0, index=work.index)
        quality_by_column[f"{column}__hard_invalid_count"] = _resample(hard, "sum").fillna(0).astype(int)
        quality_by_column[f"{column}__soft_suspect"] = _resample(soft, "max").fillna(0).astype(bool)

    values = pd.DataFrame(values_by_feature)
    quality = pd.DataFrame(quality_by_column)
    duplicate = work["duplicate_conflict"].astype(int) if "duplicate_conflict" in work else pd.Series(0, index=work.index)
    quality["duplicate_conflict"] = _resample(duplicate, "max").fillna(0).astype(bool)
    values.insert(0, "time", values.index)
    values.insert(0, "station", str(station))
    quality.insert(0, "time", quality.index)
    quality.insert(0, "station", str(station))
    return values.reset_index(drop=True), quality.reset_index(drop=True)


def _interpolate_short_series(values: pd.Series, limit_steps: int) -> pd.Series:
    interpolated = values.interpolate(method="time", limit_area="inside")
    missing = values.isna()
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("sum")
    interpolated[missing & (lengths > limit_steps)] = np.nan
    return interpolated


def reconstruct_review_values(
    values: pd.DataFrame,
    limit_steps: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a bidirectional reconstruction for review only; never a model truth table."""
    value_parts = []
    flag_parts = []
    for station, group in values.groupby("station", sort=True):
        group = group.sort_values("time").copy()
        group["time"] = pd.to_datetime(group["time"])
        indexed = group.set_index("time")
        reconstructed = indexed.copy()
        flags = pd.DataFrame(index=indexed.index)
        for column in FEATURE_COLUMNS:
            before = pd.to_numeric(indexed[column], errors="coerce")
            after = _interpolate_short_series(before, limit_steps)
            reconstructed[column] = after
            flags[f"{column}__status"] = np.select(
                [before.notna(), before.isna() & after.notna()],
                ["original", "reconstructed"],
                default="remaining_missing",
            )
        reconstructed["station"] = station
        flags["station"] = station
        value_parts.append(reconstructed.reset_index()[["station", "time", *FEATURE_COLUMNS]])
        flag_parts.append(flags.reset_index()[["station", "time", *(f"{column}__status" for column in FEATURE_COLUMNS)]])
    return pd.concat(value_parts, ignore_index=True), pd.concat(flag_parts, ignore_index=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
