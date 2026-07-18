#!/usr/bin/env python3
"""Build one auditable 4-hour panel for the candidate continuous subgraph."""

from __future__ import annotations

from scripts.common.terminal_output import console

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import audit_provincial_hourly_data as provincial
from scripts.common.wq_gru_data import FEATURE_COLUMNS, OUTLIER_RULES, interpolate_short_series


ROOT = Path(__file__).resolve().parents[2]
NODE_PATH = ROOT / "data/metadata/qujiang_continuous_subgraph_nodes.csv"
NATIONAL_VALUE_PATH = ROOT / "data/processed/v2/quantity_4h_observed.csv"
NATIONAL_QUALITY_PATH = ROOT / "data/processed/v2/quantity_4h_quality.csv"
OUTPUT_DIR = ROOT / "data/processed/v2/continuous_subgraph"
QUALITY_OUTPUT_DIR = ROOT / "outputs/quality/continuous_subgraph"
START_DATE = pd.Timestamp("2022-01-01")


def assert_unique_station_time(frame: pd.DataFrame) -> None:
    duplicate = frame.duplicated(["station", "time"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["station", "time"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate station/time keys: {examples}")


def _status(
    observed: pd.Series,
    filled: pd.Series,
    hard_invalid_count: pd.Series,
) -> np.ndarray:
    return np.select(
        [
            observed.notna(),
            observed.isna() & hard_invalid_count.gt(0),
            observed.isna() & filled.notna(),
        ],
        ["original", "hard_invalid", "interpolated"],
        default="remaining_missing",
    )


def build_provincial_station_panel(
    raw: pd.DataFrame,
    station: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean one provincial workbook while keeping target provenance separate."""
    work = raw.copy()
    if "站点名称" in work:
        work = work[work["站点名称"].astype(str).eq(station)].copy()
    work["time"] = provincial.parse_monitoring_time(work["监测时间"])
    work = work[work["time"].ge(START_DATE)].copy()
    if work.empty:
        raise ValueError(f"No provincial rows from {START_DATE.date()} for {station}")

    hard_counts = pd.DataFrame(index=pd.DatetimeIndex(work["time"]))
    for feature in FEATURE_COLUMNS:
        numeric = pd.to_numeric(work[feature], errors="coerce")
        hard_counts[feature] = OUTLIER_RULES[feature](numeric).astype(int).to_numpy()
    hard_counts = hard_counts.resample("4h").sum()

    selected, duplicates = provincial.clean_and_deduplicate(
        work.drop(columns=["time"]),
        start_date=str(START_DATE.date()),
    )
    observed, filled = provincial.to_four_hour_panel(selected)
    hard_counts = hard_counts.reindex(observed.index, fill_value=0)
    duplicate_times = set(pd.to_datetime(duplicates.get("time", pd.Series(dtype="datetime64[ns]"))))

    network_values = work.get("管理级别", pd.Series(dtype=object)).dropna()
    river_values = work.get("水体", pd.Series(dtype=object)).dropna()
    network = str(network_values.iloc[0]).strip().strip(",，") if len(network_values) else "省控"
    river = str(river_values.iloc[0]).strip().strip(",，") if len(river_values) else ""
    values = filled.reset_index(names="time")
    values.insert(0, "station", station)
    values.insert(2, "network", network)
    values.insert(3, "river", river)

    quality = pd.DataFrame({"station": station, "time": observed.index})
    for feature in FEATURE_COLUMNS:
        feature_hard = hard_counts[feature].astype(int)
        quality[f"{feature}__observed_count"] = observed[feature].notna().astype(int).to_numpy()
        quality[f"{feature}__hard_invalid_count"] = feature_hard.to_numpy()
        quality[f"{feature}__target_ok"] = observed[feature].notna().to_numpy()
        quality[f"{feature}__soft_suspect"] = False
        quality[f"{feature}__status"] = _status(
            observed[feature], filled[feature], feature_hard
        )
    quality["duplicate_conflict"] = quality["time"].isin(duplicate_times)
    return values, quality


def build_national_station_panel(
    observed: pd.DataFrame,
    quality: pd.DataFrame,
    station: str,
    river: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply input-only short interpolation while preserving V2 target approval."""
    values = observed.loc[observed["station"].eq(station), ["station", "time", *FEATURE_COLUMNS]].copy()
    values["time"] = pd.to_datetime(values["time"])
    values = values[values["time"].ge(START_DATE)].sort_values("time").reset_index(drop=True)
    if values.empty:
        raise ValueError(f"No national rows from {START_DATE.date()} for {station}")

    approved = quality.loc[quality["station"].eq(station)].copy()
    approved["time"] = pd.to_datetime(approved["time"])
    approved = values[["station", "time"]].merge(
        approved,
        on=["station", "time"],
        how="left",
        validate="one_to_one",
    )
    original = values.copy()
    for feature in FEATURE_COLUMNS:
        indexed = values.set_index("time")[feature]
        values[feature] = interpolate_short_series(indexed).to_numpy()

    values.insert(2, "network", "国控")
    values.insert(3, "river", river)
    sidecar = values[["station", "time"]].copy()
    for feature in FEATURE_COLUMNS:
        target_column = f"{feature}__target_ok"
        hard_column = f"{feature}__hard_invalid_count"
        soft_column = f"{feature}__soft_suspect"
        soft_suspect = approved.get(
            soft_column, pd.Series(False, index=approved.index)
        ).fillna(False).astype(bool)
        target_ok = (
            approved.get(target_column, pd.Series(False, index=approved.index))
            .fillna(False)
            .astype(bool)
            & ~soft_suspect
        )
        hard_count = approved.get(hard_column, pd.Series(0, index=approved.index)).fillna(0).astype(int)
        sidecar[f"{feature}__observed_count"] = original[feature].notna().astype(int)
        sidecar[hard_column] = hard_count
        sidecar[target_column] = target_ok
        sidecar[soft_column] = soft_suspect
        sidecar[f"{feature}__status"] = _status(original[feature], values[feature], hard_count)
    sidecar["duplicate_conflict"] = approved.get(
        "duplicate_conflict", pd.Series(False, index=approved.index)
    ).fillna(False).astype(bool)
    return values, sidecar


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(NODE_PATH)
    nodes = nodes[nodes["data_ready"].astype(bool)].copy()
    national_values = pd.read_csv(NATIONAL_VALUE_PATH)
    national_quality = pd.read_csv(NATIONAL_QUALITY_PATH)
    value_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []

    for row in nodes.itertuples(index=False):
        if row.network == "国控":
            values, quality = build_national_station_panel(
                national_values,
                national_quality,
                row.station,
                row.river,
            )
        else:
            raw = pd.read_excel(ROOT / row.data_source, sheet_name=0)
            values, quality = build_provincial_station_panel(raw, row.station)
        value_frames.append(values)
        quality_frames.append(quality)

    panel = pd.concat(value_frames, ignore_index=True).sort_values(["station", "time"])
    sidecar = pd.concat(quality_frames, ignore_index=True).sort_values(["station", "time"])
    assert_unique_station_time(panel)
    assert_unique_station_time(sidecar)
    return panel.reset_index(drop=True), sidecar.reset_index(drop=True)


def write_outputs(panel: pd.DataFrame, quality: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "quantity_4h.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUTPUT_DIR / "quality_4h.csv", index=False, encoding="utf-8-sig")

    rows = []
    for station, group in panel.groupby("station", sort=True):
        q = quality[quality["station"].eq(station)]
        rows.append(
            {
                "station": station,
                "network": group["network"].iloc[0],
                "river": group["river"].iloc[0],
                "start": group["time"].min(),
                "end": group["time"].max(),
                "rows": len(group),
                "all9_complete_rate": group[list(FEATURE_COLUMNS)].notna().all(axis=1).mean(),
                "target5_approved_cells": int(
                    q[[f"{feature}__target_ok" for feature in provincial.TARGETS]].sum().sum()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        QUALITY_OUTPUT_DIR / "panel_station_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    panel, quality = build_panel()
    write_outputs(panel, quality)
    console.print(
        f"stations={panel['station'].nunique()} rows={len(panel)} "
        f"start={panel['time'].min()} end={panel['time'].max()}"
    )


if __name__ == "__main__":
    main()
