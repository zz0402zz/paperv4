#!/usr/bin/env python3
"""Audit provincial hourly water-quality files for 4-hour graph modelling."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import FEATURE_COLUMNS, OUTLIER_RULES, interpolate_short_series


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/省控小时数据"
LOCATION_PATH = ROOT / "outputs/maps/provincial_station_locations.csv"
NATIONAL_METADATA_PATH = ROOT / "data/metadata/station_metadata.csv"
OUTPUT_DIR = ROOT / "outputs/quality/provincial_station_data_audit"

FEATURES = tuple(FEATURE_COLUMNS)
TARGETS = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
START_DATE = "2022-01-01"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")
INPUT_STEPS = 6
OUTPUT_STEPS = 9
REVIEW_PRIORITY = {
    "已终审": 5,
    "已三审": 4,
    "已二审": 3,
    "已一审": 2,
    "未审核": 1,
}
DUPLICATE_COLUMNS = (
    "time",
    "duplicate_row_count",
    "conflict_kind",
    "selected_source_row",
    "selected_review_status",
    "selected_valid_feature_count",
    "candidate_review_statuses",
    "candidate_valid_feature_counts",
)


def parse_monitoring_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d %H", errors="coerce")
    fallback = parsed.isna() & values.notna()
    if fallback.any():
        parsed.loc[fallback] = pd.to_datetime(values.loc[fallback], errors="coerce")
    return parsed


def clean_and_deduplicate(
    raw: pd.DataFrame,
    start_date: str = START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hard rules and choose one auditable row per duplicate timestamp."""
    missing = [feature for feature in FEATURES if feature not in raw.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    work = raw.copy()
    work["time"] = parse_monitoring_time(work["监测时间"])
    work = work[work["time"].ge(pd.Timestamp(start_date))].copy()
    work["_row_order"] = np.arange(len(work))
    for feature in FEATURES:
        values = pd.to_numeric(work[feature], errors="coerce")
        work[feature] = values.mask(OUTLIER_RULES[feature](values))
    work["_valid_count"] = work[list(FEATURES)].notna().sum(axis=1)
    work["_review_priority"] = work.get("审核状态", pd.Series("", index=work.index)).map(
        REVIEW_PRIORITY
    ).fillna(0)

    duplicate_rows = []
    selected_indices = []
    for timestamp, group in work.groupby("time", sort=True):
        ranked = group.sort_values(
            ["_valid_count", "_review_priority", "_row_order"],
            ascending=[False, False, False],
        )
        selected = ranked.iloc[0]
        selected_indices.append(selected.name)
        if len(group) == 1:
            continue
        values = group[list(FEATURES)]
        exact = len(values.fillna(-999999.0).drop_duplicates()) == 1
        unique_counts = group["_valid_count"].nunique()
        if exact:
            conflict_kind = "exact_duplicate"
        elif unique_counts > 1:
            conflict_kind = "completeness_resolved"
        else:
            conflict_kind = "ambiguous_equal_completeness"
        duplicate_rows.append(
            {
                "time": timestamp,
                "duplicate_row_count": int(len(group)),
                "conflict_kind": conflict_kind,
                "selected_source_row": int(selected["_row_order"]),
                "selected_review_status": str(selected.get("审核状态", "")),
                "selected_valid_feature_count": int(selected["_valid_count"]),
                "candidate_review_statuses": "|".join(group.get("审核状态", pd.Series("", index=group.index)).astype(str)),
                "candidate_valid_feature_counts": "|".join(group["_valid_count"].astype(str)),
            }
        )
    selected = work.loc[selected_indices].sort_values("time").reset_index(drop=True)
    return selected, pd.DataFrame(duplicate_rows, columns=DUPLICATE_COLUMNS)


def to_four_hour_panel(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_time = selected.groupby("time")[list(FEATURES)].mean().resample("4h").mean()
    full_index = pd.date_range(by_time.index.min().floor("4h"), by_time.index.max().floor("4h"), freq="4h")
    observed = by_time.reindex(full_index)
    interpolated = observed.copy()
    for feature in FEATURES:
        interpolated[feature] = interpolate_short_series(interpolated[feature], limit=3)
    return observed, interpolated


def complete_window_counts(frame: pd.DataFrame) -> dict[str, int]:
    input_ok = frame[list(FEATURES)].notna().all(axis=1).to_numpy()
    output_ok = frame[list(TARGETS)].notna().all(axis=1).to_numpy()
    counts = {"train": 0, "val": 0, "test": 0}
    for origin in range(INPUT_STEPS - 1, len(frame) - OUTPUT_STEPS):
        if not input_ok[origin - INPUT_STEPS + 1 : origin + 1].all():
            continue
        if not output_ok[origin + 1 : origin + OUTPUT_STEPS + 1].all():
            continue
        target_start = frame.index[origin + 1]
        target_end = frame.index[origin + OUTPUT_STEPS]
        if target_end < TRAIN_END:
            counts["train"] += 1
        elif target_start >= TRAIN_END and target_end < VAL_END:
            counts["val"] += 1
        elif target_start >= VAL_END:
            counts["test"] += 1
    return counts


def is_data_ready(row: pd.Series) -> bool:
    return bool(
        row["schema_complete"]
        and float(row["all9_rate_after_short_interp"]) >= 0.75
        and int(row["train_windows_after"]) >= 1000
        and int(row["val_windows_after"]) >= 500
        and int(row["test_windows_after"]) >= 100
        and pd.Timestamp(row["end"]) >= pd.Timestamp("2025-05-01")
    )


def workbook_metadata(raw: pd.DataFrame, path: Path) -> dict[str, object]:
    def first(column: str) -> str:
        if column not in raw:
            return ""
        values = raw[column].dropna()
        return str(values.iloc[0]).strip().strip(",，") if len(values) else ""

    return {
        "station": first("站点名称") or path.name.split("2015年")[0],
        "network": first("管理级别"),
        "city": first("所在设区市"),
        "county": first("所在县(市、区)"),
        "river": first("水体"),
        "source_file": str(path.relative_to(ROOT)),
    }


def audit_workbook(path: Path) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=0)
    metadata = workbook_metadata(raw, path)
    missing = [feature for feature in FEATURES if feature not in raw.columns]
    if missing:
        summary = {
            **metadata,
            "schema_complete": False,
            "missing_feature_columns": "|".join(missing),
            "start": pd.NaT,
            "end": pd.NaT,
            "span_4h": 0,
            "all9_rate_before": 0.0,
            "all9_rate_after_short_interp": 0.0,
            "target5_rate_before": 0.0,
            "target5_rate_after_short_interp": 0.0,
            "train_windows_before": 0,
            "val_windows_before": 0,
            "test_windows_before": 0,
            "train_windows_after": 0,
            "val_windows_after": 0,
            "test_windows_after": 0,
            "duplicate_groups_2022plus": 0,
            "conflicting_duplicate_groups": 0,
            "ambiguous_duplicate_groups": 0,
        }
        return summary, [], pd.DataFrame()

    selected, duplicate = clean_and_deduplicate(raw)
    observed, interpolated = to_four_hour_panel(selected)
    before_counts = complete_window_counts(observed)
    after_counts = complete_window_counts(interpolated)
    duplicate.insert(0, "station", metadata["station"])
    duplicate.insert(1, "river", metadata["river"])
    all9_before = observed[list(FEATURES)].notna().all(axis=1)
    all9_after = interpolated[list(FEATURES)].notna().all(axis=1)
    target5_before = observed[list(TARGETS)].notna().all(axis=1)
    target5_after = interpolated[list(TARGETS)].notna().all(axis=1)
    summary = {
        **metadata,
        "schema_complete": True,
        "missing_feature_columns": "",
        "start": observed.index.min(),
        "end": observed.index.max(),
        "span_4h": int(len(observed)),
        "all9_rate_before": float(all9_before.mean()),
        "all9_rate_after_short_interp": float(all9_after.mean()),
        "target5_rate_before": float(target5_before.mean()),
        "target5_rate_after_short_interp": float(target5_after.mean()),
        "train_windows_before": before_counts["train"],
        "val_windows_before": before_counts["val"],
        "test_windows_before": before_counts["test"],
        "train_windows_after": after_counts["train"],
        "val_windows_after": after_counts["val"],
        "test_windows_after": after_counts["test"],
        "duplicate_groups_2022plus": int(len(duplicate)),
        "conflicting_duplicate_groups": int(duplicate["conflict_kind"].ne("exact_duplicate").sum()),
        "ambiguous_duplicate_groups": int(duplicate["conflict_kind"].eq("ambiguous_equal_completeness").sum()),
    }
    feature_rows = [
        {
            "station": metadata["station"],
            "river": metadata["river"],
            "feature": feature,
            "valid_4h_before": int(observed[feature].notna().sum()),
            "valid_rate_before": float(observed[feature].notna().mean()),
            "valid_4h_after_short_interp": int(interpolated[feature].notna().sum()),
            "valid_rate_after_short_interp": float(interpolated[feature].notna().mean()),
        }
        for feature in FEATURES
    ]
    return summary, feature_rows, duplicate


def attach_location_status(summary: pd.DataFrame) -> pd.DataFrame:
    locations = pd.read_csv(LOCATION_PATH)
    columns = [
        "station",
        "longitude",
        "latitude",
        "coordinate_confidence",
        "address",
        "address_confidence",
        "address_source",
    ]
    output = summary.merge(locations[columns], on="station", how="left", validate="one_to_one")
    output["has_official_address_evidence"] = output["address"].notna() & output["address"].astype(str).ne("")
    output["coordinate_is_centroid"] = output["coordinate_confidence"].isin(("county_centroid", "city_centroid"))
    output["data_ready_current_split"] = output.apply(is_data_ready, axis=1)
    output["strict_graph_ready"] = False
    return output


def same_river_inventory(summary: pd.DataFrame) -> pd.DataFrame:
    national = pd.read_csv(NATIONAL_METADATA_PATH)
    rows = []
    for river in sorted(set(summary["river"].dropna().astype(str)) | set(national["river"].dropna().astype(str))):
        provincial = summary[summary["river"].astype(str).eq(river)]
        national_rows = national[national["river"].astype(str).eq(river)]
        rows.append(
            {
                "river": river,
                "provincial_station_count": int(len(provincial)),
                "provincial_data_ready_count": int(provincial["data_ready_current_split"].sum()),
                "provincial_stations": "|".join(provincial["station"].astype(str)),
                "provincial_data_ready_stations": "|".join(
                    provincial.loc[provincial["data_ready_current_split"], "station"].astype(str)
                ),
                "national_station_count": int(len(national_rows)),
                "national_stations": "|".join(national_rows["station"].astype(str)),
                "candidate_mixed_network_reach": bool(
                    provincial["data_ready_current_split"].any() and len(national_rows) > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_mixed_network_reach", "provincial_data_ready_count", "national_station_count"],
        ascending=[False, False, False],
    )


def write_report(summary: pd.DataFrame, rivers: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    ready = summary[summary["data_ready_current_split"]].sort_values(
        ["train_windows_after", "all9_rate_after_short_interp"], ascending=False
    )
    candidates = rivers[rivers["candidate_mixed_network_reach"]]
    lines = [
        "# Provincial Hourly Data Audit",
        "",
        "- Frequency policy: aggregate native hourly/four-hour observations into aligned 4-hour bins.",
        "- Quality policy: apply the existing hard rules, select one row per duplicate timestamp, and interpolate only internal gaps of at most three 4-hour steps.",
        "- Window policy: prior 24h of nine inputs, direct five-target output through 36h.",
        "- Data-ready is separate from graph-ready: no provincial edge is promoted into the strict graph by this audit.",
        "",
        "## Totals",
        "",
        f"- Workbooks: {len(summary)}.",
        f"- Complete nine-feature schema: {int(summary['schema_complete'].sum())}.",
        f"- Data-ready under the current split: {int(summary['data_ready_current_split'].sum())}.",
        f"- Stations with precise/non-centroid coordinates: {int((~summary['coordinate_is_centroid']).sum())}.",
        f"- Stations with official address evidence: {int(summary['has_official_address_evidence'].sum())}.",
        f"- Duplicate groups from 2022 onward: {len(duplicates)}; conflicting: {int(duplicates['conflict_kind'].ne('exact_duplicate').sum()) if len(duplicates) else 0}.",
        "",
        "## Data-Ready Stations",
        "",
        "```text",
        ready[["station", "river", "all9_rate_after_short_interp", "train_windows_after", "val_windows_after", "test_windows_after", "coordinate_confidence"]].to_string(index=False),
        "```",
        "",
        "## Mixed-Network River Candidates",
        "",
        "```text",
        candidates[["river", "provincial_data_ready_stations", "national_stations"]].to_string(index=False),
        "```",
        "",
        "## Decision",
        "",
        "The provincial measurements are useful for adding intermediate monitoring nodes and increasing event coverage. They are not yet ready for strict graph training because station-level coordinates and directed river order remain largely unverified.",
    ]
    (OUTPUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    feature_rows = []
    duplicate_frames = []
    for path in sorted(DATA_DIR.glob("*.xls")):
        summary, features, duplicates = audit_workbook(path)
        summaries.append(summary)
        feature_rows.extend(features)
        if len(duplicates):
            duplicate_frames.append(duplicates)
    summary = attach_location_status(pd.DataFrame(summaries))
    features = pd.DataFrame(feature_rows)
    duplicates = pd.concat(duplicate_frames, ignore_index=True) if duplicate_frames else pd.DataFrame()
    rivers = same_river_inventory(summary)

    summary.sort_values(["data_ready_current_split", "train_windows_after"], ascending=False).to_csv(
        OUTPUT_DIR / "station_4h_usability.csv", index=False, encoding="utf-8-sig"
    )
    features.to_csv(OUTPUT_DIR / "station_feature_4h_coverage.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(OUTPUT_DIR / "duplicate_timestamp_audit_2022plus.csv", index=False, encoding="utf-8-sig")
    rivers.to_csv(OUTPUT_DIR / "same_river_mixed_network_candidates.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source_directory": str(DATA_DIR.relative_to(ROOT)),
        "workbook_count": len(summaries),
        "start_date": START_DATE,
        "train_end": str(TRAIN_END),
        "validation_end": str(VAL_END),
        "resample_rule": "4h",
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "hard_rules": list(OUTLIER_RULES),
        "short_interpolation_limit_steps": 3,
        "raw_files_modified": False,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, rivers, duplicates)
    console.print(OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
