#!/usr/bin/env python3
"""Validate the frozen V2 experiment protocol before rerunning models."""

from __future__ import annotations

from scripts.common.terminal_output import console

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.baselines import gat_gru_baseline as graph
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import load_processed_4h_data
from scripts.data.wq_preprocessing_v2 import write_json


OUTPUT_DIR = protocol.PROTOCOL_OUTPUT_ROOT / "stage1_protocol"
FLAGS_PATH = Path("data/processed/v2/quantity_4h_reconstruction_flags.csv")


def split_name(target_time: pd.Timestamp) -> str:
    target_time = pd.Timestamp(target_time)
    if target_time < pd.Timestamp(protocol.TRAIN_END):
        return "train"
    if target_time < pd.Timestamp(protocol.VAL_END):
        return "val"
    return "test"


def count_reconstructed_approved(
    quality: pd.DataFrame,
    flags: pd.DataFrame,
    features: tuple[str, ...],
) -> int:
    joined = quality.merge(flags, on=["station", "time"], how="inner", validate="one_to_one")
    return sum(
        int(
            (
                joined[f"{feature}__status"].eq("reconstructed")
                & joined[f"{feature}__target_ok"].fillna(False).astype(bool)
            ).sum()
        )
        for feature in features
    )


def delta_coverage(frame: pd.DataFrame, features: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    work = frame.sort_values(["station", "time"]).copy()
    for feature in features:
        approved = work[f"{feature}__target_ok"].fillna(False).astype(bool) & work[feature].notna()
        anchor_ok = approved.groupby(work["station"], sort=False).shift(1, fill_value=False)
        consecutive = work["time"].sub(work.groupby("station", sort=False)["time"].shift()).eq(
            pd.Timedelta(hours=4)
        )
        valid_delta = approved & anchor_ok & consecutive
        rows.append(
            {
                "feature": feature,
                "approved_absolute_targets": int(approved.sum()),
                "approved_delta_targets": int(valid_delta.sum()),
            }
        )
    return rows


def split_target_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = frame[frame["time"] >= pd.Timestamp(protocol.START_DATE)].copy()
    work["split"] = work["time"].map(split_name)
    for (split, station), group in work.groupby(["split", "station"], sort=True):
        for feature in protocol.TARGET_FEATURE_COLUMNS:
            approved = group[f"{feature}__target_ok"].fillna(False).astype(bool) & group[feature].notna()
            rows.append(
                {
                    "split": split,
                    "station": station,
                    "feature": feature,
                    "time_rows": int(len(group)),
                    "approved_targets": int(approved.sum()),
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_preflight(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = pd.read_csv(protocol.OBSERVED_DATA_PATH, parse_dates=["time"])
    quality = pd.read_csv(protocol.QUALITY_DATA_PATH, parse_dates=["time"])
    flags = pd.read_csv(FLAGS_PATH, parse_dates=["time"])
    loaded = load_processed_4h_data(protocol.OBSERVED_DATA_PATH)
    loaded = loaded[loaded["time"] >= pd.Timestamp(protocol.START_DATE)].reset_index(drop=True)

    key_columns = ["station", "time"]
    same_keys = values[key_columns].equals(quality[key_columns]) and values[key_columns].equals(flags[key_columns])
    unique_keys = all(not frame.duplicated(key_columns).any() for frame in (values, quality, flags))
    aligned_4h = bool(
        (
            values["time"].dt.minute.eq(0)
            & values["time"].dt.second.eq(0)
            & values["time"].dt.hour.mod(4).eq(0)
        ).all()
    )
    reconstructed_approved = count_reconstructed_approved(
        quality,
        flags,
        tuple(protocol.INPUT_FEATURE_COLUMNS),
    )

    metadata = json.loads(protocol.PREPROCESSING_METADATA_PATH.read_text(encoding="utf-8"))
    output_hashes_match = (
        metadata["output_sha256"]["observed_values"] == _sha256(protocol.OBSERVED_DATA_PATH)
        and metadata["output_sha256"]["quality"] == _sha256(protocol.QUALITY_DATA_PATH)
        and metadata["output_sha256"]["reconstruction_flags"] == _sha256(FLAGS_PATH)
    )

    stations = tuple(sorted(loaded["station"].dropna().astype(str).unique()))
    edges = pd.read_csv(protocol.STRICT_EDGES_PATH)
    edge_nodes = set(edges["source_station"].astype(str)) | set(edges["target_station"].astype(str))
    graph_nodes_valid = edge_nodes.issubset(stations)
    dataset = graph.build_graph_dataset(
        loaded,
        stations=stations,
        input_steps=9,
        output_steps=1,
        input_columns=tuple(protocol.INPUT_FEATURE_COLUMNS),
        target_columns=tuple(protocol.TARGET_FEATURE_COLUMNS),
        freq=protocol.RESAMPLE_RULE,
    )
    splits = graph.split_graph_by_time(dataset, protocol.TRAIN_END, protocol.VAL_END)

    coverage = split_target_coverage(loaded)
    coverage.to_csv(output_dir / "split_target_coverage.csv", index=False, encoding="utf-8-sig")
    delta_rows = delta_coverage(loaded, tuple(protocol.TARGET_FEATURE_COLUMNS))
    pd.DataFrame(delta_rows).to_csv(output_dir / "delta_target_coverage.csv", index=False, encoding="utf-8-sig")

    checks = {
        "unique_station_time_keys": unique_keys,
        "value_quality_flag_keys_match": same_keys,
        "four_hour_alignment": aligned_4h,
        "preprocessing_output_hashes_match": output_hashes_match,
        "reconstructed_targets_accepted": reconstructed_approved,
        "strict_edge_nodes_exist_in_data": graph_nodes_valid,
    }
    failures = [
        name
        for name, value in checks.items()
        if (name == "reconstructed_targets_accepted" and value != 0)
        or (name != "reconstructed_targets_accepted" and value is not True)
    ]
    manifest = protocol.build_run_manifest(
        experiment="stage1_preflight",
        output_dir=output_dir,
        code_paths=(Path(__file__).relative_to(Path.cwd()), Path("scripts/common/v2_experiment_protocol.py")),
    )
    summary = {
        "manifest": manifest,
        "checks": checks,
        "failures": failures,
        "data": {
            "stations": len(stations),
            "rows_from_2022": int(len(loaded)),
            "start": str(loaded["time"].min()),
            "end": str(loaded["time"].max()),
            "strict_edges": int(len(edges)),
        },
        "nine_to_one_windows": {
            split: {
                "windows": int(len(part["x"])),
                "valid_target_cells": int(part["y_mask"].sum()),
            }
            for split, part in splits.items()
        },
        "delta_target_coverage": delta_rows,
    }
    write_json(output_dir / "preflight.json", summary)

    report_lines = [
        "# V2 Stage 1 Preflight",
        "",
        f"- Stations: {len(stations)}",
        f"- Rows from 2022: {len(loaded)}",
        f"- Strict direct edges: {len(edges)}",
        f"- Reconstructed cells accepted as targets: {reconstructed_approved}",
        f"- Checks passed: {len(checks) - len(failures)}/{len(checks)}",
        "",
        "## Nine-to-one windows",
        "",
        "```text",
        pd.DataFrame(
            [{"split": split, **metrics} for split, metrics in summary["nine_to_one_windows"].items()]
        ).to_string(index=False),
        "```",
        "",
        "## Delta coverage",
        "",
        "```text",
        pd.DataFrame(delta_rows).to_string(index=False),
        "```",
    ]
    (output_dir / "run_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    if failures:
        raise RuntimeError(f"V2 preflight failed: {failures}")
    return summary


def main() -> int:
    summary = run_preflight()
    console.print(json.dumps(summary["checks"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
