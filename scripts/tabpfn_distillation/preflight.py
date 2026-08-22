#!/usr/bin/env python3
"""Audit 18-horizon label coverage and causal OOF feasibility without training."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io
from scripts.tabpfn_distillation.teacher import select_tasks


def audit_task(
    panel: pd.DataFrame, station: str, target: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    splits = data.split_by_time(data.build_station_target_dataset(panel, station, target))
    train = splits["train"]
    validation = splits["val"]
    folds = data.causal_oof_folds(train)
    coverage_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for horizon, hours in enumerate(config.HORIZON_HOURS):
        train_valid = np.asarray(train["y_mask"], dtype=bool)[:, horizon] & np.isfinite(
            np.asarray(train["y_delta"], dtype=float)[:, horizon]
        )
        val_valid = np.asarray(validation["y_mask"], dtype=bool)[:, horizon] & np.isfinite(
            np.asarray(validation["y_abs"], dtype=float)[:, horizon]
        )
        fold_fit_counts = []
        fold_prediction_counts = []
        for fold in folds:
            fit_mask = np.asarray(fold["fit_mask"], dtype=bool)
            prediction_mask = np.asarray(fold["prediction_mask"], dtype=bool)
            fit_valid = fit_mask & train_valid
            prediction_valid = prediction_mask & np.asarray(
                train["current_mask"], dtype=bool
            )[:, 0]
            fold_fit_counts.append(int(fit_valid.sum()))
            fold_prediction_counts.append(int(prediction_valid.sum()))

            fit_ends = np.asarray(train["target_end"])[fit_mask]
            prediction_starts = np.asarray(train["target_start"])[prediction_mask]
            causal = bool(
                len(fit_ends)
                and len(prediction_starts)
                and fit_ends.max() < prediction_starts.min()
            )
            fold_rows.append(
                {
                    "station": station,
                    "target": target,
                    "fold": fold["name"],
                    "horizon_hours": hours,
                    "fit_valid_labels": int(fit_valid.sum()),
                    "prediction_origins": int(prediction_valid.sum()),
                    "strictly_causal": causal,
                }
            )

        ready = bool(
            fold_fit_counts
            and min(fold_fit_counts) >= config.MIN_TEACHER_TRAIN_ROWS
            and min(fold_prediction_counts) > 0
            and int(val_valid.sum()) > 0
        )
        coverage_rows.append(
            {
                "station": station,
                "target": target,
                "horizon_hours": hours,
                "train_origins": int(len(train["target_start"])),
                "train_valid_labels": int(train_valid.sum()),
                "validation_origins": int(len(validation["target_start"])),
                "validation_valid_labels": int(val_valid.sum()),
                "oof_origins": int(sum(fold_prediction_counts)),
                "minimum_fold_fit_labels": min(fold_fit_counts) if fold_fit_counts else 0,
                "ready": ready,
            }
        )
    return coverage_rows, fold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = data.load_v2_panel()
    # The preflight must not inspect any 2025 target values.
    panel = panel.loc[pd.to_datetime(panel["time"]) < pd.Timestamp(config.VAL_END)].copy()
    try:
        stations, targets = select_tasks(
            panel, args.stations, args.targets, args.all_stations, args.all_targets
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    coverage_rows = []
    fold_rows = []
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            coverage, folds = audit_task(panel, station, target)
            coverage_rows.extend(coverage)
            fold_rows.extend(folds)

    coverage = pd.DataFrame(coverage_rows)
    fold_audit = pd.DataFrame(fold_rows)
    output_dir = config.OUTPUT_DIR / "预检"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "数据覆盖.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output_dir / "OOF因果性审计.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment": config.EXPERIMENT_ID,
        "stations": list(stations),
        "targets": list(targets),
        "horizon_hours": list(config.HORIZON_HOURS),
        "cells": int(len(coverage)),
        "ready_cells": int(coverage["ready"].sum()),
        "all_ready": bool(coverage["ready"].all()),
        "all_folds_strictly_causal": bool(fold_audit["strictly_causal"].all()),
        "test_period_inspected": False,
        **io.data_identity(),
    }
    io.save_json(output_dir / "预检摘要.json", summary)
    console.table(
        "预检摘要",
        coverage,
        columns=(
            "station",
            "target",
            "horizon_hours",
            "train_valid_labels",
            "validation_valid_labels",
            "minimum_fold_fit_labels",
            "ready",
        ),
    )
    if not summary["all_ready"] or not summary["all_folds_strictly_causal"]:
        raise SystemExit(f"预检未通过，查看 {output_dir}")
    console.done(output_dir, cells=len(coverage))


if __name__ == "__main__":
    main()
