#!/usr/bin/env python3
"""Validate per-step raw level plus diff inputs on all stations."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol

# [01] Stage 3 C：每个历史步输入该步 raw level + diff1，固定过去 36h。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "C_step_raw_diff_9to1_seed42"
WINDOW_STEPS: tuple[int, ...] = (9,)


def raw_diff_columns_from_features(features: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """[02] 每个选中特征同时传原始值和该步 diff1。"""
    features = tuple(features)
    return (*features, *(f"{feature}_diff1" for feature in features))


def experiment_name(input_steps: int) -> str:
    """[03] C 方案实验名。"""
    return f"all_station_corr_top3_step_level_{input_steps}step_delta"


def experiment_names() -> tuple[str, ...]:
    """[04] 返回本轮要跑的实验名。"""
    return tuple(experiment_name(input_steps) for input_steps in WINDOW_STEPS)


def run_one_step_level_experiment(
    torch,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_steps: int,
    device,
) -> tuple[dict, list[dict[str, object]]]:
    """[05] 每个目标先 corr-top3 选 diff 特征，再传 raw+diff 序列训练。"""
    name = experiment_name(input_steps)
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []

    for target_feature in window.TARGET_FEATURE_COLUMNS:
        selected_features, rows = window.select_corr_features_for_target(
            data,
            stations,
            target_feature,
            input_steps,
            name,
            include_current_level=False,
        )
        for row in rows:
            item = dict(row)
            item["input_mode"] = "step_raw_plus_diff"
            selected_rows.append(item)

        input_columns = raw_diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            input_steps,
            include_current_level=False,
        )
        result, arrays_by_split = window.fit_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            stations,
            input_steps,
            include_current_level=False,
            device=device,
        )
        result["input_mode"] = "step_raw_plus_diff"
        target_results[target_feature] = result
        history_rows.extend({"sub_experiment": f"{name}_{target_feature}", **row} for row in result["history"])
        for split_name, arrays in arrays_by_split.items():
            arrays_by_split_and_target[split_name][target_feature] = arrays

    metrics = {
        split_name: window.self_ablation.aggregate_single_target_arrays(
            arrays_by_target,
            stations,
            window.TARGET_FEATURE_COLUMNS,
        )
        for split_name, arrays_by_target in arrays_by_split_and_target.items()
    }
    best_epochs = {target: result["best_epoch"] for target, result in target_results.items()}
    return {
        "experiment": name,
        "input_steps": input_steps,
        "window_hours": window.steps_to_hours(input_steps),
        "include_current_level": False,
        "input_mode": "step_raw_plus_diff",
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, selected_rows


def overall_rows(results: dict[str, dict], persistence_by_steps: dict[int, dict]) -> list[dict[str, object]]:
    """[06] 整体指标表，增加 input_mode 字段。"""
    rows = []
    for experiment, result in results.items():
        test = result["best_checkpoint"]["test"]
        rows.append(
            {
                "experiment": experiment,
                "input_steps": result["input_steps"],
                "window_hours": result["window_hours"],
                "input_mode": result.get("input_mode", ""),
                "val_rmse": result["best_epoch"].get("val_rmse"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    for input_steps, test in persistence_by_steps.items():
        rows.append(
            {
                "experiment": f"persistence_{input_steps}step",
                "input_steps": input_steps,
                "window_hours": window.steps_to_hours(input_steps),
                "input_mode": "persistence",
                "val_rmse": "",
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    return rows


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[07] 分指标 test 指标。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]
        for feature in window.TARGET_FEATURE_COLUMNS:
            rows.append(
                {
                    "experiment": experiment,
                    "input_steps": result["input_steps"],
                    "window_hours": result["window_hours"],
                    "input_mode": result.get("input_mode", ""),
                    "feature": feature,
                    "valid_points": metrics["feature_valid_points"].get(feature, 0),
                    "test_mae": metrics["feature_mae"].get(feature),
                    "test_rmse": metrics["feature_rmse"].get(feature),
                    "test_nse": metrics["feature_nse"].get(feature),
                }
            )
    return rows


def save_tables(
    results: dict[str, dict],
    stations: tuple[str, ...],
    persistence_by_steps: dict[int, dict],
    selected_rows: list[dict[str, object]],
) -> None:
    """[08] 保存结果表。"""
    pd.DataFrame(overall_rows(results, persistence_by_steps)).to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(feature_metric_rows(results)).to_csv(
        OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(window.station_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(window.station_feature_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def write_report(results: dict[str, dict], persistence_by_steps: dict[int, dict], station_count: int) -> None:
    """[09] 写简短报告。"""
    overall = pd.DataFrame(overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    lines = [
        "# 全站 step-level raw+diff 输入验证",
        "",
        f"- 站点数：{station_count}",
        "- 窗口：9 步，对应过去 36 小时。",
        "- 每个目标使用训练集 corr-top3 筛出的特征。",
        "- 每个历史步输入该步 raw level + diff1。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = protocol.PILOT_SEED) -> int:
    """[10] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    window.OUTPUT_DIR = output_dir
    paper.SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    dataset_summary = {
        "model": "all_station_per_target_corr_top3_step_level_raw_diff_ablation",
        "output_dir": str(OUTPUT_DIR),
        "processed_data_path": str(all_window.PROCESSED_DATA_PATH),
        "start_date": all_window.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": all_window.RESAMPLE_RULE,
        "window_steps": list(WINDOW_STEPS),
        "window_hours": [window.steps_to_hours(steps) for steps in WINDOW_STEPS],
        "output_steps": window.OUTPUT_STEPS,
        "target_features": list(window.TARGET_FEATURE_COLUMNS),
        "station_count": len(stations),
        "stations": list(stations),
        "experiments": list(experiment_names()),
        "seed": int(seed),
    }
    manifest = protocol.build_run_manifest(
        experiment="stage3_C_step_raw_diff_9to1",
        output_dir=OUTPUT_DIR,
        seed=seed,
        code_paths=(Path("scripts/gru/run_all_station_step_level_ablation.py"),),
    )
    dataset_summary["run_manifest"] = manifest
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    base.save_json(OUTPUT_DIR / "run_manifest.json", manifest)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = base.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    selected_rows = []
    for input_steps in WINDOW_STEPS:
        result, rows = run_one_step_level_experiment(torch, data, stations, input_steps, device)
        results[result["experiment"]] = result
        selected_rows.extend(rows)

    persistence_by_steps = {
        input_steps: window.persistence_for_window(data, stations, input_steps) for input_steps in WINDOW_STEPS
    }
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence_by_steps,
        "experiments": window._json_safe_results(results),
        "selected_features": selected_rows,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence_by_steps, selected_rows)
    write_report(results, persistence_by_steps, len(stations))
    overall = pd.DataFrame(overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    console.print(overall.to_string(index=False), flush=True)
    console.print(pd.DataFrame(feature_metric_rows(results)).to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
