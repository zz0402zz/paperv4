#!/usr/bin/env python3
"""Run all-station dual-branch D and compare 36h A/C/D variants."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_all_station_step_level_ablation as all_step
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol

# [01] Stage 3 A/C/D：固定 36h 输入窗口，所有结果必须来自同一 V2 根目录。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "D_dual_branch_9to1_seed42"
ALL_STATION_A_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "A_diff_only_9to1_seed42"
ALL_STATION_C_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "C_step_raw_diff_9to1_seed42"
INPUT_STEPS = 9
OUTPUT_STEPS = 1
INPUT_MODE_D = "dual_branch_current_mlp"


def steps_to_hours(input_steps: int) -> int:
    """[02] 4 小时粒度下把步数换成小时数。"""
    return int(input_steps * 4)


def dual_branch_experiment_name() -> str:
    """[03] 全站 D 方案实验名。"""
    return f"all_station_corr_top3_dual_branch_current_mlp_{INPUT_STEPS}step_delta"


def comparison_experiment_names() -> tuple[str, ...]:
    """[04] A/C/D 对比只保留 36h 三个方案。"""
    return (
        all_window.all_station_name(window.experiment_name(INPUT_STEPS, include_current_level=False)),
        all_step.experiment_name(INPUT_STEPS),
        dual_branch_experiment_name(),
    )


def _infer_input_mode(experiment: str) -> str:
    if "dual_branch_current_mlp" in experiment:
        return INPUT_MODE_D
    if "step_level" in experiment:
        return "step_raw_plus_diff"
    if "current_level" in experiment:
        return "repeated_current_target"
    if "persistence" in experiment:
        return "persistence"
    return "diff_only"


def _read_result_table(path: Path, source: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "input_mode" not in table.columns:
        table["input_mode"] = table["experiment"].map(_infer_input_mode)
    table["source"] = source
    return table


def _filter_and_order(frame: pd.DataFrame) -> pd.DataFrame:
    wanted = list(comparison_experiment_names())
    filtered = frame[frame["experiment"].isin(wanted)].copy()
    order = {name: idx for idx, name in enumerate(wanted)}
    filtered["_order"] = filtered["experiment"].map(order)
    return filtered.sort_values(["_order"]).drop(columns=["_order"]).reset_index(drop=True)


def validate_v2_comparison_sources(*directories: Path) -> None:
    manifests = []
    for directory in directories:
        path = Path(directory) / "run_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing V2 run manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("protocol_version") != "v2_reprocessed_20260710":
            raise ValueError(f"Not a V2 rerun manifest: {path}")
        manifests.append(manifest)
    observed_hashes = {item["inputs"]["observed"]["sha256"] for item in manifests}
    quality_hashes = {item["inputs"]["quality"]["sha256"] for item in manifests}
    if len(observed_hashes) != 1 or len(quality_hashes) != 1:
        raise ValueError("A/C/D comparison sources do not use identical V2 observed and quality files.")


def load_overall_comparison_rows(
    a_dir: Path = ALL_STATION_A_DIR,
    c_dir: Path = ALL_STATION_C_DIR,
    d_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """[05] 读取并合并 36h A/C/D 整体结果。"""
    validate_v2_comparison_sources(a_dir, c_dir, d_dir)
    combined = pd.concat(
        [
            _read_result_table(a_dir / "overall_summary.csv", "A_diff_only_existing"),
            _read_result_table(c_dir / "overall_summary.csv", "C_step_level_existing"),
            _read_result_table(d_dir / "overall_summary.csv", "D_dual_branch_new"),
        ],
        ignore_index=True,
    )
    return _filter_and_order(combined)


def load_metric_comparison_rows(
    filename: str,
    a_dir: Path = ALL_STATION_A_DIR,
    c_dir: Path = ALL_STATION_C_DIR,
    d_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """[06] 读取并合并 36h A/C/D 的分指标、分站点或分站点分指标表。"""
    validate_v2_comparison_sources(a_dir, c_dir, d_dir)
    combined = pd.concat(
        [
            _read_result_table(a_dir / filename, "A_diff_only_existing"),
            _read_result_table(c_dir / filename, "C_step_level_existing"),
            _read_result_table(d_dir / filename, "D_dual_branch_new"),
        ],
        ignore_index=True,
    )
    return _filter_and_order(combined)


def run_dual_branch_experiment(torch, data: pd.DataFrame, stations: tuple[str, ...], device) -> tuple[dict, list[dict[str, object]]]:
    """[07] 跑全站 D 方案：GRU 学 corr-top3 diff 序列，MLP 学当前目标值。"""
    name = dual_branch_experiment_name()
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []

    for target_feature in window.TARGET_FEATURE_COLUMNS:
        selected_features, rows = window.select_corr_features_for_target(
            data, stations, target_feature, INPUT_STEPS, name, include_current_level=False
        )
        for row in rows:
            item = dict(row)
            item["input_mode"] = INPUT_MODE_D
            selected_rows.append(item)

        input_columns = window.diff_columns_from_features(selected_features)
        raw_splits, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            INPUT_STEPS,
            include_current_level=False,
        )
        scaled_splits, current_scaler = dual.attach_scaled_current_level(raw_splits, scaled_splits)
        result, arrays_by_split = dual.fit_dual_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            current_scaler,
            stations,
            device,
        )
        result["input_mode"] = INPUT_MODE_D
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
        "input_steps": INPUT_STEPS,
        "window_hours": steps_to_hours(INPUT_STEPS),
        "include_current_level": False,
        "input_mode": INPUT_MODE_D,
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
    """[08] D 方案整体 test 指标。"""
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
                "window_hours": steps_to_hours(input_steps),
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
    """[09] D 方案分指标 test 指标。"""
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
    """[10] 保存 D 结果与 A/C/D 合并比较表。"""
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

    load_overall_comparison_rows().to_csv(OUTPUT_DIR / "comparison_overall_acd.csv", index=False, encoding="utf-8-sig")
    load_metric_comparison_rows("feature_metrics.csv").to_csv(
        OUTPUT_DIR / "comparison_feature_acd.csv", index=False, encoding="utf-8-sig"
    )
    load_metric_comparison_rows("station_metrics.csv").to_csv(
        OUTPUT_DIR / "comparison_station_acd.csv", index=False, encoding="utf-8-sig"
    )
    load_metric_comparison_rows("station_feature_metrics.csv").to_csv(
        OUTPUT_DIR / "comparison_station_feature_acd.csv", index=False, encoding="utf-8-sig"
    )


def write_report() -> None:
    """[11] 写 A/C/D 对比报告。"""
    overall = load_overall_comparison_rows().sort_values("test_rmse")
    feature_best = (
        load_metric_comparison_rows("feature_metrics.csv")
        .sort_values("test_rmse")
        .groupby("feature", as_index=False)
        .first()
    )
    lines = [
        "# 全站 36h 输入 A/C/D 单步对比",
        "",
        "## 口径",
        "- A：corr-top3 diff-only，读取本轮 V2 全站 36h 输入结果。",
        "- C：每个历史步 raw level + diff1，读取本轮 V2 全站 36h 输入结果。",
        "- D：corr-top3 diff 序列走 GRU，当前目标值走小 MLP，本脚本新训练。",
        "- 输出：未来 1 个 4 小时时间步 delta，最终预测值 = 当前目标值 + 预测 delta。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## 分指标最佳",
        "```text",
        feature_best[
            ["feature", "experiment", "input_mode", "test_mae", "test_rmse", "test_nse"]
        ].to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def _json_safe_results(results: dict[str, dict]) -> dict[str, dict]:
    safe = {}
    for name, result in results.items():
        safe[name] = {
            "input_steps": result["input_steps"],
            "window_hours": result["window_hours"],
            "input_mode": result.get("input_mode", ""),
            "best_epoch": result["best_epoch"],
            "best_checkpoint": result["best_checkpoint"],
            "targets": {
                target: {
                    "best_epoch": target_result["best_epoch"],
                    "best_model_path": target_result["best_model_path"],
                    "input_columns": target_result["input_columns"],
                    "input_mode": target_result.get("input_mode", result.get("input_mode", "")),
                }
                for target, target_result in result["targets"].items()
            },
        }
    return safe


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = protocol.PILOT_SEED) -> int:
    """[12] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    dual.OUTPUT_DIR = output_dir
    paper.SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    dataset_summary = {
        "model": "all_station_corr_top3_dual_branch_current_mlp_ablation",
        "output_dir": str(OUTPUT_DIR),
        "processed_data_path": str(all_window.PROCESSED_DATA_PATH),
        "start_date": all_window.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": all_window.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "window_hours": steps_to_hours(INPUT_STEPS),
        "output_steps": OUTPUT_STEPS,
        "target_features": list(window.TARGET_FEATURE_COLUMNS),
        "station_count": len(stations),
        "stations": list(stations),
        "experiments": [dual_branch_experiment_name()],
        "comparison_experiments": list(comparison_experiment_names()),
        "existing_a_results_dir": str(ALL_STATION_A_DIR),
        "existing_c_results_dir": str(ALL_STATION_C_DIR),
        "seed": int(seed),
    }
    manifest = protocol.build_run_manifest(
        experiment="stage3_D_dual_branch_9to1",
        output_dir=OUTPUT_DIR,
        seed=seed,
        code_paths=(Path("scripts/gru/run_all_station_dual_branch_delta_gru.py"),),
    )
    dataset_summary["run_manifest"] = manifest
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    base.save_json(OUTPUT_DIR / "run_manifest.json", manifest)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = base.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)

    result, selected_rows = run_dual_branch_experiment(torch, data, stations, device)
    results = {result["experiment"]: result}
    persistence_by_steps = {INPUT_STEPS: window.persistence_for_window(data, stations, INPUT_STEPS)}
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence_by_steps,
        "experiments": _json_safe_results(results),
        "selected_features": selected_rows,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence_by_steps, selected_rows)
    write_report()

    console.print(load_overall_comparison_rows().sort_values("test_rmse").to_string(index=False), flush=True)
    console.print(load_metric_comparison_rows("feature_metrics.csv").to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
