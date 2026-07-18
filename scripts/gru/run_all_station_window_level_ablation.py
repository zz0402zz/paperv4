#!/usr/bin/env python3
"""Validate window/current-level diff-delta GRU experiments on all stations."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import add_feature_enhancements, load_or_build_4h_quantity_data


# [01] Stage 3 A：只跑 36h diff-only，避免混入已淘汰组合。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "A_diff_only_9to1_seed42"
WINDOW_STEPS: tuple[int, ...] = (9,)
INCLUDE_CURRENT_LEVEL_OPTIONS: tuple[bool, ...] = (False,)
DATA_DIR = base.DATA_DIR
PROCESSED_DATA_PATH = base.PROCESSED_DATA_PATH
START_DATE = paper.START_DATE
RESAMPLE_RULE = paper.RESAMPLE_RULE
DROP_OUTLIERS = paper.DROP_OUTLIERS
REBUILD_PROCESSED_DATA = paper.REBUILD_PROCESSED_DATA


def all_station_name(name: str) -> str:
    """[02] 给实验名加全站前缀。"""
    return f"all_station_{name}"


def experiment_names() -> tuple[str, ...]:
    """[03] 返回本轮全站验证要跑的实验名。"""
    return tuple(
        all_station_name(window.experiment_name(input_steps, include_current_level))
        for input_steps in WINDOW_STEPS
        for include_current_level in INCLUDE_CURRENT_LEVEL_OPTIONS
    )


def rename_result_for_all_stations(result: dict) -> dict:
    """[04] 将复用函数产生的三站实验名改成全站实验名。"""
    renamed = dict(result)
    old_name = str(result["experiment"])
    new_name = all_station_name(old_name)
    renamed["experiment"] = new_name
    renamed["history"] = []
    for row in result.get("history", []):
        item = dict(row)
        if "sub_experiment" in item:
            item["sub_experiment"] = str(item["sub_experiment"]).replace(old_name, new_name, 1)
        renamed["history"].append(item)
    return renamed


def rename_selected_rows_for_all_stations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """[05] selected_features 表同步使用全站实验名。"""
    renamed = []
    for row in rows:
        item = dict(row)
        item["experiment"] = all_station_name(str(item["experiment"]))
        renamed.append(item)
    return renamed


def load_all_station_diff1_data() -> pd.DataFrame:
    """[06] 读取全站 START_DATE 起 4h 清洗数据，并按站点加入 diff1。"""
    data = load_or_build_4h_quantity_data(
        DATA_DIR,
        PROCESSED_DATA_PATH,
        START_DATE,
        RESAMPLE_RULE,
        DROP_OUTLIERS,
        REBUILD_PROCESSED_DATA,
    )
    return add_feature_enhancements(data, ("diff1",))


def write_report(results: dict[str, dict], persistence_by_steps: dict[int, dict], station_count: int) -> None:
    """[07] 写全站验证报告。"""
    overall = pd.DataFrame(window.overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    lines = [
        "# 全站 A 方案：36h diff-only 单步预测",
        "",
        f"- 站点数：{station_count}",
        "- 窗口：9 步，对应过去 36 小时。",
        "- 每个目标使用训练集 corr-top3 diff 特征。",
        "- A 方案只输入 diff 序列。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = protocol.PILOT_SEED) -> int:
    """[08] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    window.OUTPUT_DIR = output_dir
    paper.SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    dataset_summary = {
        "model": "all_station_per_target_corr_top3_window_and_current_level_ablation",
        "output_dir": str(OUTPUT_DIR),
        "processed_data_path": str(PROCESSED_DATA_PATH),
        "start_date": START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": RESAMPLE_RULE,
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
        experiment="stage3_A_diff_only_9to1",
        output_dir=OUTPUT_DIR,
        seed=seed,
        code_paths=(Path("scripts/gru/run_all_station_window_level_ablation.py"),),
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
        for include_current_level in INCLUDE_CURRENT_LEVEL_OPTIONS:
            result, rows = window.run_one_experiment(torch, data, stations, input_steps, include_current_level, device)
            renamed = rename_result_for_all_stations(result)
            results[renamed["experiment"]] = renamed
            selected_rows.extend(rename_selected_rows_for_all_stations(rows))

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
    window.save_tables(results, stations, persistence_by_steps, selected_rows)
    write_report(results, persistence_by_steps, len(stations))
    overall = pd.DataFrame(window.overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    console.print(overall.to_string(index=False), flush=True)
    console.print(pd.DataFrame(window.feature_metric_rows(results)).to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
