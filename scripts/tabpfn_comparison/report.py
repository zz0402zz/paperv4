#!/usr/bin/env python3
"""Aggregate the causal single-station short-history comparison results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tabpfn_comparison import config, data, io, run


def _metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return {"valid_points": 0, "mae": None, "rmse": None, "nse": None}
    error = np.asarray(pred, dtype=float)[valid] - np.asarray(true, dtype=float)[valid]
    truth = np.asarray(true, dtype=float)[valid]
    denominator = float(np.square(truth - truth.mean()).sum())
    return {
        "valid_points": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "nse": None if denominator <= np.finfo(float).eps else float(1.0 - np.square(error).sum() / denominator),
    }


def _expected(
    evaluation_split: str,
    model: str,
    seed: int,
    target: str,
    station: str,
) -> dict[str, object]:
    fit_splits = ("train",) if evaluation_split == "val" else ("train", "val")
    return run.task_metadata(
        evaluation_split=evaluation_split,
        fit_splits=fit_splits,
        model=model,
        seed=seed,
        target=target,
        station=station,
    )


def cells_for_model(
    evaluation_split: str,
    model: str,
    stations: tuple[str, ...],
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Calculate one station-target-horizon metric row per saved prediction."""
    rows: list[dict[str, object]] = []
    for seed in config.model_seeds(model, seeds):
        for station in stations:
            for target in targets:
                path = io.prediction_path(evaluation_split, model, seed, target, station)
                expected = _expected(evaluation_split, model, seed, target, station)
                if not io.is_complete(path, expected):
                    raise FileNotFoundError(
                        f"Missing or incompatible prediction: {path}. Run the same protocol first."
                    )
                arrays, _ = io.load_prediction(path)
                for horizon in range(config.OUTPUT_STEPS):
                    rows.append(
                        {
                            "variant": model,
                            "seed": seed,
                            "station": station,
                            "target": target,
                            "horizon_step": horizon + 1,
                            "horizon_hours": (horizon + 1) * config.STEP_HOURS,
                            **_metrics(
                                arrays["pred"][:, horizon],
                                arrays["true"][:, horizon],
                                arrays["mask"][:, horizon],
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use equal station-target-horizon weights, not raw observation counts."""
    summaries: dict[str, pd.DataFrame] = {}
    for name, groups in (
        ("overall", ["variant"]),
        ("by_station", ["station", "variant"]),
        ("by_target", ["target", "variant"]),
        ("by_horizon", ["horizon_hours", "variant"]),
        ("by_seed", ["seed", "variant"]),
    ):
        summaries[name] = (
            cells.groupby(groups, as_index=False)
            .agg(
                macro_rmse=("rmse", "mean"),
                macro_mae=("mae", "mean"),
                macro_nse=("nse", "mean"),
                valid_points=("valid_points", "sum"),
            )
            .sort_values(groups)
            .reset_index(drop=True)
        )
    return summaries


def tabpfn_vs_gru(cells: pd.DataFrame) -> pd.DataFrame:
    """Pair only identical station, target, horizon, and random-seed cells."""
    keys = ["seed", "station", "target", "horizon_step", "horizon_hours"]
    left = cells.loc[cells["variant"].eq(config.DELTA_TABPFN_KEY), keys + ["rmse"]].rename(
        columns={"rmse": "tabpfn_rmse"}
    )
    right = cells.loc[cells["variant"].eq(config.DELTA_GRU_KEY), keys + ["rmse"]].rename(
        columns={"rmse": "gru_rmse"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return pd.DataFrame(
            columns=["scope", "cells", "tabpfn_macro_rmse", "gru_macro_rmse", "difference", "relative_pct", "tabpfn_win_rate"]
        )
    paired["difference"] = paired["tabpfn_rmse"] - paired["gru_rmse"]
    paired["tabpfn_win"] = paired["difference"] < 0
    rows: list[dict[str, object]] = []
    for scope, frame in [("overall", paired), *[(f"target:{target}", group) for target, group in paired.groupby("target", sort=True)]]:
        tabpfn_value = float(frame["tabpfn_rmse"].mean())
        gru_value = float(frame["gru_rmse"].mean())
        rows.append(
            {
                "scope": scope,
                "cells": int(len(frame)),
                "tabpfn_macro_rmse": tabpfn_value,
                "gru_macro_rmse": gru_value,
                "difference": float(frame["difference"].mean()),
                "relative_pct": None if gru_value == 0 else (tabpfn_value / gru_value - 1.0) * 100.0,
                "tabpfn_win_rate": float(frame["tabpfn_win"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    evaluation_split: str,
    stations: tuple[str, ...],
    targets: tuple[str, ...],
    summaries: dict[str, pd.DataFrame],
    paired: pd.DataFrame,
) -> None:
    lines = [
        "# 单站短历史 TabPFN 与变化量 GRU 对比",
        "",
        "## 协议",
        "",
        f"- 评估集：`{evaluation_split}`；拟合集：`{'train' if evaluation_split == 'val' else 'train + val'}`。",
        f"- 每个站点和预测目标独立建模；站点数：{len(stations)}，目标数：{len(targets)}。",
        f"- 输入：过去 {config.INPUT_STEPS} 个 {config.STEP_HOURS} 小时时间步（{config.INPUT_STEPS * config.STEP_HOURS} 小时）。",
        f"- 输出：未来 {config.OUTPUT_STEPS} 个时间步；主时距为 {config.STEP_HOURS} 小时。",
        "- 两个学习模型使用相同的本站原值、变化量、缺失掩码和当前目标值；没有跨站特征。",
        "- TabPFN 和 GRU都预测变化量，随后加回当前目标值；持久性直接延用当前值。",
        "- 正式标签仅使用 V2 质量侧表批准的原始观测；重建审阅值不进入输入或标签。",
        "- 宏平均对站点 × 目标 × 时距单元等权；较低 RMSE 更好。",
    ]
    for title, key in (
        ("总体", "overall"),
        ("分站点", "by_station"),
        ("分指标", "by_target"),
        ("分时距", "by_horizon"),
        ("分随机种子", "by_seed"),
    ):
        lines.extend(["", f"## {title}", "", "```text", summaries[key].to_string(index=False), "```"])
    lines.extend(["", "## TabPFN 相对 GRU", "", "```text", paired.to_string(index=False), "```"])
    lines.append("")
    (output_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def _selection(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    panel = data.load_v2_panel()
    return run._selection(panel, args.stations, args.targets, args.all_stations, args.all_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="Comma-separated V2 station names.")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets", help="Comma-separated official targets.")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, config.FORMAL_SEEDS)))
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        stations, targets = _selection(args)
        seeds = run._parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cells = pd.concat(
        [
            cells_for_model(args.evaluation_split, model, stations, targets, seeds)
            for model in config.MODEL_KEYS
        ],
        ignore_index=True,
    )
    summaries = summarize(cells)
    paired = tabpfn_vs_gru(cells)
    output_dir = config.output_dir_for_split(args.evaluation_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "站点指标时距明细.csv", index=False, encoding="utf-8-sig")
    summary_filenames = {
        "overall": "总体比较.csv",
        "by_station": "分站点比较.csv",
        "by_target": "分指标比较.csv",
        "by_horizon": "分时距比较.csv",
        "by_seed": "分随机种子比较.csv",
    }
    for name, frame in summaries.items():
        frame.to_csv(output_dir / summary_filenames[name], index=False, encoding="utf-8-sig")
    paired.to_csv(output_dir / "TabPFN与GRU对比.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, args.evaluation_split, stations, targets, summaries, paired)


if __name__ == "__main__":
    main()
