#!/usr/bin/env python3
"""Run and summarize the five-seed hourly representation ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import v2_experiment_protocol as protocol
from scripts.gru import run_v2_hourly_representation_ablation as experiment


OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3d_hourly_representation_ablation" / "formal_multiseed"
SEEDS = protocol.FORMAL_SEEDS
FORCE = False


def report_seed(seed: int, output_dir: Path, status: str) -> None:
    overall = pd.read_csv(output_dir / "overall_metrics.csv")
    validation = overall[
        overall["split"].eq("val") & overall["mode"].isin(experiment.MODES)
    ].sort_values("macro_station_rmse")
    winner = validation.iloc[0]
    console.info(
        status,
        seed=seed,
        winner=experiment.TERMINAL_MODE_NAMES[str(winner["mode"])],
        val_macro_rmse=float(winner["macro_station_rmse"]),
    )


def run_seed(seed: int) -> Path:
    output_dir = OUTPUT_DIR / f"seed_{seed}"
    expected = output_dir / "overall_metrics.csv"
    if expected.exists() and not FORCE:
        report_seed(seed, output_dir, "reused")
        return output_dir
    console.info("running", seed=seed)
    with console.muted():
        experiment.run_suite(output_dir=output_dir, seed=seed)
    report_seed(seed, output_dir, "finished")
    return output_dir


def load_seed_tables(seed_dirs: dict[int, Path], filename: str) -> pd.DataFrame:
    parts = []
    for seed, directory in seed_dirs.items():
        frame = pd.read_csv(directory / filename)
        frame.insert(0, "seed", seed)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def summarize_overall(overall: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "mae",
        "rmse",
        "nse",
        "macro_station_rmse",
        "skill_vs_persistence_pct",
        "delta_mae",
        "delta_rmse",
        "delta_nse",
    ]
    return overall.groupby(["split", "mode"], as_index=False)[numeric].agg(["mean", "std"]).reset_index()


def seed_comparisons(overall: pd.DataFrame) -> pd.DataFrame:
    pivot = overall.pivot(index=["seed", "split"], columns="mode", values="macro_station_rmse").reset_index()
    pivot["endpoint_vs_mean_pct"] = (
        (pivot["mean_history"] - pivot["endpoint_history"]) / pivot["mean_history"] * 100
    )
    pivot["stats_vs_mean_pct"] = (
        (pivot["mean_history"] - pivot["endpoint_plus_window_stats"]) / pivot["mean_history"] * 100
    )
    pivot["stats_vs_endpoint_pct"] = (
        (pivot["endpoint_history"] - pivot["endpoint_plus_window_stats"]) / pivot["endpoint_history"] * 100
    )
    pivot["aligned_stats_vs_shifted_pct"] = (
        (pivot["endpoint_plus_shifted_window_stats"] - pivot["endpoint_plus_window_stats"])
        / pivot["endpoint_plus_shifted_window_stats"]
        * 100
    )
    return pivot


def station_win_summary(stations: pd.DataFrame) -> pd.DataFrame:
    pivot = stations.pivot(index=["seed", "split", "station"], columns="mode", values="rmse").reset_index()
    comparisons = {
        "endpoint_beats_mean": ("endpoint_history", "mean_history"),
        "stats_beats_mean": ("endpoint_plus_window_stats", "mean_history"),
        "stats_beats_endpoint": ("endpoint_plus_window_stats", "endpoint_history"),
        "aligned_stats_beats_shifted": (
            "endpoint_plus_window_stats",
            "endpoint_plus_shifted_window_stats",
        ),
    }
    rows = []
    for (seed, split), group in pivot.groupby(["seed", "split"]):
        for comparison, (left, right) in comparisons.items():
            valid = group[left].notna() & group[right].notna()
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "comparison": comparison,
                    "wins": int((group.loc[valid, left] < group.loc[valid, right]).sum()),
                    "stations": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    numeric = ["mae", "rmse", "nse", "skill_vs_persistence_pct", "delta_mae", "delta_rmse", "delta_nse"]
    return features.groupby(["split", "mode", "feature"], as_index=False)[numeric].agg(["mean", "std"]).reset_index()


def write_report(
    overall: pd.DataFrame,
    comparisons: pd.DataFrame,
    station_wins: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    validation = (
        overall[overall["split"].eq("val")]
        .groupby("mode", as_index=False)
        .agg(
            macro_station_rmse_mean=("macro_station_rmse", "mean"),
            macro_station_rmse_std=("macro_station_rmse", "std"),
            pooled_rmse_mean=("rmse", "mean"),
            pooled_rmse_std=("rmse", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
            nse_mean=("nse", "mean"),
        )
        .sort_values("macro_station_rmse_mean")
    )
    candidates = validation[validation["mode"].isin(experiment.MODES)]
    winner = str(candidates.iloc[0]["mode"])
    test = (
        overall[overall["split"].eq("test")]
        .groupby("mode", as_index=False)
        .agg(
            macro_station_rmse_mean=("macro_station_rmse", "mean"),
            macro_station_rmse_std=("macro_station_rmse", "std"),
            pooled_rmse_mean=("rmse", "mean"),
            pooled_rmse_std=("rmse", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
            nse_mean=("nse", "mean"),
        )
        .sort_values("macro_station_rmse_mean")
    )
    validation_comparisons = comparisons[comparisons["split"].eq("val")]
    test_comparisons = comparisons[comparisons["split"].eq("test")]
    validation_wins = station_wins[station_wins["split"].eq("val")].groupby("comparison")["wins"].agg(
        ["mean", "min", "max"]
    )
    feature_validation = (
        features[features["split"].eq("val")]
        .groupby(["mode", "feature"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
        .sort_values(["feature", "rmse_mean"])
    )
    lines = [
        "# 小时数据表示五随机种子正式消融",
        "",
        "## 口径",
        f"- 随机种子：{list(SEEDS)}。",
        "- 三个候选方案与一个同维度过去错位负对照，使用共同端点标签。",
        "- 验证集先确定方案，再读取测试集；主指标为25站宏平均 RMSE。",
        "",
        "## 验证集决策",
        f"- 五种子均值最优候选方案：`{winner}`。",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## 每种子相对变化（正数表示前者改善）",
        "```text",
        validation_comparisons.to_string(index=False),
        "```",
        "",
        "## 验证集分站点胜场",
        "```text",
        validation_wins.to_string(),
        "```",
        "",
        "## 锁定后的测试集",
        "```text",
        test.to_string(index=False),
        "```",
        "",
        "## 测试集每种子相对变化",
        "```text",
        test_comparisons.to_string(index=False),
        "```",
        "",
        "## 验证集分指标",
        "```text",
        feature_validation.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "formal_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.phase("hourly representation five-seed ablation")
    console.info("configuration", seeds=list(SEEDS), modes=len(experiment.ALL_MODES))
    seed_dirs = {seed: run_seed(seed) for seed in SEEDS}
    overall = load_seed_tables(seed_dirs, "overall_metrics.csv")
    features = load_seed_tables(seed_dirs, "feature_metrics.csv")
    stations = load_seed_tables(seed_dirs, "station_metrics.csv")
    station_features = load_seed_tables(seed_dirs, "station_feature_metrics.csv")
    tails = load_seed_tables(seed_dirs, "tail_change_metrics.csv")

    comparisons = seed_comparisons(overall)
    station_wins = station_win_summary(stations)
    overall.to_csv(OUTPUT_DIR / "all_seed_overall_metrics.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUTPUT_DIR / "all_seed_feature_metrics.csv", index=False, encoding="utf-8-sig")
    stations.to_csv(OUTPUT_DIR / "all_seed_station_metrics.csv", index=False, encoding="utf-8-sig")
    station_features.to_csv(
        OUTPUT_DIR / "all_seed_station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    tails.to_csv(OUTPUT_DIR / "all_seed_tail_change_metrics.csv", index=False, encoding="utf-8-sig")
    summarize_overall(overall).to_csv(OUTPUT_DIR / "overall_multiseed_summary.csv", index=False, encoding="utf-8-sig")
    feature_summary(features).to_csv(OUTPUT_DIR / "feature_multiseed_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUTPUT_DIR / "seed_comparisons.csv", index=False, encoding="utf-8-sig")
    station_wins.to_csv(OUTPUT_DIR / "station_win_counts.csv", index=False, encoding="utf-8-sig")
    write_report(overall, comparisons, station_wins, features)
    validation = (
        overall[overall["split"].eq("val")]
        .groupby("mode", as_index=False)
        .agg(macro_rmse=("macro_station_rmse", "mean"), std=("macro_station_rmse", "std"))
        .sort_values("macro_rmse")
    )
    validation["representation"] = validation["mode"].map(experiment.TERMINAL_MODE_NAMES)
    console.table(
        "five-seed validation summary",
        validation,
        columns=("representation", "macro_rmse", "std"),
    )
    console.done(OUTPUT_DIR, report="formal_report.md", details="multiseed CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
