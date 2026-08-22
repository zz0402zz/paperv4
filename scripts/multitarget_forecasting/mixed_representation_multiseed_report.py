#!/usr/bin/env python3
"""Report the five-seed confirmation of the mixed target representation."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data
from scripts.multitarget_forecasting import head_ablation_config as head_config
from scripts.multitarget_forecasting.head_ablation_report import build_tables
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


BASELINE = head_config.REFERENCE_VARIANT
CANDIDATE = "mixed_linear"
OUTPUT_DIR = (
    base_config.OUTPUT_DIR / "验证集" / "混合输出表示五种子确认"
)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无结果"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def paired_cells(forecast: pd.DataFrame) -> pd.DataFrame:
    paired = forecast.pivot_table(
        index=["station", "seed", "target", "horizon_hours"],
        columns="variant",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    missing = {BASELINE, CANDIDATE}.difference(paired.columns)
    if missing:
        raise RuntimeError(f"缺少配对模型结果: {sorted(missing)}")
    paired = paired.rename(
        columns={
            BASELINE: "统一变化量RMSE",
            CANDIDATE: "指标混合表示RMSE",
        }
    )
    paired["混合表示相对变化百分比"] = 100.0 * (
        paired["指标混合表示RMSE"] / paired["统一变化量RMSE"] - 1.0
    )
    paired["混合表示获胜"] = (
        paired["指标混合表示RMSE"] < paired["统一变化量RMSE"]
    )
    return paired


def _paired_summary(
    paired: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    return (
        paired.groupby(group_columns, as_index=False)
        .agg(
            混合表示RMSE变化百分比=("混合表示相对变化百分比", "mean"),
            混合表示胜率=("混合表示获胜", "mean"),
            配对单元数=("混合表示获胜", "size"),
        )
        .sort_values(group_columns)
    )


def write_report(
    stations: tuple[str, ...],
    seeds: tuple[int, ...],
    forecast: pd.DataFrame,
    warnings: pd.DataFrame,
    runtime: pd.DataFrame,
) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paired = paired_cells(forecast)

    forecast_path = OUTPUT_DIR / "两种表示逐站点指标时距结果.csv"
    paired_path = OUTPUT_DIR / "混合表示与统一变化量配对结果.csv"
    forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")

    overall = (
        forecast.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均相对持续性RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
            评价单元数=("rmse", "size"),
        )
        .sort_values("平均相对持续性RMSE变化百分比")
    )
    paired_overall = _paired_summary(paired.assign(比较="混合表示对统一变化量"), ["比较"])
    by_seed = _paired_summary(paired, ["seed"])
    by_target = _paired_summary(paired, ["target"])
    by_horizon = _paired_summary(paired, ["horizon_hours"])
    by_station = _paired_summary(paired, ["station"])

    joint_warning = warnings.loc[warnings["model"] == "joint_gru"].copy()
    warning_summary = (
        joint_warning.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均召回率=("recall", "mean"),
            平均F1=("f1", "mean"),
            平均误报率=("false_alarm_rate", "mean"),
        )
        .sort_values("平均F1", ascending=False)
    )
    persistence_warning = warnings.loc[
        (warnings["variant"] == BASELINE)
        & (warnings["model"] == "persistence")
    ]
    if not persistence_warning.empty:
        persistence_summary = pd.DataFrame(
            [
                {
                    "variant": "persistence",
                    "variant_label": "持续性",
                    "平均召回率": persistence_warning["recall"].mean(),
                    "平均F1": persistence_warning["f1"].mean(),
                    "平均误报率": persistence_warning[
                        "false_alarm_rate"
                    ].mean(),
                }
            ]
        )
        warning_summary = pd.concat(
            [warning_summary, persistence_summary], ignore_index=True
        ).sort_values("平均F1", ascending=False)

    runtime_summary = (
        runtime.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均选定轮数=("selected_epoch", "mean"),
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
            参数量=("parameter_count", "mean"),
        )
        .sort_values("平均训练秒数")
    )

    by_seed.to_csv(
        OUTPUT_DIR / "分随机种子配对结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_target.to_csv(
        OUTPUT_DIR / "分指标配对结果.csv", index=False, encoding="utf-8-sig"
    )
    by_horizon.to_csv(
        OUTPUT_DIR / "分时距配对结果.csv", index=False, encoding="utf-8-sig"
    )
    by_station.to_csv(
        OUTPUT_DIR / "分站点配对结果.csv", index=False, encoding="utf-8-sig"
    )
    warning_summary.to_csv(
        OUTPUT_DIR / "提前预警结果.csv", index=False, encoding="utf-8-sig"
    )
    runtime_summary.to_csv(
        OUTPUT_DIR / "运行时间.csv", index=False, encoding="utf-8-sig"
    )

    seed_effects = by_seed["混合表示RMSE变化百分比"].to_numpy(dtype=float)
    favorable_seeds = int(np.sum(seed_effects < 0.0))
    stable = favorable_seeds == len(seeds)
    conclusion = (
        f"混合表示在{favorable_seeds}/{len(seeds)}个随机种子上降低配对RMSE。"
        + (
            "方向对训练随机性稳定，但仍需未使用的测试集确认。"
            if stable
            else "方向尚未对训练随机性稳定，不能据此冻结为正式模型。"
        )
    )

    report = "\n".join(
        (
            f"# {len(stations)}站五指标混合输出表示五种子确认报告",
            "",
            f"- 随机种子：{', '.join(map(str, seeds))}。",
            "- 两个模型均使用24小时输入、共享线性预测头、一次输出18时距乘5指标；唯一差异是输出表示。",
            "- 基线对五指标统一预测变化量；候选对pH、溶解氧和氨氮预测变化量，对高锰酸盐指数和总磷预测原值。",
            "- 本轮仍只使用2024验证集。多种子只能检验训练随机性，不能消除在同一验证集选择混合映射造成的开发偏差。",
            f"- 自动判定：{conclusion}",
            "",
            "## 总体结果",
            "",
            _markdown_table(overall),
            "",
            "## 严格配对结果",
            "",
            _markdown_table(paired_overall),
            "",
            "## 分随机种子配对结果",
            "",
            _markdown_table(by_seed),
            "",
            "## 分指标配对结果",
            "",
            _markdown_table(by_target),
            "",
            "## 分时距配对结果",
            "",
            _markdown_table(by_horizon),
            "",
            "## 分站点配对结果",
            "",
            _markdown_table(by_station),
            "",
            "## 提前预警结果",
            "",
            _markdown_table(warning_summary),
            "",
            "## 训练与推理开销",
            "",
            _markdown_table(runtime_summary),
            "",
        )
    )
    report_path = OUTPUT_DIR / f"{len(stations)}站五指标混合表示五种子确认报告.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总共享线性GRU混合输出表示的五种子确认实验"
    )
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="逗号分隔的站点")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument(
        "--seeds", default=",".join(map(str, base_config.FORMAL_SEEDS))
    )
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    seeds = _parse_seeds(args.seeds)
    forecast, warnings, runtime = build_tables(
        stations, (CANDIDATE,), seeds
    )
    path = write_report(stations, seeds, forecast, warnings, runtime)
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
