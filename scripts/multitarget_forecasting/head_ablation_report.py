#!/usr/bin/env python3
"""Report the parameter-controlled target-specific forecasting-head ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import head_ablation_config as config
from scripts.multitarget_forecasting.head_ablation_run import parse_variants
from scripts.multitarget_forecasting.report import (
    event_flags,
    regression_metrics,
    warning_metrics,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def _result_path(station: str, variant: str, seed: int) -> Path:
    if variant == config.REFERENCE_VARIANT:
        return base_config.prediction_path(
            station, config.CONTEXT, "delta", seed
        )
    return config.prediction_path(station, variant, seed)


def _variant_label(variant: str) -> str:
    return (
        config.REFERENCE_LABEL
        if variant == config.REFERENCE_VARIANT
        else config.VARIANT_LABELS[variant]
    )


def build_tables(
    stations: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    compared_variants = (config.REFERENCE_VARIANT, *variants)
    for station in stations:
        for variant in compared_variants:
            for seed in seeds:
                path = _result_path(station, variant, seed)
                if not path.exists():
                    raise FileNotFoundError(f"缺少专属头消融结果: {path}")
                arrays, metadata = io.load_archive(path)
                expected_experiment = (
                    base_config.EXPERIMENT_ID
                    if variant == config.REFERENCE_VARIANT
                    else config.EXPERIMENT_ID
                )
                if metadata.get("experiment") != expected_experiment:
                    raise RuntimeError(f"结果不属于预期实验: {path}")
                pred = np.asarray(arrays["pred"], dtype=float)
                true = np.asarray(arrays["true"], dtype=float)
                mask = np.asarray(arrays["mask"], dtype=bool)
                current = np.asarray(arrays["current"], dtype=float)
                expected_shape = (
                    len(current),
                    config.OUTPUT_STEPS,
                    len(config.TARGETS),
                )
                if pred.shape != expected_shape or true.shape != expected_shape:
                    raise RuntimeError(
                        f"联合预测必须是[样本,18,5]，实际为{pred.shape}: {path}"
                    )
                lower = np.asarray(arrays["warning_lower"], dtype=float)
                upper = np.asarray(arrays["warning_upper"], dtype=float)
                for target_index, target in enumerate(config.TARGETS):
                    for horizon_index, horizon_hours in enumerate(
                        config.HORIZON_HOURS
                    ):
                        valid = (
                            mask[:, horizon_index, target_index]
                            & np.isfinite(true[:, horizon_index, target_index])
                            & np.isfinite(pred[:, horizon_index, target_index])
                            & np.isfinite(current[:, target_index])
                        )
                        observed = true[valid, horizon_index, target_index]
                        predicted = pred[valid, horizon_index, target_index]
                        persistence = current[valid, target_index]
                        model_metric = regression_metrics(observed, predicted)
                        persistence_metric = regression_metrics(
                            observed, persistence
                        )
                        persistence_rmse = persistence_metric["rmse"]
                        relative = (
                            100.0
                            * (model_metric["rmse"] / persistence_rmse - 1.0)
                            if np.isfinite(persistence_rmse)
                            and persistence_rmse > 0
                            else np.nan
                        )
                        shared = {
                            "station": station,
                            "variant": variant,
                            "variant_label": _variant_label(variant),
                            "seed": seed,
                            "target": target,
                            "target_output_mode": (
                                "delta"
                                if variant == config.REFERENCE_VARIANT
                                else config.TARGET_OUTPUT_MODES[target]
                            ),
                            "horizon_hours": horizon_hours,
                        }
                        forecast_rows.append(
                            {
                                **shared,
                                "valid_rows": int(valid.sum()),
                                **model_metric,
                                "persistence_rmse": persistence_rmse,
                                "relative_rmse_pct": relative,
                                "beats_persistence": bool(
                                    model_metric["rmse"] < persistence_rmse
                                ),
                            }
                        )
                        actual_event = event_flags(
                            observed, lower[target_index], upper[target_index]
                        )
                        for model_name, event_prediction in (
                            (
                                "joint_gru",
                                event_flags(
                                    predicted,
                                    lower[target_index],
                                    upper[target_index],
                                ),
                            ),
                            (
                                "persistence",
                                event_flags(
                                    persistence,
                                    lower[target_index],
                                    upper[target_index],
                                ),
                            ),
                        ):
                            warning_rows.append(
                                {
                                    **shared,
                                    "model": model_name,
                                    "warning_lower": lower[target_index],
                                    "warning_upper": upper[target_index],
                                    **warning_metrics(
                                        actual_event, event_prediction
                                    ),
                                }
                            )
                runtime_rows.append(
                    {
                        "station": station,
                        "variant": variant,
                        "variant_label": _variant_label(variant),
                        "seed": seed,
                        "selected_epoch": int(arrays["selected_epoch"]),
                        "best_internal_val_loss": float(
                            arrays["best_internal_val_loss"]
                        ),
                        "training_seconds": float(arrays["training_seconds"]),
                        "inference_seconds": float(arrays["inference_seconds"]),
                        "parameter_count": int(arrays["parameter_count"]),
                    }
                )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(warning_rows),
        pd.DataFrame(runtime_rows),
    )


def _paired_comparisons(forecast: pd.DataFrame) -> pd.DataFrame:
    paired = forecast.pivot_table(
        index=["station", "seed", "target", "horizon_hours"],
        columns="variant",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    comparisons = (
        (
            "指标混合表示效果",
            config.REFERENCE_VARIANT,
            "mixed_linear",
        ),
        ("非线性容量效果", "mixed_linear", "mixed_shared_mlp"),
        (
            "专属头效果",
            "mixed_shared_mlp",
            "mixed_target_heads",
        ),
    )
    parts = []
    for comparison, baseline, candidate in comparisons:
        if baseline not in paired or candidate not in paired:
            continue
        part = paired[
            ["station", "seed", "target", "horizon_hours", baseline, candidate]
        ].dropna()
        part = part.rename(
            columns={baseline: "baseline_rmse", candidate: "candidate_rmse"}
        )
        part["comparison"] = comparison
        part["baseline"] = _variant_label(baseline)
        part["candidate"] = _variant_label(candidate)
        part["candidate_relative_rmse_pct"] = 100.0 * (
            part["candidate_rmse"] / part["baseline_rmse"] - 1.0
        )
        part["candidate_wins"] = (
            part["candidate_rmse"] < part["baseline_rmse"]
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无结果"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_report(
    forecast: pd.DataFrame,
    warnings: pd.DataFrame,
    runtime: pd.DataFrame,
) -> Path:
    output = config.OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(
        output / "专属头逐站点指标时距结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    warnings.to_csv(
        output / "专属头预警结果.csv", index=False, encoding="utf-8-sig"
    )
    runtime.to_csv(
        output / "专属头运行时间.csv", index=False, encoding="utf-8-sig"
    )
    paired = _paired_comparisons(forecast)
    paired.to_csv(
        output / "专属头配对比较.csv", index=False, encoding="utf-8-sig"
    )

    overall = (
        forecast.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均相对RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
            评价单元数=("rmse", "size"),
        )
        .sort_values("平均相对RMSE变化百分比")
    )
    by_target = (
        forecast.groupby(["variant_label", "target"], as_index=False)
        .agg(
            平均相对RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
        )
        .sort_values(["target", "平均相对RMSE变化百分比"])
    )
    station_variant = forecast.groupby(
        ["station", "variant", "variant_label"], as_index=False
    ).agg(平均相对RMSE变化百分比=("relative_rmse_pct", "mean"))
    station_winners = station_variant.loc[
        station_variant.groupby("station")[
            "平均相对RMSE变化百分比"
        ].idxmin()
    ]
    station_winner_counts = (
        station_winners.groupby("variant_label", as_index=False)
        .agg(最优站点数=("station", "size"))
        .sort_values("最优站点数", ascending=False)
    )
    paired_summary = pd.DataFrame()
    paired_by_target = pd.DataFrame()
    if not paired.empty:
        paired_summary = (
            paired.groupby(["comparison", "baseline", "candidate"], as_index=False)
            .agg(
                候选模型RMSE变化百分比=(
                    "candidate_relative_rmse_pct",
                    "mean",
                ),
                候选模型胜率=("candidate_wins", "mean"),
                配对单元数=("candidate_wins", "size"),
            )
            .sort_values("comparison")
        )
        paired_by_target = (
            paired.groupby(["comparison", "target"], as_index=False)
            .agg(
                候选模型RMSE变化百分比=(
                    "candidate_relative_rmse_pct",
                    "mean",
                ),
                候选模型胜率=("candidate_wins", "mean"),
            )
            .sort_values(["comparison", "target"])
        )
    warning_summary = (
        warnings.groupby(["variant_label", "model"], as_index=False)
        .agg(
            平均召回率=("recall", "mean"),
            平均F1=("f1", "mean"),
            平均误报率=("false_alarm_rate", "mean"),
        )
        .sort_values(["variant_label", "model"])
    )
    runtime_summary = (
        runtime.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均选定轮数=("selected_epoch", "mean"),
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
            参数量=("parameter_count", "mean"),
        )
        .sort_values("参数量")
    )

    report = "\n".join(
        (
            "# 五指标专属预测头消融报告",
            "",
            "- 全部模型使用相同的24小时输入、18个时距、5项指标、时间前向早停和标签掩码。",
            "- 混合表示固定为pH/溶解氧/氨氮预测变化量，高锰酸盐指数/总磷预测原值；所有站点共用，不按站点调整。",
            "- 共享非线性头与指标专属头的头部参数量近似匹配；两者配对比较是专属头的主要证据。",
            "- 混合表示映射由本轮2024验证结果预先固定，因此其相对统一变化量的改善仅作开发阶段证据，最终结论必须在未使用的测试集确认。",
            "",
            "## 总体结果",
            "",
            _markdown_table(overall),
            "",
            "## 受控配对比较",
            "",
            _markdown_table(paired_summary),
            "",
            "## 受控配对的分指标结果",
            "",
            _markdown_table(paired_by_target),
            "",
            "## 分指标结果",
            "",
            _markdown_table(by_target),
            "",
            "## 分站点最优模型计数",
            "",
            _markdown_table(station_winner_counts),
            "",
            "## 提前预警结果",
            "",
            _markdown_table(warning_summary),
            "",
            "## 训练、推理和参数量",
            "",
            _markdown_table(runtime_summary),
            "",
        )
    )
    path = output / "五指标专属预测头消融报告.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总指标专属预测头消融")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--variants", default=",".join(config.VARIANTS))
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    forecast, warnings, runtime = build_tables(
        stations, parse_variants(args.variants), _parse_seeds(args.seeds)
    )
    path = write_report(forecast, warnings, runtime)
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
