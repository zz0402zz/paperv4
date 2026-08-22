#!/usr/bin/env python3
"""Report forecast accuracy and training-quantile early-warning skill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config, data, io
from scripts.multitarget_forecasting.run import (
    _parse_contexts,
    _parse_seeds,
    _parse_target_modes,
    select_stations,
)


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if not len(true):
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan}
    error = pred - true
    denominator = float(np.square(true - np.mean(true)).sum())
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "nse": (
            float(1.0 - np.square(error).sum() / denominator)
            if denominator > 0
            else np.nan
        ),
    }


def event_flags(
    values: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    flags = np.zeros(values.shape, dtype=bool)
    if np.isfinite(lower):
        flags |= values < lower
    if np.isfinite(upper):
        flags |= values > upper
    return flags


def warning_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    false_alarm_rate = fp / (fp + tn) if fp + tn else np.nan
    return {
        "events": int(actual.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def build_tables(
    stations: tuple[str, ...],
    contexts: tuple[str, ...],
    target_modes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows = []
    warning_rows = []
    runtime_rows = []
    curve_rows = []
    for station in stations:
        for context in contexts:
            for target_mode in target_modes:
                for seed in seeds:
                    path = config.prediction_path(
                        station, context, target_mode, seed
                    )
                    if not path.exists():
                        raise FileNotFoundError(f"缺少联合预测结果: {path}")
                    arrays, metadata = io.load_archive(path)
                    if metadata.get("experiment") != config.EXPERIMENT_ID:
                        raise RuntimeError(f"结果不属于当前实验: {path}")
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
                                "context": context,
                                "context_label": config.CONTEXT_LABELS[context],
                                "target_mode": target_mode,
                                "target_mode_label": config.TARGET_MODE_LABELS[
                                    target_mode
                                ],
                                "seed": seed,
                                "target": target,
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
                            warning_predictions = (
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
                            )
                            for model_name, predicted_event in warning_predictions:
                                warning_rows.append(
                                    {
                                        **shared,
                                        "model": model_name,
                                        "warning_lower": lower[target_index],
                                        "warning_upper": upper[target_index],
                                        **warning_metrics(
                                            actual_event, predicted_event
                                        ),
                                    }
                                )
                    runtime_rows.append(
                        {
                            "station": station,
                            "context": context,
                            "context_label": config.CONTEXT_LABELS[context],
                            "target_mode": target_mode,
                            "target_mode_label": config.TARGET_MODE_LABELS[
                                target_mode
                            ],
                            "seed": seed,
                            "selected_epoch": int(arrays["selected_epoch"]),
                            "best_internal_val_loss": float(
                                arrays["best_internal_val_loss"]
                            ),
                            "selection_training_seconds": float(
                                arrays["selection_training_seconds"]
                            ),
                            "refit_training_seconds": float(
                                arrays["refit_training_seconds"]
                            ),
                            "training_seconds": float(arrays["training_seconds"]),
                            "inference_seconds": float(
                                arrays["inference_seconds"]
                            ),
                            "parameter_count": int(arrays["parameter_count"]),
                            "final_training_loss": float(
                                arrays["final_training_loss"]
                            ),
                        }
                    )
                    for curve_epoch, train_loss, val_loss in zip(
                        np.asarray(arrays["selection_epochs"], dtype=int),
                        np.asarray(arrays["selection_train_loss"], dtype=float),
                        np.asarray(arrays["selection_val_loss"], dtype=float),
                        strict=True,
                    ):
                        curve_rows.append(
                            {
                                "station": station,
                                "context": context,
                                "context_label": config.CONTEXT_LABELS[context],
                                "target_mode": target_mode,
                                "target_mode_label": config.TARGET_MODE_LABELS[
                                    target_mode
                                ],
                                "seed": seed,
                                "epoch": int(curve_epoch),
                                "train_loss": float(train_loss),
                                "internal_validation_loss": float(val_loss),
                                "selected": int(curve_epoch)
                                == int(arrays["selected_epoch"]),
                            }
                        )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(warning_rows),
        pd.DataFrame(runtime_rows),
        pd.DataFrame(curve_rows),
    )


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
    curves: pd.DataFrame,
) -> Path:
    output = config.VALIDATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output / "五指标逐站点时距结果.csv", index=False, encoding="utf-8-sig")
    warnings.to_csv(output / "训练分位数预警结果.csv", index=False, encoding="utf-8-sig")
    runtime.to_csv(output / "联合模型运行时间.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(output / "训练期内部验证曲线.csv", index=False, encoding="utf-8-sig")

    paired_modes = pd.DataFrame()
    if {"absolute", "delta"}.issubset(set(forecast["target_mode"].unique())):
        paired_modes = (
            forecast.pivot_table(
                index=[
                    "station",
                    "context",
                    "context_label",
                    "seed",
                    "target",
                    "horizon_hours",
                ],
                columns="target_mode",
                values="rmse",
                aggfunc="first",
            )
            .dropna(subset=["absolute", "delta"])
            .reset_index()
        )
        paired_modes["变化量相对原值RMSE变化百分比"] = 100.0 * (
            paired_modes["delta"] / paired_modes["absolute"] - 1.0
        )
        paired_modes["变化量胜出"] = paired_modes["delta"] < paired_modes["absolute"]
        paired_modes.to_csv(
            output / "原值变化量配对比较.csv", index=False, encoding="utf-8-sig"
        )

    overall = (
        forecast.groupby(
            ["context", "context_label", "target_mode", "target_mode_label"],
            as_index=False,
        )
        .agg(
            平均相对RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
            评价单元数=("rmse", "size"),
        )
        .sort_values("平均相对RMSE变化百分比")
    )
    by_target = (
        forecast.groupby(
            ["context_label", "target_mode_label", "target"], as_index=False
        )
        .agg(
            平均相对RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
        )
        .sort_values(["target", "平均相对RMSE变化百分比"])
    )
    warning_summary = (
        warnings.groupby(
            ["context_label", "target_mode_label", "model", "horizon_hours"],
            as_index=False,
        )
        .agg(
            平均召回率=("recall", "mean"),
            平均F1=("f1", "mean"),
            平均误报率=("false_alarm_rate", "mean"),
        )
        .sort_values(["context_label", "target_mode_label", "horizon_hours", "model"])
    )
    runtime_summary = (
        runtime.groupby(["context_label", "target_mode_label"], as_index=False)
        .agg(
            选择轮数=("selected_epoch", "mean"),
            最佳内部验证损失=("best_internal_val_loss", "mean"),
            选轮数秒数=("selection_training_seconds", "mean"),
            完整重训秒数=("refit_training_seconds", "mean"),
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
            参数量=("parameter_count", "mean"),
        )
        .sort_values("平均训练秒数")
    )
    mode_summary = pd.DataFrame()
    if not paired_modes.empty:
        mode_summary = (
            paired_modes.groupby(["context_label", "target"], as_index=False)
            .agg(
                变化量相对原值RMSE变化百分比=(
                    "变化量相对原值RMSE变化百分比",
                    "mean",
                ),
                变化量胜率=("变化量胜出", "mean"),
                配对单元数=("变化量胜出", "size"),
            )
            .sort_values(["target", "变化量相对原值RMSE变化百分比"])
        )

    report = "\n".join(
        (
            "# 五指标联合预测时间前向早停输入尺度消融报告",
            "",
            "- 每个模型一次输出5项指标在4至72小时的全部90个预测值。",
            "- 原值和变化量是两个联合模型，不拆分成单指标模型。",
            "- 轮数只由2023年下半年内部验证选择，再用完整2022至2023年重训。",
            "- 主比较指标是相对持续性RMSE，避免直接平均不同量纲的RMSE。",
            "- 预警阈值只由训练集分位数确定，用于方法验证，不是国家或断面考核标准。",
            "- 验证集标签未参与模型拟合，测试集未使用。",
            "",
            "## 输入尺度总体结果",
            "",
            _markdown_table(overall),
            "",
            "## 分指标结果",
            "",
            _markdown_table(by_target),
            "",
            "## 原值与变化量配对结果",
            "",
            _markdown_table(mode_summary),
            "",
            "## 提前预警结果",
            "",
            _markdown_table(warning_summary),
            "",
            "## 选定轮数、训练与一次性推理时间",
            "",
            _markdown_table(runtime_summary),
            "",
        )
    )
    path = output / "五指标联合时间前向早停报告.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总五指标联合预测输入尺度消融")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--contexts", default=",".join(config.CONTEXTS))
    parser.add_argument("--target-modes", default=",".join(config.TARGET_MODES))
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    forecast, warnings, runtime, curves = build_tables(
        stations,
        _parse_contexts(args.contexts),
        _parse_target_modes(args.target_modes),
        _parse_seeds(args.seeds),
    )
    path = write_report(forecast, warnings, runtime, curves)
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
