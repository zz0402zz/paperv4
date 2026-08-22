#!/usr/bin/env python3
"""Compare same-sample XGBoost with the leading joint-GRU variants."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config, data, io
from scripts.multitarget_forecasting import head_ablation_config as head_config
from scripts.multitarget_forecasting.report import (
    event_flags,
    regression_metrics,
    warning_metrics,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations
from scripts.multitarget_forecasting import xgboost_baseline as xgb


VARIANT_LABELS = {
    "xgboost_absolute": "XGBoost_统一原值",
    "xgboost_delta": "XGBoost_统一变化量",
    "xgboost_mixed": "XGBoost_指标混合表示",
    "gru_mixed_linear": "GRU_共享线性头_指标混合表示",
    "gru_target_heads": "GRU_指标专属头_指标混合表示",
    "persistence": "持续性",
}
MODEL_VARIANTS = tuple(VARIANT_LABELS)


def mixed_prediction(
    absolute: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    absolute = np.asarray(absolute, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if absolute.shape != delta.shape:
        raise ValueError("XGBoost原值与变化量预测形状不一致。")
    result = np.empty_like(absolute)
    for target_index, target in enumerate(config.TARGETS):
        source = (
            delta
            if head_config.TARGET_OUTPUT_MODES[target] == "delta"
            else absolute
        )
        result[:, :, target_index] = source[:, :, target_index]
    return result


def _load_station_predictions(
    station: str, seed: int, device: str
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    xgb_arrays: dict[str, dict[str, np.ndarray]] = {}
    for mode in ("absolute", "delta"):
        path = xgb.prediction_path(station, mode, seed, device)
        if not path.exists():
            raise FileNotFoundError(f"缺少同协议XGBoost结果: {path}")
        arrays, metadata = io.load_archive(path)
        if (
            metadata.get("experiment") != xgb.EXPERIMENT_ID
            or metadata.get("target_mode") != mode
            or metadata.get("device_type") != device
        ):
            raise RuntimeError(f"XGBoost结果身份不一致: {path}")
        xgb_arrays[mode] = arrays

    head_arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant, output_key in (
        ("mixed_linear", "gru_mixed_linear"),
        ("mixed_target_heads", "gru_target_heads"),
    ):
        path = head_config.prediction_path(station, variant, seed)
        if not path.exists():
            raise FileNotFoundError(f"缺少GRU对照结果: {path}")
        arrays, metadata = io.load_archive(path)
        if metadata.get("experiment") != head_config.EXPERIMENT_ID:
            raise RuntimeError(f"GRU结果身份不一致: {path}")
        head_arrays[output_key] = arrays

    reference = xgb_arrays["absolute"]
    reference_times = np.asarray(reference["target_start"])
    for name, arrays in {**xgb_arrays, **head_arrays}.items():
        if not np.array_equal(np.asarray(arrays["target_start"]), reference_times):
            raise RuntimeError(f"预测时间轴不一致: {station}/{name}")
        for key in ("true", "mask", "current"):
            if np.asarray(arrays[key]).shape != np.asarray(reference[key]).shape:
                raise RuntimeError(f"评价数组形状不一致: {station}/{name}/{key}")

    predictions = {
        "xgboost_absolute": np.asarray(xgb_arrays["absolute"]["pred"], dtype=float),
        "xgboost_delta": np.asarray(xgb_arrays["delta"]["pred"], dtype=float),
        "xgboost_mixed": mixed_prediction(
            xgb_arrays["absolute"]["pred"], xgb_arrays["delta"]["pred"]
        ),
        "gru_mixed_linear": np.asarray(
            head_arrays["gru_mixed_linear"]["pred"], dtype=float
        ),
        "gru_target_heads": np.asarray(
            head_arrays["gru_target_heads"]["pred"], dtype=float
        ),
    }
    return predictions, {**xgb_arrays, **head_arrays}


def build_tables(
    stations: tuple[str, ...], seeds: tuple[int, ...], device: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for seed in seeds:
            predictions, arrays_by_source = _load_station_predictions(
                station, seed, device
            )
            reference = arrays_by_source["absolute"]
            true = np.asarray(reference["true"], dtype=float)
            mask = np.asarray(reference["mask"], dtype=bool)
            current = np.asarray(reference["current"], dtype=float)
            lower = np.asarray(reference["warning_lower"], dtype=float)
            upper = np.asarray(reference["warning_upper"], dtype=float)
            predictions["persistence"] = np.repeat(
                current[:, None, :], config.OUTPUT_STEPS, axis=1
            )
            for variant, prediction in predictions.items():
                if prediction.shape != true.shape:
                    raise RuntimeError(
                        f"预测必须是[样本,18,5]: {station}/{variant}/"
                        f"{prediction.shape}"
                    )
                for target_index, target in enumerate(config.TARGETS):
                    for horizon_index, horizon_hours in enumerate(
                        config.HORIZON_HOURS
                    ):
                        valid = (
                            mask[:, horizon_index, target_index]
                            & np.isfinite(true[:, horizon_index, target_index])
                            & np.isfinite(prediction[:, horizon_index, target_index])
                            & np.isfinite(current[:, target_index])
                        )
                        observed = true[valid, horizon_index, target_index]
                        predicted = prediction[valid, horizon_index, target_index]
                        persistent = current[valid, target_index]
                        metric = regression_metrics(observed, predicted)
                        persistence_metric = regression_metrics(
                            observed, persistent
                        )
                        persistence_rmse = persistence_metric["rmse"]
                        relative = (
                            100.0 * (metric["rmse"] / persistence_rmse - 1.0)
                            if np.isfinite(persistence_rmse)
                            and persistence_rmse > 0
                            else np.nan
                        )
                        shared = {
                            "station": station,
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
                            "seed": seed,
                            "target": target,
                            "horizon_hours": horizon_hours,
                        }
                        forecast_rows.append(
                            {
                                **shared,
                                "valid_rows": int(valid.sum()),
                                **metric,
                                "persistence_rmse": persistence_rmse,
                                "relative_rmse_pct": relative,
                                "beats_persistence": bool(
                                    metric["rmse"] < persistence_rmse
                                ),
                            }
                        )
                        warning_rows.append(
                            {
                                **shared,
                                **warning_metrics(
                                    event_flags(
                                        observed,
                                        lower[target_index],
                                        upper[target_index],
                                    ),
                                    event_flags(
                                        predicted,
                                        lower[target_index],
                                        upper[target_index],
                                    ),
                                ),
                            }
                        )

            for mode, variant in (
                ("absolute", "xgboost_absolute"),
                ("delta", "xgboost_delta"),
            ):
                source = arrays_by_source[mode]
                runtime_rows.append(
                    {
                        "station": station,
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "seed": seed,
                        "device": device,
                        "training_seconds": float(source["training_seconds"]),
                        "inference_seconds": float(source["inference_seconds"]),
                        "tree_count": int(source["tree_count"]),
                        "parameter_count": 0,
                        "fitted_output_count": int(source["fitted_output_count"]),
                    }
                )
            absolute = arrays_by_source["absolute"]
            delta = arrays_by_source["delta"]
            chosen = np.asarray(
                [
                    head_config.TARGET_OUTPUT_MODES[target] == "delta"
                    for target in config.TARGETS
                ],
                dtype=bool,
            )
            mixed_training = np.where(
                chosen,
                np.asarray(delta["training_seconds_by_target"], dtype=float),
                np.asarray(absolute["training_seconds_by_target"], dtype=float),
            )
            mixed_inference = np.where(
                chosen,
                np.asarray(delta["inference_seconds_by_target"], dtype=float),
                np.asarray(absolute["inference_seconds_by_target"], dtype=float),
            )
            mixed_trees = np.where(
                chosen,
                np.asarray(delta["tree_count_by_target"], dtype=np.int64),
                np.asarray(absolute["tree_count_by_target"], dtype=np.int64),
            )
            runtime_rows.append(
                {
                    "station": station,
                    "variant": "xgboost_mixed",
                    "variant_label": VARIANT_LABELS["xgboost_mixed"],
                    "seed": seed,
                    "device": device,
                    "training_seconds": float(mixed_training.sum()),
                    "inference_seconds": float(mixed_inference.sum()),
                    "tree_count": int(mixed_trees.sum()),
                    "parameter_count": 0,
                    "fitted_output_count": config.OUTPUT_STEPS
                    * len(config.TARGETS),
                }
            )
            for source_key, variant in (
                ("gru_mixed_linear", "gru_mixed_linear"),
                ("gru_target_heads", "gru_target_heads"),
            ):
                source = arrays_by_source[source_key]
                runtime_rows.append(
                    {
                        "station": station,
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "seed": seed,
                        "device": device,
                        "training_seconds": float(source["training_seconds"]),
                        "inference_seconds": float(source["inference_seconds"]),
                        "tree_count": 0,
                        "parameter_count": int(source["parameter_count"]),
                        "fitted_output_count": 1,
                    }
                )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(warning_rows),
        pd.DataFrame(runtime_rows),
    )


def paired_comparisons(forecast: pd.DataFrame) -> pd.DataFrame:
    paired = forecast.pivot_table(
        index=["station", "seed", "target", "horizon_hours"],
        columns="variant",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    comparisons = (
        (
            "XGBoost混合表示对GRU共享线性头",
            "gru_mixed_linear",
            "xgboost_mixed",
        ),
        (
            "XGBoost混合表示对GRU专属头",
            "gru_target_heads",
            "xgboost_mixed",
        ),
        (
            "XGBoost混合表示对统一变化量",
            "xgboost_delta",
            "xgboost_mixed",
        ),
        (
            "XGBoost变化量对原值",
            "xgboost_absolute",
            "xgboost_delta",
        ),
    )
    parts = []
    for comparison, baseline, candidate in comparisons:
        part = paired[
            ["station", "seed", "target", "horizon_hours", baseline, candidate]
        ].dropna().copy()
        part = part.rename(
            columns={baseline: "baseline_rmse", candidate: "candidate_rmse"}
        )
        part["comparison"] = comparison
        part["baseline"] = VARIANT_LABELS[baseline]
        part["candidate"] = VARIANT_LABELS[candidate]
        part["candidate_relative_rmse_pct"] = 100.0 * (
            part["candidate_rmse"] / part["baseline_rmse"] - 1.0
        )
        part["candidate_wins"] = (
            part["candidate_rmse"] < part["baseline_rmse"]
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


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
    device: str,
) -> str:
    output = xgb.device_output_dir(device)
    output.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(
        output / "XGBoost与GRU逐站点指标时距结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    warnings.to_csv(
        output / "XGBoost与GRU预警结果.csv", index=False, encoding="utf-8-sig"
    )
    runtime.to_csv(
        output / "XGBoost与GRU运行时间.csv", index=False, encoding="utf-8-sig"
    )
    paired = paired_comparisons(forecast)
    paired.to_csv(
        output / "XGBoost与GRU配对比较.csv", index=False, encoding="utf-8-sig"
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
    station_variant = forecast.groupby(
        ["station", "variant_label"], as_index=False
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
    warning_summary = (
        warnings.groupby("variant_label", as_index=False)
        .agg(
            平均召回率=("recall", "mean"),
            平均F1=("f1", "mean"),
            平均误报率=("false_alarm_rate", "mean"),
        )
        .sort_values("平均F1", ascending=False)
    )
    runtime_summary = (
        runtime.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
            平均树数=("tree_count", "mean"),
            神经网络参数量=("parameter_count", "mean"),
            独立拟合模型数=("fitted_output_count", "mean"),
        )
        .sort_values("平均训练秒数")
    )

    report = "\n".join(
        (
            "# 25站五指标同协议XGBoost与GRU比较报告",
            "",
            f"- XGBoost训练设备：{device}；设备在实验开始前固定，不根据验证集挑选。",
            "- 所有模型使用完全相同的站点、时间切分、24小时输入信息、验证样本、五指标标签掩码和18个时距。",
            "- XGBoost原值和变化量版分别训练；混合版固定选择pH/溶解氧/氨氮变化量与高锰酸盐指数/总磷原值。",
            "- XGBoost每个站点实际拟合90个标量回归器，是强准确率基线，不等价于GRU的一个模型一次输出18×5。",
            "- 混合表示映射由2024验证阶段的GRU结果固定，因此混合表示本身的改善仍属开发证据；测试集未使用。",
            "- 本轮先比较监督XGBoost，不同时扩展OOF门控和TabPFN蒸馏；只有跨站结果确认XGBoost优势后才进入昂贵阶段。",
            "",
            "## 总体结果",
            "",
            _markdown_table(overall),
            "",
            "## 逐单元配对比较",
            "",
            _markdown_table(paired_summary),
            "",
            "## 配对比较的分指标结果",
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
            "## 训练、推理和模型数",
            "",
            _markdown_table(runtime_summary),
            "",
        )
    )
    path = output / "25站五指标XGBoost与GRU同协议报告.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总同协议XGBoost与GRU结果")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--device", choices=xgb.DEVICES, default="cuda")
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    forecast, warnings, runtime = build_tables(
        stations, _parse_seeds(args.seeds), args.device
    )
    path = write_report(forecast, warnings, runtime, args.device)
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
