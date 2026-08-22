#!/usr/bin/env python3
"""Strictly paired A/B/C/D/E report for preprocessing components."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import head_ablation_config as head_config
from scripts.multitarget_forecasting import preprocessing_ablation_config as old_config
from scripts.multitarget_forecasting import preprocessing_component_config as config
from scripts.multitarget_forecasting.report import (
    event_flags,
    regression_metrics,
    warning_metrics,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def result_path(station: str, variant: str, seed: int):
    if variant == "original":
        return head_config.prediction_path(station, "mixed_linear", seed)
    if variant in config.TRAIN_VARIANTS:
        return config.prediction_path(station, variant, seed)
    if variant in {"robust_huber", "robust_huber_log"}:
        return old_config.prediction_path(station, variant, seed)
    raise ValueError(f"未知报告变体: {variant}")


def build_tables(
    stations: tuple[str, ...], seeds: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for variant in config.REPORT_VARIANTS:
            for seed in seeds:
                path = result_path(station, variant, seed)
                if not path.exists():
                    raise FileNotFoundError(f"缺少预处理组件结果: {path}")
                arrays, metadata = io.load_archive(path)
                prediction = np.asarray(arrays["pred"], dtype=float)
                truth = np.asarray(arrays["true"], dtype=float)
                mask = np.asarray(arrays["mask"], dtype=bool)
                current = np.asarray(arrays["current"], dtype=float)
                expected = (
                    len(current),
                    config.OUTPUT_STEPS,
                    len(config.TARGETS),
                )
                if prediction.shape != expected or truth.shape != expected:
                    raise RuntimeError(f"联合预测形状错误: {path}")
                lower = np.asarray(arrays["warning_lower"], dtype=float)
                upper = np.asarray(arrays["warning_upper"], dtype=float)
                for target_index, target in enumerate(config.TARGETS):
                    for horizon_index, horizon_hours in enumerate(config.HORIZON_HOURS):
                        valid = (
                            mask[:, horizon_index, target_index]
                            & np.isfinite(truth[:, horizon_index, target_index])
                            & np.isfinite(prediction[:, horizon_index, target_index])
                            & np.isfinite(current[:, target_index])
                        )
                        observed = truth[valid, horizon_index, target_index]
                        predicted = prediction[valid, horizon_index, target_index]
                        persistence = current[valid, target_index]
                        metric = regression_metrics(observed, predicted)
                        persistence_metric = regression_metrics(observed, persistence)
                        persistence_rmse = persistence_metric["rmse"]
                        relative = (
                            100.0 * (metric["rmse"] / persistence_rmse - 1.0)
                            if np.isfinite(persistence_rmse) and persistence_rmse > 0
                            else np.nan
                        )
                        shared = {
                            "station": station,
                            "seed": seed,
                            "variant": variant,
                            "variant_label": config.VARIANT_LABELS[variant],
                            "target": target,
                            "horizon_hours": horizon_hours,
                        }
                        forecast_rows.append(
                            {
                                **shared,
                                "valid_rows": int(valid.sum()),
                                **metric,
                                "persistence_rmse": persistence_rmse,
                                "relative_persistence_rmse_pct": relative,
                            }
                        )
                        actual_event = event_flags(
                            observed, lower[target_index], upper[target_index]
                        )
                        warning_rows.append(
                            {
                                **shared,
                                "model": "model",
                                **warning_metrics(
                                    actual_event,
                                    event_flags(
                                        predicted,
                                        lower[target_index],
                                        upper[target_index],
                                    ),
                                ),
                            }
                        )
                        if variant == "original":
                            warning_rows.append(
                                {
                                    **shared,
                                    "variant": "persistence",
                                    "variant_label": "外部参照_持续性",
                                    "model": "persistence",
                                    **warning_metrics(
                                        actual_event,
                                        event_flags(
                                            persistence,
                                            lower[target_index],
                                            upper[target_index],
                                        ),
                                    ),
                                }
                            )
                runtime_rows.append(
                    {
                        "station": station,
                        "seed": seed,
                        "variant": variant,
                        "variant_label": config.VARIANT_LABELS[variant],
                        "selected_epoch": int(arrays["selected_epoch"]),
                        "training_seconds": float(arrays["training_seconds"]),
                        "inference_seconds": float(arrays["inference_seconds"]),
                        "parameter_count": int(arrays["parameter_count"]),
                        "source_experiment": metadata.get("experiment", ""),
                    }
                )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(warning_rows),
        pd.DataFrame(runtime_rows),
    )


def build_contrasts(forecast: pd.DataFrame) -> pd.DataFrame:
    keys = ["station", "seed", "target", "horizon_hours"]
    pivot = forecast.pivot(index=keys, columns="variant", values="rmse").reset_index()
    parts: list[pd.DataFrame] = []
    for comparison, reference, candidate in config.CONTRASTS:
        if reference not in pivot or candidate not in pivot:
            continue
        part = pivot[keys].copy()
        part["比较"] = comparison
        part["参照变体"] = reference
        part["候选变体"] = candidate
        part["参照RMSE"] = pivot[reference]
        part["候选RMSE"] = pivot[candidate]
        part["候选相对参照RMSE变化百分比"] = 100.0 * (
            pivot[candidate] / pivot[reference] - 1.0
        )
        part["候选胜出"] = pivot[candidate] < pivot[reference]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _aggregate_warning(frame: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    group_fields = ["variant", "variant_label", *extra, "model"]
    result = frame.groupby(group_fields, as_index=False)[["tp", "fp", "fn", "tn"]].sum()
    result["precision"] = result["tp"] / (result["tp"] + result["fp"])
    result["recall"] = result["tp"] / (result["tp"] + result["fn"])
    result["f1"] = 2.0 * result["precision"] * result["recall"] / (
        result["precision"] + result["recall"]
    )
    result["false_alarm_rate"] = result["fp"] / (result["fp"] + result["tn"])
    return result


def _text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def build_report(
    forecast: pd.DataFrame,
    warning: pd.DataFrame,
    runtime: pd.DataFrame,
    contrasts: pd.DataFrame,
    seeds: tuple[int, ...],
) -> tuple[str, dict[str, pd.DataFrame]]:
    external = (
        forecast.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均相对持续性RMSE变化百分比=("relative_persistence_rmse_pct", "mean"),
            平均NSE=("nse", "mean"),
            评价单元数=("rmse", "size"),
        )
    )
    contrast_summary = (
        contrasts.groupby("比较", as_index=False)
        .agg(
            候选相对参照RMSE变化百分比=("候选相对参照RMSE变化百分比", "mean"),
            候选胜率=("候选胜出", "mean"),
            配对单元数=("候选RMSE", "size"),
        )
    )
    by_target = (
        contrasts.groupby(["比较", "target"], as_index=False)
        .agg(
            候选相对参照RMSE变化百分比=("候选相对参照RMSE变化百分比", "mean"),
            候选胜率=("候选胜出", "mean"),
        )
    )
    by_horizon = (
        contrasts.groupby(["比较", "horizon_hours"], as_index=False)
        .agg(
            候选相对参照RMSE变化百分比=("候选相对参照RMSE变化百分比", "mean"),
            候选胜率=("候选胜出", "mean"),
        )
    )
    warning_summary = _aggregate_warning(warning)
    warning_by_target = _aggregate_warning(warning, ("target",))
    runtime_summary = (
        runtime.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            平均选定轮数=("selected_epoch", "mean"),
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
            平均参数量=("parameter_count", "mean"),
        )
    )
    report = f"""# 五指标联合GRU预处理组件拆分消融报告

- 随机种子：{', '.join(map(str, seeds))}。
- A/B/C/D/E使用相同25站、24小时输入、共享线性头和固定混合输出。
- 主结论只使用候选相对参照的严格配对RMSE；持续性仅作外部参照。
- 轮数由2023年下半年内部时间验证选择，2024年只用于开发验证，不读取2025测试标签。
- 本轮为种子42组件筛选，不是最终显著性确认。

## 主要组件对比

```text
{_text(contrast_summary)}
```

## 分指标组件对比

```text
{_text(by_target)}
```

## 分时距组件对比

```text
{_text(by_horizon)}
```

## 外部持续性参照

```text
{_text(external)}
```

## 提前预警

```text
{_text(warning_summary)}
```

## 分指标提前预警

```text
{_text(warning_by_target)}
```

## 训练与推理开销

```text
{_text(runtime_summary)}
```
"""
    return report, {
        "主要组件严格配对.csv": contrast_summary,
        "分指标组件严格配对.csv": by_target,
        "分时距组件严格配对.csv": by_horizon,
        "组件严格配对明细.csv": contrasts,
        "外部持续性参照.csv": external,
        "提前预警结果.csv": warning_summary,
        "分指标提前预警结果.csv": warning_by_target,
        "训练与推理开销.csv": runtime_summary,
        "逐站点指标时距结果.csv": forecast,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成预处理A/B/C/D/E组件拆分报告")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    seeds = _parse_seeds(args.seeds)
    forecast, warning, runtime = build_tables(stations, seeds)
    contrasts = build_contrasts(forecast)
    report, tables = build_report(forecast, warning, runtime, contrasts, seeds)
    output_dir = config.OUTPUT_DIR / "报告"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    path = output_dir / "预处理组件拆分消融报告.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
