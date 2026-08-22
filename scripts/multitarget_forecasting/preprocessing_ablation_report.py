#!/usr/bin/env python3
"""Report preprocessing ablations and station-exclusion sensitivity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import head_ablation_config
from scripts.multitarget_forecasting import preprocessing_ablation_config as config
from scripts.multitarget_forecasting.preprocessing_ablation_data import enrich_station_dataset
from scripts.multitarget_forecasting.preprocessing_ablation_run import parse_variants
from scripts.multitarget_forecasting.report import event_flags, regression_metrics, warning_metrics
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def result_path(station: str, variant: str, seed: int) -> Path:
    if variant == config.REFERENCE_VARIANT:
        return head_ablation_config.prediction_path(station, "mixed_linear", seed)
    return config.prediction_path(station, variant, seed)


def _quality_masks(panel, stations: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for station in stations:
        dataset = enrich_station_dataset(panel, station)
        result[station] = np.asarray(
            data.split_by_time(dataset)["val"]["quality_y_mask"], dtype=bool
        )
    return result


def build_tables(
    panel,
    stations: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_masks = _quality_masks(panel, stations)
    forecast_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for variant in (config.REFERENCE_VARIANT, *variants):
            for seed in seeds:
                path = result_path(station, variant, seed)
                if not path.exists():
                    raise FileNotFoundError(f"缺少预处理消融结果: {path}")
                arrays, metadata = io.load_archive(path)
                prediction = np.asarray(arrays["pred"], dtype=float)
                truth = np.asarray(arrays["true"], dtype=float)
                base_mask = np.asarray(arrays["mask"], dtype=bool)
                current = np.asarray(arrays["current"], dtype=float)
                quality_mask = quality_masks[station]
                if quality_mask.shape != base_mask.shape:
                    raise RuntimeError(f"质量掩码形状不匹配: {station}")
                lower = np.asarray(arrays["warning_lower"], dtype=float)
                upper = np.asarray(arrays["warning_upper"], dtype=float)
                for scope, scope_mask in (
                    ("全量正式标签", base_mask),
                    ("排除软存疑标签", base_mask & quality_mask),
                ):
                    for target_index, target in enumerate(config.TARGETS):
                        for horizon_index, horizon_hours in enumerate(config.HORIZON_HOURS):
                            valid = (
                                scope_mask[:, horizon_index, target_index]
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
                                "variant": variant,
                                "variant_label": config.VARIANT_LABELS[variant],
                                "seed": seed,
                                "evaluation_scope": scope,
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
                                    "beats_persistence": bool(metric["rmse"] < persistence_rmse),
                                }
                            )
                            actual_event = event_flags(
                                observed, lower[target_index], upper[target_index]
                            )
                            for model_name, values in (
                                ("model", predicted),
                                ("persistence", persistence),
                            ):
                                warning_rows.append(
                                    {
                                        **shared,
                                        "model": model_name,
                                        **warning_metrics(
                                            actual_event,
                                            event_flags(values, lower[target_index], upper[target_index]),
                                        ),
                                    }
                                )
                runtime_rows.append(
                    {
                        "station": station,
                        "variant": variant,
                        "variant_label": config.VARIANT_LABELS[variant],
                        "seed": seed,
                        "selected_epoch": int(arrays["selected_epoch"]),
                        "training_seconds": float(arrays["training_seconds"]),
                        "inference_seconds": float(arrays["inference_seconds"]),
                        "parameter_count": int(arrays["parameter_count"]),
                        "experiment": metadata.get("experiment", ""),
                    }
                )
    return pd.DataFrame(forecast_rows), pd.DataFrame(warning_rows), pd.DataFrame(runtime_rows)


def paired_table(forecast: pd.DataFrame) -> pd.DataFrame:
    keys = ["station", "seed", "evaluation_scope", "target", "horizon_hours"]
    pivot = forecast.pivot(index=keys, columns="variant", values="rmse").reset_index()
    baseline = pivot[config.REFERENCE_VARIANT]
    parts = []
    for variant in config.VARIANTS:
        if variant not in pivot:
            continue
        part = pivot[keys].copy()
        part["variant"] = variant
        part["variant_label"] = config.VARIANT_LABELS[variant]
        part["candidate_rmse"] = pivot[variant]
        part["baseline_rmse"] = baseline
        part["candidate_relative_rmse_pct"] = 100.0 * (pivot[variant] / baseline - 1.0)
        part["candidate_log_baseline_rmse_ratio"] = np.log(
            pivot[variant] / baseline
        )
        part["candidate_wins"] = pivot[variant] < baseline
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _paired_summary(
    frame: pd.DataFrame, group_fields: list[str]
) -> pd.DataFrame:
    summary = (
        frame.groupby(group_fields, as_index=False)
        .agg(
            相对当前基线RMSE算术变化百分比=(
                "candidate_relative_rmse_pct",
                "mean",
            ),
            相对当前基线RMSE平均对数比=(
                "candidate_log_baseline_rmse_ratio",
                "mean",
            ),
            候选胜率=("candidate_wins", "mean"),
            配对单元数=("candidate_rmse", "size"),
        )
    )
    summary["相对当前基线RMSE几何变化百分比"] = 100.0 * np.expm1(
        summary.pop("相对当前基线RMSE平均对数比")
    )
    return summary


def _aggregate_warning(
    frame: pd.DataFrame, extra_group_fields: tuple[str, ...] = ()
) -> pd.DataFrame:
    group_fields = [
        "evaluation_scope",
        "variant",
        "variant_label",
        *extra_group_fields,
        "model",
    ]
    group = frame.groupby(
        group_fields, as_index=False
    )[["tp", "fp", "fn", "tn"]].sum()
    group["precision"] = group["tp"] / (group["tp"] + group["fp"])
    group["recall"] = group["tp"] / (group["tp"] + group["fn"])
    group["f1"] = 2 * group["precision"] * group["recall"] / (
        group["precision"] + group["recall"]
    )
    group["false_alarm_rate"] = group["fp"] / (group["fp"] + group["tn"])
    return group


def _text(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return selected.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def build_report(
    forecast: pd.DataFrame,
    warning: pd.DataFrame,
    runtime: pd.DataFrame,
    paired: pd.DataFrame,
    seeds: tuple[int, ...],
) -> tuple[str, dict[str, pd.DataFrame]]:
    full = forecast.loc[forecast["evaluation_scope"].eq("全量正式标签")]
    overall = (
        forecast.groupby(["evaluation_scope", "variant", "variant_label"], as_index=False)
        .agg(
            平均相对持续性RMSE变化百分比=("relative_rmse_pct", "mean"),
            相对持续性胜率=("beats_persistence", "mean"),
            平均NSE=("nse", "mean"),
            评价单元数=("rmse", "size"),
        )
    )
    paired_summary = _paired_summary(
        paired,
        ["evaluation_scope", "variant", "variant_label"],
    )
    by_seed = _paired_summary(
        paired,
        ["evaluation_scope", "variant", "variant_label", "seed"],
    )
    by_target = _paired_summary(
        paired,
        ["evaluation_scope", "variant", "variant_label", "target"],
    )
    by_horizon = _paired_summary(
        paired,
        ["evaluation_scope", "variant", "variant_label", "horizon_hours"],
    )
    cohort_parts = []
    for cohort, spec in config.SENSITIVITY_COHORTS.items():
        excluded = set(spec["excluded_stations"])
        part = full.loc[~full["station"].isin(excluded)]
        summary = (
            part.groupby(["variant", "variant_label"], as_index=False)
            .agg(
                平均相对持续性RMSE变化百分比=("relative_rmse_pct", "mean"),
                平均NSE=("nse", "mean"),
                评价单元数=("rmse", "size"),
            )
        )
        summary.insert(0, "cohort", cohort)
        summary.insert(1, "cohort_label", spec["label"])
        cohort_parts.append(summary)
    cohort_summary = pd.concat(cohort_parts, ignore_index=True)
    paired_cohort_parts = []
    full_paired = paired.loc[paired["evaluation_scope"].eq("全量正式标签")]
    for cohort, spec in config.SENSITIVITY_COHORTS.items():
        excluded = set(spec["excluded_stations"])
        part = full_paired.loc[~full_paired["station"].isin(excluded)]
        summary = _paired_summary(part, ["variant", "variant_label"])
        summary.insert(0, "cohort", cohort)
        summary.insert(1, "cohort_label", spec["label"])
        paired_cohort_parts.append(summary)
    paired_cohort_summary = pd.concat(paired_cohort_parts, ignore_index=True)
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
    full_pair = paired_summary.loc[
        paired_summary["evaluation_scope"].eq("全量正式标签")
    ].sort_values("相对当前基线RMSE几何变化百分比")
    best = full_pair.iloc[0] if len(full_pair) else None
    if best is None:
        decision = "尚无完整候选结果。"
    else:
        best_seed_rows = by_seed.loc[
            by_seed["evaluation_scope"].eq("全量正式标签")
            & by_seed["variant"].eq(best["variant"])
        ]
        favorable_seeds = int(
            (
                best_seed_rows["相对当前基线RMSE几何变化百分比"]
                < 0
            ).sum()
        )
        decision = (
            f"当前最优候选为{best['variant_label']}，相对基线RMSE几何变化"
            f"{best['相对当前基线RMSE几何变化百分比']:.3f}%；"
            f"在{favorable_seeds}/{len(best_seed_rows)}个随机种子上方向一致。"
        )
    report = f"""# 五指标联合GRU数据预处理消融报告

- 随机种子：{', '.join(map(str, seeds))}。
- 全部模型使用24小时输入、共享线性头、指标混合输出，一次预测18时距×5指标。
- 主报告保留浦阳江出口、闸口和浮石渡；删站只作敏感性统计，不改数据。
- 仅使用2024验证集，不读取2025测试标签。
- 自动判定：{decision}

## 总体结果

```text
{_text(overall)}
```

## 相对当前基线的严格配对结果

```text
{_text(paired_summary)}
```

## 分随机种子配对结果

```text
{_text(by_seed)}
```

## 分指标配对结果

```text
{_text(by_target)}
```

## 分时距配对结果

```text
{_text(by_horizon)}
```

## 删站敏感性

```text
{_text(cohort_summary)}
```

## 候选相对当前基线的删站敏感性

```text
{_text(paired_cohort_summary)}
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
    tables = {
        "总体结果.csv": overall,
        "相对当前基线配对结果.csv": paired_summary,
        "分随机种子配对结果.csv": by_seed,
        "分指标配对结果.csv": by_target,
        "分时距配对结果.csv": by_horizon,
        "删站敏感性.csv": cohort_summary,
        "候选相对基线删站敏感性.csv": paired_cohort_summary,
        "提前预警结果.csv": warning_summary,
        "分指标提前预警结果.csv": warning_by_target,
        "运行时间.csv": runtime_summary,
        "逐站点指标时距结果.csv": forecast,
        "严格配对明细.csv": paired,
    }
    return report, tables


def main() -> None:
    parser = argparse.ArgumentParser(description="生成数据预处理消融和删站敏感性报告")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--variants", default=",".join(config.VARIANTS))
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument(
        "--report-folder",
        default="报告",
        help="保存在数据预处理消融目录下的单层中文报告文件夹名",
    )
    args = parser.parse_args()
    report_folder = Path(args.report_folder)
    if report_folder.name != args.report_folder or args.report_folder in {"", ".", ".."}:
        parser.error("--report-folder 必须是不含路径分隔符的单层文件夹名")
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    variants = parse_variants(args.variants)
    seeds = _parse_seeds(args.seeds)
    forecast, warning, runtime = build_tables(panel, stations, variants, seeds)
    paired = paired_table(forecast)
    report, tables = build_report(forecast, warning, runtime, paired, seeds)
    output_dir = config.OUTPUT_DIR / report_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    path = output_dir / "数据预处理消融报告.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
