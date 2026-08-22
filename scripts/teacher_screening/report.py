#!/usr/bin/env python3
"""Compare candidate teachers by target and forecast horizon on common OOF rows."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import io
from scripts.multitarget_forecasting.report import regression_metrics, warning_metrics
from scripts.multitarget_forecasting.run import _parse_seeds
from scripts.teacher_screening import config, data
from scripts.teacher_screening.run import (
    parse_horizons,
    parse_models,
    representative_stations,
    select_screening_stations,
)


def _assert_same(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    if reference.shape != candidate.shape or not np.array_equal(
        reference, candidate, equal_nan=True
    ):
        raise RuntimeError(f"不同教师的{label}不一致，不能严格配对。")


def build_metrics(
    stations: tuple[str, ...],
    selected_models: tuple[str, ...],
    seeds: tuple[int, ...],
    horizon_hours: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for seed in seeds:
            loaded: dict[str, tuple[dict[str, np.ndarray], dict]] = {}
            for model in selected_models:
                path = config.prediction_path(station, model, seed, horizon_hours)
                if not path.exists():
                    raise FileNotFoundError(f"缺少教师OOF结果: {path}")
                arrays, metadata = io.load_archive(path)
                if metadata.get("experiment") != config.EXPERIMENT_ID:
                    raise RuntimeError(f"结果不属于当前教师筛选实验: {path}")
                if tuple(metadata.get("horizon_hours", ())) != horizon_hours:
                    raise RuntimeError(f"结果时距与报告参数不一致: {path}")
                if not np.asarray(arrays["completed_folds"], dtype=bool).all():
                    raise RuntimeError(f"教师OOF折尚未全部完成: {path}")
                loaded[model] = (arrays, metadata)
                runtime_rows.append(
                    {
                        "station": station,
                        "seed": seed,
                        "model": model,
                        "model_label": config.MODEL_LABELS[model],
                        "training_seconds": float(
                            np.asarray(arrays["training_seconds_by_fold"]).sum()
                        ),
                        "inference_seconds": float(
                            np.asarray(arrays["inference_seconds_by_fold"]).sum()
                        ),
                        "fitted_models": int(
                            np.asarray(arrays["fitted_models_by_fold"]).sum()
                        ),
                    }
                )

            reference = loaded[selected_models[0]][0]
            for model in selected_models[1:]:
                candidate = loaded[model][0]
                for key in (
                    "true",
                    "mask",
                    "current",
                    "target_start",
                    "fold_index",
                    "warning_lower_by_fold",
                    "warning_upper_by_fold",
                ):
                    _assert_same(reference[key], candidate[key], key)

            truth = np.asarray(reference["true"], dtype=float)
            truth_mask = np.asarray(reference["mask"], dtype=bool)
            current = np.asarray(reference["current"], dtype=float)
            oof_rows = np.asarray(reference["fold_index"], dtype=int) >= 0
            fold_index_by_row = np.asarray(reference["fold_index"], dtype=int)
            lower_by_fold = np.asarray(
                reference["warning_lower_by_fold"], dtype=float
            )
            upper_by_fold = np.asarray(
                reference["warning_upper_by_fold"], dtype=float
            )
            predictions = {
                model: np.asarray(loaded[model][0]["pred"], dtype=float)
                for model in selected_models
            }
            for horizon_index, horizon in enumerate(horizon_hours):
                for target_index, target in enumerate(config.TARGETS):
                    common = (
                        oof_rows
                        & truth_mask[:, horizon_index, target_index]
                        & np.isfinite(truth[:, horizon_index, target_index])
                        & np.isfinite(current[:, target_index])
                    )
                    for prediction in predictions.values():
                        common &= np.isfinite(
                            prediction[:, horizon_index, target_index]
                        )
                    if not common.any():
                        raise ValueError(
                            f"没有共同OOF评价行: {station}/{target}/{horizon}h"
                        )
                    observed = truth[common, horizon_index, target_index]
                    persistence = current[common, target_index]
                    common_folds = fold_index_by_row[common]
                    lower = lower_by_fold[common_folds, target_index]
                    upper = upper_by_fold[common_folds, target_index]
                    actual_event = np.zeros(len(observed), dtype=bool)
                    actual_event |= np.isfinite(lower) & (observed < lower)
                    actual_event |= np.isfinite(upper) & (observed > upper)
                    persistence_metric = regression_metrics(observed, persistence)
                    persistence_event = np.zeros(len(observed), dtype=bool)
                    persistence_event |= np.isfinite(lower) & (persistence < lower)
                    persistence_event |= np.isfinite(upper) & (persistence > upper)
                    persistence_warning = warning_metrics(
                        actual_event, persistence_event
                    )
                    for model in selected_models:
                        predicted = predictions[model][
                            common, horizon_index, target_index
                        ]
                        metric = regression_metrics(observed, predicted)
                        predicted_event = np.zeros(len(predicted), dtype=bool)
                        predicted_event |= np.isfinite(lower) & (predicted < lower)
                        predicted_event |= np.isfinite(upper) & (predicted > upper)
                        warning = warning_metrics(actual_event, predicted_event)
                        ratio = (
                            metric["rmse"] / persistence_metric["rmse"]
                            if np.isfinite(persistence_metric["rmse"])
                            and persistence_metric["rmse"] > 0
                            else np.nan
                        )
                        rows.append(
                            {
                                "station": station,
                                "seed": seed,
                                "model": model,
                                "model_label": config.MODEL_LABELS[model],
                                "target": target,
                                "horizon_hours": horizon,
                                "horizon_group": config.horizon_group(horizon),
                                "valid_rows": int(common.sum()),
                                **metric,
                                "persistence_rmse": persistence_metric["rmse"],
                                "persistence_nse": persistence_metric["nse"],
                                "warning_events": warning["events"],
                                "warning_tp": warning["tp"],
                                "warning_fp": warning["fp"],
                                "warning_fn": warning["fn"],
                                "warning_tn": warning["tn"],
                                "warning_f1": warning["f1"],
                                "warning_recall": warning["recall"],
                                "warning_false_alarm_rate": warning[
                                    "false_alarm_rate"
                                ],
                                "persistence_warning_f1": persistence_warning[
                                    "f1"
                                ],
                                "rmse_ratio_to_persistence": ratio,
                                "log_rmse_ratio_to_persistence": (
                                    float(np.log(ratio))
                                    if np.isfinite(ratio) and ratio > 0
                                    else np.nan
                                ),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(runtime_rows)


def aggregate_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(
            ["model", "model_label", "target", "horizon_hours", "horizon_group"],
            as_index=False,
        )
        .agg(
            mean_log_rmse_ratio=("log_rmse_ratio_to_persistence", "mean"),
            mean_nse=("nse", "mean"),
            warning_tp=("warning_tp", "sum"),
            warning_fp=("warning_fp", "sum"),
            warning_fn=("warning_fn", "sum"),
            warning_tn=("warning_tn", "sum"),
            mean_persistence_warning_f1=("persistence_warning_f1", "mean"),
            station_seed_cells=("rmse", "size"),
            station_win_rate_vs_persistence=(
                "rmse_ratio_to_persistence",
                lambda values: float(np.mean(np.asarray(values) < 1.0)),
            ),
        )
    )
    grouped["geometric_rmse_ratio_to_persistence"] = np.exp(
        grouped["mean_log_rmse_ratio"]
    )
    grouped["relative_rmse_to_persistence_pct"] = 100.0 * (
        grouped["geometric_rmse_ratio_to_persistence"] - 1.0
    )
    grouped["warning_precision"] = grouped["warning_tp"] / (
        grouped["warning_tp"] + grouped["warning_fp"]
    )
    grouped["warning_recall"] = grouped["warning_tp"] / (
        grouped["warning_tp"] + grouped["warning_fn"]
    )
    grouped["warning_f1"] = (
        2.0
        * grouped["warning_precision"]
        * grouped["warning_recall"]
        / (grouped["warning_precision"] + grouped["warning_recall"])
    )
    grouped["warning_false_alarm_rate"] = grouped["warning_fp"] / (
        grouped["warning_fp"] + grouped["warning_tn"]
    )
    grouped["rank_within_target_horizon"] = grouped.groupby(
        ["target", "horizon_hours"]
    )["mean_log_rmse_ratio"].rank(method="min")
    return grouped.sort_values(
        ["target", "horizon_hours", "rank_within_target_horizon", "model"]
    ).reset_index(drop=True)


def horizon_weights(
    cell_summary: pd.DataFrame, temperature: float = 0.10
) -> pd.DataFrame:
    """Create training-only reliability weights; do not score this fit in-sample."""

    if temperature <= 0:
        raise ValueError("temperature必须大于0。")
    parts: list[pd.DataFrame] = []
    for (_, _), part in cell_summary.groupby(["target", "horizon_hours"]):
        part = part.copy()
        losses = np.asarray(part["mean_log_rmse_ratio"], dtype=float)
        logits = -(losses - np.nanmin(losses)) / temperature
        logits = np.clip(logits, -50.0, 50.0)
        weights = np.exp(logits)
        weights /= weights.sum()
        part["teacher_weight"] = weights
        part["teacher_reliability"] = np.maximum(
            0.0,
            1.0 - np.asarray(part["geometric_rmse_ratio_to_persistence"], dtype=float),
        )
        part["eligible_for_distillation"] = part[
            "geometric_rmse_ratio_to_persistence"
        ].lt(1.0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def winner_change_diagnostics(cell_summary: pd.DataFrame) -> pd.DataFrame:
    winners = cell_summary.loc[
        cell_summary["rank_within_target_horizon"].eq(1)
    ].copy()
    rows: list[dict[str, object]] = []
    for target, part in winners.groupby("target"):
        ordered = part.sort_values(["horizon_hours", "model"])
        by_horizon = (
            ordered.groupby("horizon_hours")["model_label"]
            .apply(lambda values: "+".join(sorted(set(values))))
            .to_dict()
        )
        sequence = [by_horizon[horizon] for horizon in sorted(by_horizon)]
        changes = sum(
            previous != current
            for previous, current in zip(sequence, sequence[1:], strict=False)
        )
        rows.append(
            {
                "target": target,
                "winner_model_count": len(set(sequence)),
                "winner_changes_with_horizon": len(set(sequence)) > 1,
                "adjacent_anchor_change_count": changes,
                "winner_by_horizon": "；".join(
                    f"{horizon}h={by_horizon[horizon]}" for horizon in sorted(by_horizon)
                ),
            }
        )
    return pd.DataFrame(rows)


def global_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    result = (
        cell_summary.groupby(["model", "model_label"], as_index=False)
        .agg(
            mean_log_rmse_ratio=("mean_log_rmse_ratio", "mean"),
            mean_nse=("mean_nse", "mean"),
            mean_warning_f1=("warning_f1", "mean"),
            target_horizon_wins=(
                "rank_within_target_horizon",
                lambda values: int(np.sum(np.asarray(values) == 1)),
            ),
            evaluated_target_horizons=("rank_within_target_horizon", "size"),
        )
    )
    result["geometric_rmse_ratio_to_persistence"] = np.exp(
        result["mean_log_rmse_ratio"]
    )
    result["relative_rmse_to_persistence_pct"] = 100.0 * (
        result["geometric_rmse_ratio_to_persistence"] - 1.0
    )
    return result.sort_values("mean_log_rmse_ratio").reset_index(drop=True)


def group_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    result = (
        cell_summary.groupby(
            ["model", "model_label", "horizon_group"], as_index=False
        )
        .agg(
            mean_log_rmse_ratio=("mean_log_rmse_ratio", "mean"),
            mean_nse=("mean_nse", "mean"),
            mean_warning_f1=("warning_f1", "mean"),
        )
    )
    result["geometric_rmse_ratio_to_persistence"] = np.exp(
        result["mean_log_rmse_ratio"]
    )
    result["relative_rmse_to_persistence_pct"] = 100.0 * (
        result["geometric_rmse_ratio_to_persistence"] - 1.0
    )
    result["rank_within_horizon_group"] = result.groupby("horizon_group")[
        "mean_log_rmse_ratio"
    ].rank(method="min")
    return result.sort_values(
        ["horizon_group", "rank_within_horizon_group"]
    ).reset_index(drop=True)


def _text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def write_report(
    metrics: pd.DataFrame,
    runtime: pd.DataFrame,
    selected_models: tuple[str, ...],
    horizon_hours: tuple[int, ...],
) -> None:
    cells = aggregate_cells(metrics)
    weights = horizon_weights(cells)
    changes = winner_change_diagnostics(cells)
    overall = global_summary(cells)
    groups = group_summary(cells)
    runtime_summary = (
        runtime.groupby(["model", "model_label"], as_index=False)
        .agg(
            total_training_seconds=("training_seconds", "sum"),
            total_inference_seconds=("inference_seconds", "sum"),
            total_fitted_models=("fitted_models", "sum"),
        )
        .sort_values("total_training_seconds")
    )
    report = f"""# 多候选教师训练期因果OOF初筛报告

- 候选教师：{', '.join(config.MODEL_LABELS[item] for item in selected_models)}。
- 初筛时距：{', '.join(f'{hour}小时' for hour in horizon_hours)}；初筛结论不会自动外推到未运行的时距。
- 所有指标使用冻结E预处理和冻结混合输出表示，各折的变换参数只由该折历史拟合区间估计。
- 评价严格使用各教师共同拥有预测的前向OOF行，未读取2024和2025标签。
- 教师排名按相对持续性的配对几何RMSE汇总；NSE作为辅助指标。

## 全局表现

```text
{_text(overall)}
```

## 短、中、长时距表现

```text
{_text(groups)}
```

## 教师是否随时距改变

```text
{_text(changes)}
```

## 计算开销

```text
{_text(runtime_summary)}
```

## 解释约束

`教师指标时距权重.csv`中的权重只允许用于下一阶段训练和2024验证，不能再把同一OOF上的加权结果当作无偏性能。若某个指标时距的最佳教师仍不如持续性，该单元的蒸馏可靠度为0，应保留纯监督学生，而不是为了指标强制蒸馏。若教师胜者随时距变化，则补齐入围教师的18个时距，并比较单一教师、分时距硬选择和分时距软加权三种学生。
"""
    output = config.OUTPUT_DIR / "报告"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "教师全局表现.csv": overall,
        "教师分时距组表现.csv": groups,
        "教师指标时距排名.csv": cells,
        "教师指标时距权重.csv": weights,
        "教师随时距变化诊断.csv": changes,
        "教师逐站指标时距明细.csv": metrics,
        "教师计算开销.csv": runtime_summary,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, encoding="utf-8-sig")
    path = output / "多候选教师因果OOF初筛报告.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    station_group.add_argument("--representative-stations", action="store_true")
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument(
        "--horizons", default=",".join(map(str, config.ANCHOR_HOURS))
    )
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    args = parser.parse_args()
    panel = data.load_training_panel()
    stations = select_screening_stations(
        panel,
        stations=args.stations,
        all_stations=args.all_stations,
        representatives=args.representative_stations,
    )
    selected_models = parse_models(args.models)
    horizons = parse_horizons(args.horizons)
    metrics, runtime = build_metrics(
        stations,
        selected_models,
        _parse_seeds(args.seeds),
        horizons,
    )
    write_report(metrics, runtime, selected_models, horizons)


if __name__ == "__main__":
    main()
