#!/usr/bin/env python3
"""Relate train-only water-quality dynamics to absolute-versus-delta skill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config, data, io
from scripts.multitarget_forecasting import head_ablation_config
from scripts.multitarget_forecasting.report import regression_metrics
from scripts.multitarget_forecasting.run import select_stations


OUTPUT_DIR = config.OUTPUT_DIR / "验证集" / "输出表示与波动机制"
CONTEXT = "24h"


def normalized_four_hour_dynamics(
    train: dict[str, np.ndarray], target_index: int
) -> dict[str, float]:
    current = np.asarray(train["current"], dtype=float)[:, target_index]
    current_mask = (
        np.asarray(train["current_mask"], dtype=bool)[:, target_index]
        & np.isfinite(current)
    )
    approved_current = current[current_mask]
    four_hour_delta = np.asarray(train["y_delta"], dtype=float)[:, 0, target_index]
    delta_mask = (
        np.asarray(train["y_mask"], dtype=bool)[:, 0, target_index]
        & np.isfinite(four_hour_delta)
    )
    approved_delta = np.abs(four_hour_delta[delta_mask])
    if not len(approved_current) or not len(approved_delta):
        return {
            "training_value_rows": int(len(approved_current)),
            "training_delta_rows": int(len(approved_delta)),
            "value_iqr": np.nan,
            "median_absolute_4h_delta": np.nan,
            "normalized_4h_dynamics": np.nan,
        }
    lower, upper = np.quantile(approved_current, (0.25, 0.75))
    value_iqr = float(upper - lower)
    median_delta = float(np.median(approved_delta))
    score = median_delta / value_iqr if value_iqr > 0 else np.nan
    return {
        "training_value_rows": int(len(approved_current)),
        "training_delta_rows": int(len(approved_delta)),
        "value_iqr": value_iqr,
        "median_absolute_4h_delta": median_delta,
        "normalized_4h_dynamics": score,
    }


def _load_mode(station: str, mode: str, seed: int):
    path = config.prediction_path(station, CONTEXT, mode, seed)
    if not path.exists():
        raise FileNotFoundError(f"缺少{config.TARGET_MODE_LABELS[mode]}联合预测: {path}")
    arrays, metadata = io.load_archive(path)
    if metadata.get("context") != CONTEXT or metadata.get("target_mode") != mode:
        raise RuntimeError(f"输出表示结果协议不匹配: {path}")
    return arrays


def _aligned_mode_rows(
    station: str, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    absolute = _load_mode(station, "absolute", seed)
    delta = _load_mode(station, "delta", seed)
    for key in ("true", "mask", "current", "target_start"):
        left = np.asarray(absolute[key])
        right = np.asarray(delta[key])
        allow_nan = np.issubdtype(left.dtype, np.inexact)
        aligned = (
            np.array_equal(left, right, equal_nan=True)
            if allow_nan
            else np.array_equal(left, right)
        )
        if left.shape != right.shape or not aligned:
            raise RuntimeError(f"原值与变化量结果未严格对齐: {station} / {key}")

    truth = np.asarray(absolute["true"], dtype=float)
    mask = np.asarray(absolute["mask"], dtype=bool)
    pred_absolute = np.asarray(absolute["pred"], dtype=float)
    pred_delta = np.asarray(delta["pred"], dtype=float)
    horizon_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(config.TARGETS):
        relative_values: list[float] = []
        delta_wins: list[bool] = []
        for horizon_index, horizon_hours in enumerate(config.HORIZON_HOURS):
            valid = (
                mask[:, horizon_index, target_index]
                & np.isfinite(truth[:, horizon_index, target_index])
                & np.isfinite(pred_absolute[:, horizon_index, target_index])
                & np.isfinite(pred_delta[:, horizon_index, target_index])
            )
            observed = truth[valid, horizon_index, target_index]
            predicted_absolute = pred_absolute[valid, horizon_index, target_index]
            predicted_delta = pred_delta[valid, horizon_index, target_index]
            absolute_rmse = regression_metrics(observed, predicted_absolute)["rmse"]
            delta_rmse = regression_metrics(observed, predicted_delta)["rmse"]
            relative = (
                100.0 * (delta_rmse / absolute_rmse - 1.0)
                if np.isfinite(absolute_rmse) and absolute_rmse > 0
                else np.nan
            )
            relative_values.append(relative)
            delta_wins.append(bool(delta_rmse < absolute_rmse))
            horizon_rows.append(
                {
                    "station": station,
                    "seed": seed,
                    "target": target,
                    "horizon_hours": horizon_hours,
                    "valid_rows": int(valid.sum()),
                    "absolute_rmse": absolute_rmse,
                    "delta_rmse": delta_rmse,
                    "delta_relative_absolute_rmse_pct": relative,
                    "delta_wins": bool(delta_rmse < absolute_rmse),
                }
            )
        finite_relative = np.asarray(relative_values, dtype=float)
        finite_relative = finite_relative[np.isfinite(finite_relative)]
        mean_relative = (
            float(np.mean(finite_relative)) if len(finite_relative) else np.nan
        )
        summary_rows.append(
            {
                "station": station,
                "seed": seed,
                "target": target,
                "delta_relative_absolute_rmse_pct": mean_relative,
                "delta_horizon_win_rate": float(np.mean(delta_wins)),
                "empirical_preferred_mode": (
                    "delta" if np.isfinite(mean_relative) and mean_relative < 0 else "absolute"
                ),
                "current_mixed_mode": head_ablation_config.TARGET_OUTPUT_MODES[target],
            }
        )
    return horizon_rows, summary_rows


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    ranked_x = x.loc[valid].rank(method="average")
    ranked_y = y.loc[valid].rank(method="average")
    return float(ranked_x.corr(ranked_y))


def _within_target_rank_correlation(summary: pd.DataFrame) -> float:
    valid = summary.loc[
        np.isfinite(summary["normalized_4h_dynamics"])
        & np.isfinite(summary["delta_relative_absolute_rmse_pct"])
    ].copy()
    if len(valid) < 3:
        return np.nan
    valid["dynamics_rank_within_target"] = valid.groupby("target")[
        "normalized_4h_dynamics"
    ].rank(method="average", pct=True)
    valid["advantage_rank_within_target"] = valid.groupby("target")[
        "delta_relative_absolute_rmse_pct"
    ].rank(method="average", pct=True)
    return float(
        valid["dynamics_rank_within_target"].corr(
            valid["advantage_rank_within_target"]
        )
    )


def station_bootstrap_correlation(
    summary: pd.DataFrame,
    repeats: int = 2000,
    seed: int = 20260821,
    *,
    within_target: bool = False,
) -> tuple[float, float]:
    stations = np.asarray(sorted(summary["station"].unique()), dtype=object)
    if len(stations) < 2:
        return np.nan, np.nan
    grouped = {station: summary.loc[summary["station"].eq(station)] for station in stations}
    generator = np.random.default_rng(seed)
    correlations: list[float] = []
    for _ in range(repeats):
        selected = generator.choice(stations, size=len(stations), replace=True)
        sample = pd.concat([grouped[station] for station in selected], ignore_index=True)
        correlation = (
            _within_target_rank_correlation(sample)
            if within_target
            else _spearman(
                sample["normalized_4h_dynamics"],
                sample["delta_relative_absolute_rmse_pct"],
            )
        )
        if np.isfinite(correlation):
            correlations.append(correlation)
    if not correlations:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(correlations, (0.025, 0.975)))


def build_analysis(
    panel: pd.DataFrame, stations: tuple[str, ...], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_parts: list[dict[str, object]] = []
    summary_parts: list[dict[str, object]] = []
    for station in stations:
        dataset = data.build_station_dataset(panel, station)
        train = data.split_by_time(dataset)["train"]
        dynamics = {
            target: normalized_four_hour_dynamics(train, index)
            for index, target in enumerate(config.TARGETS)
        }
        horizon_rows, summary_rows = _aligned_mode_rows(station, seed)
        horizon_parts.extend(horizon_rows)
        for row in summary_rows:
            row.update(dynamics[str(row["target"])])
            row["mapping_matches_empirical_preference"] = (
                row["current_mixed_mode"] == row["empirical_preferred_mode"]
            )
            summary_parts.append(row)

    horizon = pd.DataFrame(horizon_parts)
    summary = pd.DataFrame(summary_parts)
    target_summary = (
        summary.groupby("target", as_index=False)
        .agg(
            平均归一化4小时波动=("normalized_4h_dynamics", "mean"),
            变化量相对原值RMSE变化百分比=("delta_relative_absolute_rmse_pct", "mean"),
            变化量站点胜率=(
                "empirical_preferred_mode",
                lambda values: float(np.mean(np.asarray(values) == "delta")),
            ),
            当前映射一致站点比例=("mapping_matches_empirical_preference", "mean"),
        )
    )
    target_summary["指标总体经验首选"] = np.where(
        target_summary["变化量相对原值RMSE变化百分比"] < 0,
        "变化量",
        "原值",
    )
    target_summary["当前混合映射"] = target_summary["target"].map(
        {
            target: ("变化量" if mode == "delta" else "原值")
            for target, mode in head_ablation_config.TARGET_OUTPUT_MODES.items()
        }
    )

    pooled = _spearman(
        summary["normalized_4h_dynamics"],
        summary["delta_relative_absolute_rmse_pct"],
    )
    lower, upper = station_bootstrap_correlation(summary)
    controlled = _within_target_rank_correlation(summary)
    controlled_lower, controlled_upper = station_bootstrap_correlation(
        summary, within_target=True
    )
    correlation_rows = [
        {
            "scope": "全部站点指标池化",
            "target": "全部",
            "cells": len(summary),
            "spearman_rho": pooled,
            "station_bootstrap_95pct_lower": lower,
            "station_bootstrap_95pct_upper": upper,
        },
        {
            "scope": "指标内秩控制指标类别",
            "target": "全部",
            "cells": len(summary),
            "spearman_rho": controlled,
            "station_bootstrap_95pct_lower": controlled_lower,
            "station_bootstrap_95pct_upper": controlled_upper,
        },
    ]
    for target, frame in summary.groupby("target"):
        correlation_rows.append(
            {
                "scope": "指标内跨站点",
                "target": target,
                "cells": len(frame),
                "spearman_rho": _spearman(
                    frame["normalized_4h_dynamics"],
                    frame["delta_relative_absolute_rmse_pct"],
                ),
                "station_bootstrap_95pct_lower": np.nan,
                "station_bootstrap_95pct_upper": np.nan,
            }
        )
    correlation = pd.DataFrame(correlation_rows)
    return summary, horizon, target_summary, correlation


def _text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def build_report(
    summary: pd.DataFrame,
    horizon: pd.DataFrame,
    target_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    seed: int,
) -> str:
    controlled = correlation.loc[
        correlation["scope"].eq("指标内秩控制指标类别")
    ].iloc[0]
    rho = float(controlled["spearman_rho"])
    lower = float(controlled["station_bootstrap_95pct_lower"])
    upper = float(controlled["station_bootstrap_95pct_upper"])
    if np.isfinite(lower) and lower > 0:
        decision = "波动越大越倾向原值的方向在站点重采样下保持为正。"
    elif np.isfinite(upper) and upper < 0:
        decision = "结果与波动越大越倾向原值的假设相反。"
    else:
        decision = "站点重采样区间包含0，当前不能确认波动程度单独决定输出表示。"
    mismatched = target_summary.loc[
        target_summary["指标总体经验首选"]
        .ne(target_summary["当前混合映射"]),
        "target",
    ].tolist()
    mapping_decision = (
        "指标总体方向与当前混合映射全部一致。"
        if not mismatched
        else "与当前混合映射方向不一致的指标：" + "、".join(mismatched) + "。"
    )
    return f"""# 训练期水质波动与原值/变化量输出机制报告

- 随机种子：{seed}。
- 波动指标只由2022—2023训练集计算；输出表示优势由2024验证集严格配对RMSE计算。
- 正的“变化量相对原值RMSE变化”表示变化量更差，因而更倾向原值。
- 相关分析单元为站点×指标，95%区间按站点整块重采样，不把18个相关时距当成独立样本。
- 自动判定：{decision} {mapping_decision}

## 指标总体结果

```text
{_text(target_summary)}
```

## 波动与表示优势关联

```text
{_text(correlation)}
```

## 逐站点指标结果

```text
{_text(summary)}
```

指标内秩控制后的关联={rho:.6f}，站点重采样95%区间=[{lower:.6f}, {upper:.6f}]。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="分析训练期波动与原值/变化量输出优势")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--seed", type=int, default=config.SCREENING_SEED)
    args = parser.parse_args()
    if args.seed not in config.FORMAL_SEEDS:
        parser.error(f"--seed必须属于冻结种子: {config.FORMAL_SEEDS}")
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    summary, horizon, target_summary, correlation = build_analysis(
        panel, stations, args.seed
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "逐站点指标波动与表示优势.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(OUTPUT_DIR / "逐站点指标时距原值变化量配对.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(OUTPUT_DIR / "分指标波动与表示结果.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(OUTPUT_DIR / "波动与表示优势关联.csv", index=False, encoding="utf-8-sig")
    report = build_report(summary, horizon, target_summary, correlation, args.seed)
    path = OUTPUT_DIR / "训练期波动与原值变化量输出机制报告.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
