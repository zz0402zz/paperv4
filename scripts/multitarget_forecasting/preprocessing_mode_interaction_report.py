#!/usr/bin/env python3
"""Report whether preprocessing changes each target's absolute/delta preference."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import head_ablation_config as head_config
from scripts.multitarget_forecasting import preprocessing_ablation_config as old_prep
from scripts.multitarget_forecasting import preprocessing_component_config as component
from scripts.multitarget_forecasting import preprocessing_mode_interaction_config as config
from scripts.multitarget_forecasting.report import (
    event_flags,
    regression_metrics,
    warning_metrics,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def base_result_path(station: str, preprocessing: str, seed: int):
    if preprocessing == "A":
        return head_config.prediction_path(station, "mixed_linear", seed)
    if preprocessing == "C":
        return component.prediction_path(station, "standard_huber", seed)
    if preprocessing == "D":
        return old_prep.prediction_path(station, "robust_huber", seed)
    if preprocessing == "E":
        return old_prep.prediction_path(station, "robust_huber_log", seed)
    raise ValueError(f"未知预处理链: {preprocessing}")


def _load(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少交互消融结果: {path}")
    return io.load_archive(path)


def _aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str):
    for key in (
        "true",
        "mask",
        "current",
        "current_mask",
        "target_start",
        "warning_lower",
        "warning_upper",
    ):
        left = np.asarray(reference[key])
        right = np.asarray(candidate[key])
        allow_nan = np.issubdtype(left.dtype, np.inexact)
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if allow_nan
            else np.array_equal(left, right)
        )
        if left.shape != right.shape or not equal:
            raise RuntimeError(f"交互消融未严格对齐: {label} / {key}")


def build_tables(
    stations: tuple[str, ...], seeds: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for preprocessing in config.PREPROCESSINGS:
            preprocessing_label = str(config.PREPROCESSING_SPECS[preprocessing]["label"])
            for seed in seeds:
                base_arrays, base_metadata = _load(
                    base_result_path(station, preprocessing, seed)
                )
                sources: dict[str, tuple[dict[str, np.ndarray], dict]] = {
                    "base": (base_arrays, base_metadata)
                }
                for flip in config.FLIPS:
                    arrays, metadata = _load(
                        config.prediction_path(station, preprocessing, flip, seed)
                    )
                    _aligned(base_arrays, arrays, f"{station}/{preprocessing}/{flip}")
                    sources[flip] = (arrays, metadata)

                for mapping_variant, (arrays, metadata) in sources.items():
                    prediction = np.asarray(arrays["pred"], dtype=float)
                    truth = np.asarray(arrays["true"], dtype=float)
                    mask = np.asarray(arrays["mask"], dtype=bool)
                    current = np.asarray(arrays["current"], dtype=float)
                    lower = np.asarray(arrays["warning_lower"], dtype=float)
                    upper = np.asarray(arrays["warning_upper"], dtype=float)
                    mapping_label = (
                        "当前混合映射"
                        if mapping_variant == "base"
                        else config.flip_label(mapping_variant)
                    )
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
                            metric = regression_metrics(observed, predicted)
                            shared = {
                                "station": station,
                                "seed": seed,
                                "preprocessing": preprocessing,
                                "preprocessing_label": preprocessing_label,
                                "mapping_variant": mapping_variant,
                                "mapping_label": mapping_label,
                                "target": target,
                                "horizon_hours": horizon_hours,
                            }
                            forecast_rows.append(
                                {**shared, "valid_rows": int(valid.sum()), **metric}
                            )
                            actual_event = event_flags(
                                observed, lower[target_index], upper[target_index]
                            )
                            warning_rows.append(
                                {
                                    **shared,
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
                    runtime_rows.append(
                        {
                            "station": station,
                            "seed": seed,
                            "preprocessing": preprocessing,
                            "preprocessing_label": preprocessing_label,
                            "mapping_variant": mapping_variant,
                            "mapping_label": mapping_label,
                            "selected_epoch": int(arrays["selected_epoch"]),
                            "training_seconds": float(arrays["training_seconds"]),
                            "inference_seconds": float(arrays["inference_seconds"]),
                            "parameter_count": int(arrays["parameter_count"]),
                            "source_experiment": metadata.get("experiment", ""),
                        }
                    )
    return pd.DataFrame(forecast_rows), pd.DataFrame(warning_rows), pd.DataFrame(runtime_rows)


def build_mode_detail(forecast: pd.DataFrame) -> pd.DataFrame:
    keys = ["station", "seed", "preprocessing", "preprocessing_label", "target", "horizon_hours"]
    rmse_pivot = forecast.pivot(
        index=keys, columns="mapping_variant", values="rmse"
    ).reset_index()
    rmse_pivot = rmse_pivot.rename(
        columns={variant: f"{variant}_rmse" for variant in ("base", *config.FLIPS)}
    )
    nse_pivot = forecast.pivot(
        index=keys, columns="mapping_variant", values="nse"
    ).reset_index()
    nse_pivot = nse_pivot.rename(
        columns={variant: f"{variant}_nse" for variant in ("base", *config.FLIPS)}
    )
    pivot = rmse_pivot.merge(nse_pivot, on=keys, how="inner", validate="one_to_one")
    parts: list[pd.DataFrame] = []
    for flip, target in config.FLIP_TARGETS.items():
        part = pivot.loc[pivot["target"].eq(target), keys].copy()
        target_rows = pivot["target"].eq(target)
        base_rmse = pivot.loc[target_rows, "base_rmse"].to_numpy(float)
        alternative_rmse = pivot.loc[target_rows, f"{flip}_rmse"].to_numpy(float)
        base_nse = pivot.loc[target_rows, "base_nse"].to_numpy(float)
        alternative_nse = pivot.loc[target_rows, f"{flip}_nse"].to_numpy(float)
        base_mode = config.BASE_MODES[target]
        alternative_mode = config.flipped_modes(flip)[target]
        part["flip"] = flip
        part["base_mode"] = base_mode
        part["alternative_mode"] = alternative_mode
        part["base_rmse"] = base_rmse
        part["alternative_rmse"] = alternative_rmse
        part["alternative_relative_base_rmse_pct"] = 100.0 * (
            alternative_rmse / base_rmse - 1.0
        )
        part["alternative_log_base_rmse_ratio"] = np.log(
            alternative_rmse / base_rmse
        )
        part["base_nse"] = base_nse
        part["alternative_nse"] = alternative_nse
        part["alternative_minus_base_nse"] = alternative_nse - base_nse
        part["alternative_wins"] = alternative_rmse < base_rmse
        delta_rmse = base_rmse if base_mode == "delta" else alternative_rmse
        absolute_rmse = base_rmse if base_mode == "absolute" else alternative_rmse
        delta_nse = base_nse if base_mode == "delta" else alternative_nse
        absolute_nse = base_nse if base_mode == "absolute" else alternative_nse
        part["delta_relative_absolute_rmse_pct"] = 100.0 * (
            delta_rmse / absolute_rmse - 1.0
        )
        part["delta_log_absolute_rmse_ratio"] = np.log(
            delta_rmse / absolute_rmse
        )
        part["delta_minus_absolute_nse"] = delta_nse - absolute_nse
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _station_bootstrap_geometric_percent_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    seed: int,
    repeats: int = 2_000,
) -> tuple[float, float]:
    """Bootstrap station-clustered mean log ratios, then return percent ratios."""

    station_values = (
        frame.groupby("station", as_index=False)[column]
        .mean()[column]
        .to_numpy(dtype=float)
    )
    station_values = station_values[np.isfinite(station_values)]
    if not len(station_values):
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    sampled = station_values[
        generator.integers(
            0,
            len(station_values),
            size=(repeats, len(station_values)),
        )
    ]
    means = sampled.mean(axis=1)
    interval = 100.0 * np.expm1(np.quantile(means, (0.025, 0.975)))
    return tuple(interval.tolist())


def _warning_metrics_from_counts(row: pd.Series) -> dict[str, float]:
    tp, fp, fn, tn = (float(row[name]) for name in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def flipped_target_warning_summary(warning: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for preprocessing in config.PREPROCESSINGS:
        for flip, target in config.FLIP_TARGETS.items():
            for mapping_variant, role in (("base", "当前表示"), (flip, "替代表示")):
                part = warning.loc[
                    warning["preprocessing"].eq(preprocessing)
                    & warning["mapping_variant"].eq(mapping_variant)
                    & warning["target"].eq(target)
                ]
                counts = part[["tp", "fp", "fn", "tn"]].sum()
                rows.append(
                    {
                        "preprocessing": preprocessing,
                        "preprocessing_label": config.PREPROCESSING_SPECS[preprocessing]["label"],
                        "target": target,
                        "base_mode": config.BASE_MODES[target],
                        "role": role,
                        "mode": (
                            config.BASE_MODES[target]
                            if mapping_variant == "base"
                            else config.flipped_modes(flip)[target]
                        ),
                        **{name: int(counts[name]) for name in ("tp", "fp", "fn", "tn")},
                        **_warning_metrics_from_counts(counts),
                    }
                )
    return pd.DataFrame(rows)


def _text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def build_report(
    forecast: pd.DataFrame,
    warning: pd.DataFrame,
    runtime: pd.DataFrame,
    detail: pd.DataFrame,
    seeds: tuple[int, ...],
) -> tuple[str, dict[str, pd.DataFrame]]:
    mode_summary = (
        detail.groupby(
            [
                "preprocessing",
                "preprocessing_label",
                "target",
                "base_mode",
                "alternative_mode",
            ],
            as_index=False,
        )
        .agg(
            替代表示相对当前RMSE平均对数比=(
                "alternative_log_base_rmse_ratio",
                "mean",
            ),
            替代表示胜率=("alternative_wins", "mean"),
            变化量相对原值RMSE平均对数比=(
                "delta_log_absolute_rmse_ratio",
                "mean",
            ),
            替代表示相对当前NSE变化=("alternative_minus_base_nse", "mean"),
            变化量相对原值NSE变化=("delta_minus_absolute_nse", "mean"),
            配对单元数=("base_rmse", "size"),
        )
    )
    mode_summary["替代表示相对当前RMSE几何变化百分比"] = 100.0 * np.expm1(
        mode_summary.pop("替代表示相对当前RMSE平均对数比")
    )
    mode_summary["变化量相对原值RMSE几何变化百分比"] = 100.0 * np.expm1(
        mode_summary.pop("变化量相对原值RMSE平均对数比")
    )
    interval_rows: list[dict[str, object]] = []
    grouping = [
        "preprocessing",
        "preprocessing_label",
        "target",
        "base_mode",
        "alternative_mode",
    ]
    for group_index, (group_keys, group) in enumerate(
        detail.groupby(grouping, sort=False)
    ):
        lower, upper = _station_bootstrap_geometric_percent_interval(
            group,
            "delta_log_absolute_rmse_ratio",
            seed=20_260_821 + group_index,
        )
        interval_rows.append(
            {
                **dict(zip(grouping, group_keys, strict=True)),
                "RMSE优势站点聚类95CI下限": lower,
                "RMSE优势站点聚类95CI上限": upper,
            }
        )
    mode_summary = mode_summary.merge(
        pd.DataFrame(interval_rows),
        on=grouping,
        how="left",
        validate="one_to_one",
    )
    mode_summary["经验首选表示"] = np.where(
        mode_summary["变化量相对原值RMSE几何变化百分比"] < 0,
        "delta",
        "absolute",
    )
    mode_summary["置信结论"] = np.select(
        (
            mode_summary["RMSE优势站点聚类95CI上限"] < 0,
            mode_summary["RMSE优势站点聚类95CI下限"] > 0,
        ),
        ("变化量稳定更优", "原值稳定更优"),
        default="不确定",
    )
    mode_summary["是否保持当前映射"] = mode_summary["经验首选表示"].eq(
        mode_summary["base_mode"]
    )
    original_preference = mode_summary.loc[
        mode_summary["preprocessing"].eq("A"), ["target", "经验首选表示"]
    ].rename(columns={"经验首选表示": "A原始首选表示"})
    mode_summary = mode_summary.merge(original_preference, on="target", how="left")
    mode_summary["是否相对A发生反转"] = mode_summary["经验首选表示"].ne(
        mode_summary["A原始首选表示"]
    )

    pivot = detail.pivot(
        index=["station", "seed", "target", "horizon_hours"],
        columns="preprocessing",
        values="delta_log_absolute_rmse_ratio",
    ).reset_index()
    interaction_parts: list[pd.DataFrame] = []
    for preprocessing in ("C", "D", "E"):
        part = pivot[["station", "seed", "target", "horizon_hours"]].copy()
        part["preprocessing"] = preprocessing
        part["preprocessing_label"] = config.PREPROCESSING_SPECS[preprocessing]["label"]
        part["相对A的变化量原值RMSE比对数变化"] = (
            pivot[preprocessing] - pivot["A"]
        )
        part["相对A单元首选反转"] = (pivot[preprocessing] < 0) != (pivot["A"] < 0)
        interaction_parts.append(part)
    interaction_detail = pd.concat(interaction_parts, ignore_index=True)
    interaction_summary = (
        interaction_detail.groupby(["preprocessing", "preprocessing_label", "target"], as_index=False)
        .agg(
            相对A的变化量原值RMSE比对数变化=(
                "相对A的变化量原值RMSE比对数变化",
                "mean",
            ),
            单元首选反转率=("相对A单元首选反转", "mean"),
        )
    )
    interaction_summary["相对A的变化量原值RMSE比几何变化百分比"] = (
        100.0
        * np.expm1(
            interaction_summary.pop("相对A的变化量原值RMSE比对数变化")
        )
    )

    keys = ["station", "seed", "preprocessing", "preprocessing_label", "target", "horizon_hours"]
    all_rmse = forecast.pivot(
        index=keys, columns="mapping_variant", values="rmse"
    ).reset_index()
    all_rmse = all_rmse.rename(
        columns={variant: f"{variant}_rmse" for variant in ("base", *config.FLIPS)}
    )
    all_nse = forecast.pivot(
        index=keys, columns="mapping_variant", values="nse"
    ).reset_index()
    all_nse = all_nse.rename(
        columns={variant: f"{variant}_nse" for variant in ("base", *config.FLIPS)}
    )
    all_pivot = all_rmse.merge(
        all_nse, on=keys, how="inner", validate="one_to_one"
    )
    side_parts: list[pd.DataFrame] = []
    for flip in config.FLIPS:
        part = all_pivot[["preprocessing", "preprocessing_label"]].copy()
        part["flip"] = flip
        part["flip_label"] = config.flip_label(flip)
        part["全指标替代相对基准RMSE对数比"] = np.log(
            all_pivot[f"{flip}_rmse"] / all_pivot["base_rmse"]
        )
        part["全指标替代NSE变化"] = (
            all_pivot[f"{flip}_nse"] - all_pivot["base_nse"]
        )
        part["全指标替代胜出"] = (
            all_pivot[f"{flip}_rmse"] < all_pivot["base_rmse"]
        )
        side_parts.append(part)
    side_effect_detail = pd.concat(side_parts, ignore_index=True)
    side_effect_summary = (
        side_effect_detail.groupby(
            ["preprocessing", "preprocessing_label", "flip", "flip_label"],
            as_index=False,
        )
        .agg(
            全指标RMSE平均对数比=("全指标替代相对基准RMSE对数比", "mean"),
            全指标NSE变化=("全指标替代NSE变化", "mean"),
            全指标胜率=("全指标替代胜出", "mean"),
        )
    )
    side_effect_summary["全指标RMSE几何变化百分比"] = 100.0 * np.expm1(
        side_effect_summary.pop("全指标RMSE平均对数比")
    )
    warning_summary = flipped_target_warning_summary(warning)
    runtime_summary = (
        runtime.loc[runtime["mapping_variant"].ne("base")]
        .groupby(["preprocessing", "preprocessing_label"], as_index=False)
        .agg(
            平均选定轮数=("selected_epoch", "mean"),
            平均训练秒数=("training_seconds", "mean"),
            平均推理秒数=("inference_seconds", "mean"),
        )
    )
    changed = mode_summary.loc[
        mode_summary["是否相对A发生反转"]
        & mode_summary["preprocessing"].ne("A"),
        [
            "preprocessing",
            "target",
            "A原始首选表示",
            "经验首选表示",
            "置信结论",
        ],
    ]
    confident_changed = changed.loc[changed["置信结论"].ne("不确定")]
    if changed.empty:
        point_decision = "A/C/D/E的指标总体首选表示完全一致。"
    elif confident_changed.empty:
        point_decision = (
            f"出现{len(changed)}个点估计首选反转，但其站点聚类95%CI均跨越0，"
            "不能确认预处理改变首选表示。"
        )
    else:
        point_decision = (
            f"有{len(confident_changed)}个预处理后的首选反转未跨越0，详见主表。"
        )
    uncertain_count = int(mode_summary["置信结论"].eq("不确定").sum())
    decision = (
        f"{point_decision} 20个预处理×指标组合中有"
        f"{uncertain_count}个的站点聚类95%CI跨越0。"
    )
    report = f"""# 预处理与原值/变化量输出表示交互消融报告

- 随机种子：{', '.join(map(str, seeds))}。
- A/C/D/E分别表示原始、仅Huber、稳健＋Huber、稳健＋Huber＋对数。
- 每个候选每次只翻转一个指标，其他四个指标表示不变；模型仍一次联合输出18时距×5指标。
- 正的“变化量相对原值RMSE几何变化”表示原值更优，负值表示变化量更优。
- RMSE使用对称的配对对数比后取几何变化，避免对互为倒数的百分比作算术平均导致方向矛盾。
- 首选表示以RMSE主指标判定，NSE同时报告；95%CI按站点整体重抽样，不把18个时距当成独立样本。
- 本轮仅用2024验证集和种子42筛选，不读取2025测试标签。
- 自动判定：{decision}

## 逐预处理逐指标表示选择

```text
{_text(mode_summary)}
```

## 相对原始A的表示优势交互变化

```text
{_text(interaction_summary)}
```

## 翻转一项对全五指标的联合影响

```text
{_text(side_effect_summary)}
```

## 被翻转指标的预警表现

```text
{_text(warning_summary)}
```

## 训练与推理开销

```text
{_text(runtime_summary)}
```
"""
    return report, {
        "逐预处理逐指标表示选择.csv": mode_summary,
        "相对原始表示优势交互.csv": interaction_summary,
        "表示优势交互明细.csv": interaction_detail,
        "翻转一项的全指标影响.csv": side_effect_summary,
        "被翻转指标预警结果.csv": warning_summary,
        "训练与推理开销.csv": runtime_summary,
        "原值变化量严格配对明细.csv": detail,
        "逐模型逐指标时距结果.csv": forecast,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成预处理与输出表示交互消融报告")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    args = parser.parse_args()
    panel = data.load_development_panel()
    stations = select_stations(panel, args.stations, args.all_stations)
    seeds = _parse_seeds(args.seeds)
    forecast, warning, runtime = build_tables(stations, seeds)
    detail = build_mode_detail(forecast)
    report, tables = build_report(forecast, warning, runtime, detail, seeds)
    output_dir = config.OUTPUT_DIR / "报告"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    path = output_dir / "预处理与输出表示交互消融报告.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
