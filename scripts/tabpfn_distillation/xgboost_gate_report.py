#!/usr/bin/env python3
"""Report whether XGBoost gains come from a TabPFN gate or its own OOF gate."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, inertia_report, io
from scripts.tabpfn_distillation.baseline_report import paired_comparison
from scripts.tabpfn_distillation.inertia_gate import apply_persistence_gate, load_gate
from scripts.tabpfn_distillation.protocol_baselines import (
    BASELINE_EXPERIMENT_ID,
    XGBOOST_KEY,
    baseline_prediction_path,
)
from scripts.tabpfn_distillation.report import metrics
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import select_tasks
from scripts.tabpfn_distillation.xgboost_gate_attribution import (
    load_xgboost_self_gate,
)


RAW_XGBOOST_KEY = XGBOOST_KEY
TABPFN_GATE_XGBOOST_KEY = "tabpfn_oof_gated_delta_xgboost"
SELF_GATE_XGBOOST_KEY = "xgboost_self_oof_gated_delta_xgboost"
PROPOSED_KEY = inertia_report.GATED_STUDENT_KEYS[config.DISTILLED_DELTA_KEY]
LABELS = {
    **inertia_report.VARIANT_LABELS,
    RAW_XGBOOST_KEY: "变化量XGBoost",
    TABPFN_GATE_XGBOOST_KEY: "TabPFN-OOF门控XGBoost",
    SELF_GATE_XGBOOST_KEY: "XGBoost自OOF门控XGBoost",
}
FOCUS_KEYS = (
    inertia_report.PERSISTENCE_KEY,
    inertia_report.GATED_TEACHER_KEY,
    PROPOSED_KEY,
    RAW_XGBOOST_KEY,
    TABPFN_GATE_XGBOOST_KEY,
    SELF_GATE_XGBOOST_KEY,
)


def _append_rows(
    rows: list[dict[str, object]],
    *,
    station: str,
    target: str,
    variant: str,
    seed: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> None:
    for horizon, hours in enumerate(config.HORIZON_HOURS):
        rows.append(
            {
                "station": station,
                "target": target,
                "variant": variant,
                "variant_label": LABELS[variant],
                "seed": int(seed),
                "horizon_hours": hours,
                **metrics(
                    prediction[:, horizon], truth[:, horizon], mask[:, horizon]
                ),
            }
        )


def _load_xgboost_validation(
    split: dict[str, np.ndarray], station: str, target: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = baseline_prediction_path(XGBOOST_KEY, seed, station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少XGBoost验证预测: {path}")
    arrays, metadata = io.load_archive(path)
    if (
        metadata.get("experiment") != BASELINE_EXPERIMENT_ID
        or metadata.get("kind") != "same_protocol_validation_prediction"
        or metadata.get("model") != XGBOOST_KEY
        or int(metadata.get("seed", -1)) != seed
        or metadata.get("station") != station
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"XGBoost验证预测身份不一致: {path}")
    if bool(metadata.get("validation_labels_used_for_fit", True)):
        raise RuntimeError(f"XGBoost声明使用了验证标签: {path}")
    if bool(metadata.get("test_labels_used", True)):
        raise RuntimeError(f"XGBoost声明使用了测试标签: {path}")
    if not np.array_equal(arrays["target_start"], split["target_start"]):
        raise RuntimeError(f"XGBoost验证预测时间轴不一致: {path}")
    prediction = np.asarray(arrays["pred"], dtype=float)
    truth = np.asarray(arrays["true"], dtype=float)
    mask = np.asarray(arrays["mask"], dtype=bool)
    expected_shape = (len(split["target_start"]), config.OUTPUT_STEPS)
    if prediction.shape != expected_shape:
        raise RuntimeError(f"XGBoost验证预测形状不正确: {path}")
    return prediction, truth, mask


def task_rows(
    split: dict[str, np.ndarray],
    station: str,
    target: str,
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows, _ = inertia_report.task_rows(split, station, target, seeds)
    rows = [row for row in all_rows if row["variant"] in FOCUS_KEYS]
    coefficient_rows: list[dict[str, object]] = []
    tabpfn_gate, tabpfn_metadata = load_gate(station, target)
    if bool(tabpfn_metadata.get("uses_validation_labels", True)):
        raise RuntimeError("TabPFN OOF门控声明使用了验证标签。")
    tabpfn_alpha = np.asarray(tabpfn_gate["alpha"], dtype=float)
    current = np.asarray(split["current"], dtype=float)

    for seed in seeds:
        prediction, truth, mask = _load_xgboost_validation(
            split, station, target, seed
        )
        self_gate, _ = load_xgboost_self_gate(seed, station, target)
        self_alpha = np.asarray(self_gate["alpha"], dtype=float)
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=RAW_XGBOOST_KEY,
            seed=seed,
            prediction=prediction,
            truth=truth,
            mask=mask,
        )
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=TABPFN_GATE_XGBOOST_KEY,
            seed=seed,
            prediction=apply_persistence_gate(
                prediction, current, tabpfn_alpha
            ),
            truth=truth,
            mask=mask,
        )
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=SELF_GATE_XGBOOST_KEY,
            seed=seed,
            prediction=apply_persistence_gate(prediction, current, self_alpha),
            truth=truth,
            mask=mask,
        )
        for horizon, hours in enumerate(config.HORIZON_HOURS):
            coefficient_rows.append(
                {
                    "station": station,
                    "target": target,
                    "seed": seed,
                    "horizon_hours": hours,
                    "tabpfn_oof_alpha": tabpfn_alpha[horizon],
                    "xgboost_self_oof_alpha": self_alpha[horizon],
                    "alpha_difference_xgboost_minus_tabpfn": (
                        self_alpha[horizon] - tabpfn_alpha[horizon]
                    ),
                    "xgboost_oof_valid_points": int(
                        self_gate["valid_count"][horizon]
                    ),
                    "xgboost_oof_rmse_persistence": self_gate[
                        "oof_rmse_persistence"
                    ][horizon],
                    "xgboost_oof_rmse_raw": self_gate["oof_rmse_tabpfn"][
                        horizon
                    ],
                    "xgboost_oof_rmse_self_gated": self_gate[
                        "oof_rmse_gated_tabpfn"
                    ][horizon],
                }
            )
    return rows, coefficient_rows


def _comparison_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    return comparisons.groupby(
        ["comparison", "left_variant", "right_variant"], as_index=False
    ).agg(
        left_rmse=("left_rmse", "mean"),
        right_rmse=("right_rmse", "mean"),
        relative_pct=("relative_pct", "mean"),
        left_wins=("left_wins", "sum"),
        total_cells=("left_wins", "size"),
    )


def write_report(
    output_dir,
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    overall = (
        cells.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            macro_rmse=("rmse", "mean"),
            macro_mae=("mae", "mean"),
            macro_nse=("nse", "mean"),
            cells=("rmse", "size"),
        )
        .sort_values("macro_rmse")
    )
    by_horizon = (
        cells.groupby(
            ["target", "variant", "variant_label", "horizon_hours"],
            as_index=False,
        )
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), nse=("nse", "mean"))
        .sort_values(["target", "horizon_hours", "rmse"])
    )
    coefficient_summary = (
        coefficients.groupby(
            ["target", "horizon_hours", "tabpfn_oof_alpha"], as_index=False
        )
        .agg(
            xgboost_self_oof_alpha_mean=("xgboost_self_oof_alpha", "mean"),
            xgboost_self_oof_alpha_std=("xgboost_self_oof_alpha", "std"),
            xgboost_oof_valid_points=("xgboost_oof_valid_points", "mean"),
            xgboost_oof_rmse_persistence=(
                "xgboost_oof_rmse_persistence",
                "mean",
            ),
            xgboost_oof_rmse_raw=("xgboost_oof_rmse_raw", "mean"),
            xgboost_oof_rmse_self_gated=(
                "xgboost_oof_rmse_self_gated",
                "mean",
            ),
        )
        .sort_values(["target", "horizon_hours"])
    )
    comparison_summary = _comparison_summary(comparisons)
    primary = comparison_summary.loc[
        comparison_summary["comparison"].eq(
            "TabPFN-OOF门控对XGBoost自OOF门控"
        )
    ].iloc[0]
    if float(primary["left_rmse"]) < float(primary["right_rmse"]):
        verdict = (
            "TabPFN-OOF门控优于XGBoost自OOF门控，当前任务支持"
            "TabPFN提供了超过通用自校准的可迁移信息，但仍需"
            "跨指标与跨站验证。"
        )
    else:
        verdict = (
            "XGBoost自OOF门控不差于TabPFN-OOF门控，门控增益暂不能"
            "归因于TabPFN知识，需要调整创新点表述。"
        )
    best = overall.iloc[0]
    lines = [
        "# XGBoost OOF门控归因验证报告",
        "",
        "- 三个XGBoost版本共享完全相同的验证集预测，只改变后处理门控。",
        "- TabPFN门控仅使用TabPFN训练OOF；XGBoost自门控仅使用同种子XGBoost训练OOF。",
        "- 两类门控均为逐时距受限过原点最小二乘，都不读取验证或测试标签。",
        "",
        f"- 归因结论：{verdict}",
        f"- 当前最优：{best['variant_label']}，RMSE={float(best['macro_rmse']):.6f}。",
        "",
        "## 总体结果",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## 门控归因配对比较",
        "",
        "```text",
        comparison_summary.to_string(index=False),
        "```",
        "",
        "## 门控系数与XGBoost OOF表现",
        "",
        "```text",
        coefficient_summary.to_string(index=False),
        "```",
        "",
        "## 分时距验证结果",
        "",
        "```text",
        by_horizon.to_string(index=False),
        "```",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "XGBoost门控归因报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    cells.to_csv(
        output_dir / "门控归因逐任务时距指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "门控归因总体结果.csv", index=False, encoding="utf-8-sig"
    )
    by_horizon.to_csv(
        output_dir / "门控归因分时距结果.csv", index=False, encoding="utf-8-sig"
    )
    coefficients.to_csv(
        output_dir / "门控系数逐种子对照.csv", index=False, encoding="utf-8-sig"
    )
    coefficient_summary.to_csv(
        output_dir / "门控系数平均对照.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        output_dir / "门控归因配对比较.csv", index=False, encoding="utf-8-sig"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, config.STUDENT_SEEDS)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = data.load_v2_panel()
    try:
        stations, targets = select_tasks(
            panel, args.stations, args.targets, args.all_stations, args.all_targets
        )
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for station in stations:
        for target in targets:
            split = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["val"]
            task_metrics, task_coefficients = task_rows(
                split, station, target, seeds
            )
            rows.extend(task_metrics)
            coefficient_rows.extend(task_coefficients)
    cells = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)
    comparisons = pd.concat(
        [
            paired_comparison(
                cells,
                TABPFN_GATE_XGBOOST_KEY,
                SELF_GATE_XGBOOST_KEY,
                "TabPFN-OOF门控对XGBoost自OOF门控",
            ),
            paired_comparison(
                cells,
                TABPFN_GATE_XGBOOST_KEY,
                RAW_XGBOOST_KEY,
                "TabPFN-OOF门控对原始XGBoost",
            ),
            paired_comparison(
                cells,
                SELF_GATE_XGBOOST_KEY,
                RAW_XGBOOST_KEY,
                "XGBoost自OOF门控对原始XGBoost",
            ),
            paired_comparison(
                cells,
                PROPOSED_KEY,
                TABPFN_GATE_XGBOOST_KEY,
                "门控蒸馏GRU对TabPFN门控XGBoost",
            ),
            paired_comparison(
                cells,
                PROPOSED_KEY,
                SELF_GATE_XGBOOST_KEY,
                "门控蒸馏GRU对XGBoost自门控",
            ),
        ],
        ignore_index=True,
    )
    output_dir = config.output_dir_for_split("val") / "门控归因实验"
    write_report(output_dir, cells, coefficients, comparisons)
    console.done(output_dir)


if __name__ == "__main__":
    main()
