#!/usr/bin/env python3
"""Compare supervised and causal TabPFN-distilled XGBoost pipelines."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, inertia_report
from scripts.tabpfn_distillation.baseline_report import paired_comparison
from scripts.tabpfn_distillation.distilled_xgboost import (
    DISTILLED_XGBOOST_KEY,
    DISTILLED_XGBOOST_LABEL,
    load_self_gate as load_distilled_gate,
    load_validation_prediction as load_distilled_prediction,
)
from scripts.tabpfn_distillation.inertia_gate import apply_persistence_gate
from scripts.tabpfn_distillation.protocol_baselines import XGBOOST_KEY
from scripts.tabpfn_distillation.report import metrics
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import select_tasks
from scripts.tabpfn_distillation.xgboost_gate_attribution import (
    load_xgboost_self_gate,
)
from scripts.tabpfn_distillation.xgboost_gate_report import (
    _load_xgboost_validation,
)


SUPERVISED_XGBOOST_KEY = XGBOOST_KEY
GATED_SUPERVISED_XGBOOST_KEY = "xgboost_self_oof_gated_delta_xgboost"
GATED_DISTILLED_XGBOOST_KEY = "xgboost_self_oof_gated_causal_distilled_xgboost"
PROPOSED_GRU_KEY = inertia_report.GATED_STUDENT_KEYS[
    config.DISTILLED_DELTA_KEY
]
LABELS = {
    **inertia_report.VARIANT_LABELS,
    SUPERVISED_XGBOOST_KEY: "变化量监督XGBoost",
    GATED_SUPERVISED_XGBOOST_KEY: "自OOF门控变化量监督XGBoost",
    DISTILLED_XGBOOST_KEY: DISTILLED_XGBOOST_LABEL,
    GATED_DISTILLED_XGBOOST_KEY: "自OOF门控变化量因果蒸馏XGBoost",
}
FOCUS_KEYS = (
    inertia_report.PERSISTENCE_KEY,
    inertia_report.GATED_TEACHER_KEY,
    PROPOSED_GRU_KEY,
    SUPERVISED_XGBOOST_KEY,
    GATED_SUPERVISED_XGBOOST_KEY,
    DISTILLED_XGBOOST_KEY,
    GATED_DISTILLED_XGBOOST_KEY,
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


def task_rows(
    split: dict[str, np.ndarray],
    station: str,
    target: str,
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    all_rows, _ = inertia_report.task_rows(split, station, target, seeds)
    rows = [row for row in all_rows if row["variant"] in FOCUS_KEYS]
    coefficient_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    current = np.asarray(split["current"], dtype=float)
    for seed in seeds:
        supervised_prediction, truth, mask = _load_xgboost_validation(
            split, station, target, seed
        )
        supervised_gate, _ = load_xgboost_self_gate(seed, station, target)
        supervised_alpha = np.asarray(supervised_gate["alpha"], dtype=float)
        distilled_arrays, distilled_metadata = load_distilled_prediction(
            seed, station, target
        )
        if not np.array_equal(
            distilled_arrays["target_start"], split["target_start"]
        ):
            raise RuntimeError(
                f"蒸馏XGBoost验证预测时间轴不一致: "
                f"{station}/{target}/seed={seed}"
            )
        distilled_prediction = np.asarray(distilled_arrays["pred"], dtype=float)
        distilled_truth = np.asarray(distilled_arrays["true"], dtype=float)
        distilled_mask = np.asarray(distilled_arrays["mask"], dtype=bool)
        distilled_gate, _ = load_distilled_gate(seed, station, target)
        distilled_alpha = np.asarray(distilled_gate["alpha"], dtype=float)
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=SUPERVISED_XGBOOST_KEY,
            seed=seed,
            prediction=supervised_prediction,
            truth=truth,
            mask=mask,
        )
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=GATED_SUPERVISED_XGBOOST_KEY,
            seed=seed,
            prediction=apply_persistence_gate(
                supervised_prediction, current, supervised_alpha
            ),
            truth=truth,
            mask=mask,
        )
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=DISTILLED_XGBOOST_KEY,
            seed=seed,
            prediction=distilled_prediction,
            truth=distilled_truth,
            mask=distilled_mask,
        )
        _append_rows(
            rows,
            station=station,
            target=target,
            variant=GATED_DISTILLED_XGBOOST_KEY,
            seed=seed,
            prediction=apply_persistence_gate(
                distilled_prediction, current, distilled_alpha
            ),
            truth=distilled_truth,
            mask=distilled_mask,
        )
        runtime_rows.append(
            {
                "station": station,
                "target": target,
                "seed": seed,
                "variant": DISTILLED_XGBOOST_KEY,
                "variant_label": LABELS[DISTILLED_XGBOOST_KEY],
                "training_seconds": float(
                    np.asarray(distilled_arrays["training_seconds"]).sum()
                ),
                "inference_seconds": float(
                    np.asarray(distilled_arrays["inference_seconds"]).sum()
                ),
                "tree_count": int(
                    np.asarray(distilled_arrays["tree_count"]).sum()
                ),
                "teacher_weight": float(distilled_metadata["teacher_weight"]),
            }
        )
        for horizon, hours in enumerate(config.HORIZON_HOURS):
            coefficient_rows.append(
                {
                    "station": station,
                    "target": target,
                    "seed": seed,
                    "horizon_hours": hours,
                    "supervised_xgboost_alpha": supervised_alpha[horizon],
                    "distilled_xgboost_alpha": distilled_alpha[horizon],
                    "alpha_difference_distilled_minus_supervised": (
                        distilled_alpha[horizon] - supervised_alpha[horizon]
                    ),
                    "distilled_oof_valid_points": int(
                        distilled_gate["valid_count"][horizon]
                    ),
                    "distilled_oof_rmse_persistence": distilled_gate[
                        "oof_rmse_persistence"
                    ][horizon],
                    "distilled_oof_rmse_raw": distilled_gate[
                        "oof_rmse_tabpfn"
                    ][horizon],
                    "distilled_oof_rmse_self_gated": distilled_gate[
                        "oof_rmse_gated_tabpfn"
                    ][horizon],
                }
            )
    return rows, coefficient_rows, runtime_rows


def _summary(comparisons: pd.DataFrame) -> pd.DataFrame:
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
    runtimes: pd.DataFrame,
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
        coefficients.groupby(["target", "horizon_hours"], as_index=False)
        .agg(
            supervised_xgboost_alpha_mean=("supervised_xgboost_alpha", "mean"),
            distilled_xgboost_alpha_mean=("distilled_xgboost_alpha", "mean"),
            distilled_xgboost_alpha_std=("distilled_xgboost_alpha", "std"),
            distilled_oof_valid_points=("distilled_oof_valid_points", "mean"),
            distilled_oof_rmse_persistence=(
                "distilled_oof_rmse_persistence",
                "mean",
            ),
            distilled_oof_rmse_raw=("distilled_oof_rmse_raw", "mean"),
            distilled_oof_rmse_self_gated=(
                "distilled_oof_rmse_self_gated",
                "mean",
            ),
        )
        .sort_values(["target", "horizon_hours"])
    )
    runtime_summary = runtimes.groupby(
        ["variant", "variant_label", "teacher_weight"], as_index=False
    ).agg(
        training_seconds_mean=("training_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
        tree_count=("tree_count", "max"),
    )
    comparison_summary = _summary(comparisons)
    final_comparison = comparison_summary.loc[
        comparison_summary["comparison"].eq(
            "自门控蒸馏XGBoost对自门控监督XGBoost"
        )
    ].iloc[0]
    raw_comparison = comparison_summary.loc[
        comparison_summary["comparison"].eq(
            "原始蒸馏XGBoost对原始监督XGBoost"
        )
    ].iloc[0]
    if float(final_comparison["left_rmse"]) < float(
        final_comparison["right_rmse"]
    ):
        verdict = (
            "在两类XGBoost都使用自身OOF门控后，因果蒸馏仍然更好，"
            "当前任务支持TabPFN软标签对最强学生有额外贡献。"
        )
    else:
        verdict = (
            "因果蒸馏在公平自门控后没有超过监督XGBoost，当前任务"
            "不支持把TabPFN蒸馏作为主创新。"
        )
    best = overall.iloc[0]
    lines = [
        "# 因果TabPFN蒸馏XGBoost验证报告",
        "",
        "- 蒸馏目标为真实变化量MSE + 0.5 × 严格因果TabPFN OOF软标签MSE。",
        "- XGBoost使用代数等价的加权合成标签，不使用验证集调参或早停。",
        "- 监督版和蒸馏版都同时报告原始结果与各自OOF门控结果。",
        "",
        f"- 主要结论：{verdict}",
        f"- 原始蒸馏配对相对变化：{float(raw_comparison['relative_pct']):.3f}%。",
        f"- 公平自门控后配对相对变化：{float(final_comparison['relative_pct']):.3f}%。",
        f"- 当前最优：{best['variant_label']}，RMSE={float(best['macro_rmse']):.6f}。",
        "",
        "## 总体结果",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## 蒸馏与门控配对归因",
        "",
        "```text",
        comparison_summary.to_string(index=False),
        "```",
        "",
        "## 自门控系数与蒸馏OOF表现",
        "",
        "```text",
        coefficient_summary.to_string(index=False),
        "```",
        "",
        "## 训练与推理时间",
        "",
        "```text",
        runtime_summary.to_string(index=False),
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
    (output_dir / "因果蒸馏XGBoost报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    cells.to_csv(
        output_dir / "蒸馏XGBoost逐任务时距指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "蒸馏XGBoost总体结果.csv", index=False, encoding="utf-8-sig"
    )
    by_horizon.to_csv(
        output_dir / "蒸馏XGBoost分时距结果.csv", index=False, encoding="utf-8-sig"
    )
    coefficients.to_csv(
        output_dir / "蒸馏XGBoost门控系数逐种子.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        output_dir / "蒸馏XGBoost配对归因.csv", index=False, encoding="utf-8-sig"
    )
    runtimes.to_csv(
        output_dir / "蒸馏XGBoost运行时间.csv", index=False, encoding="utf-8-sig"
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
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for target in targets:
            split = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["val"]
            task_metrics, task_coefficients, task_runtimes = task_rows(
                split, station, target, seeds
            )
            rows.extend(task_metrics)
            coefficient_rows.extend(task_coefficients)
            runtime_rows.extend(task_runtimes)
    cells = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)
    runtimes = pd.DataFrame(runtime_rows)
    comparisons = pd.concat(
        [
            paired_comparison(
                cells,
                DISTILLED_XGBOOST_KEY,
                SUPERVISED_XGBOOST_KEY,
                "原始蒸馏XGBoost对原始监督XGBoost",
            ),
            paired_comparison(
                cells,
                GATED_DISTILLED_XGBOOST_KEY,
                GATED_SUPERVISED_XGBOOST_KEY,
                "自门控蒸馏XGBoost对自门控监督XGBoost",
            ),
            paired_comparison(
                cells,
                GATED_DISTILLED_XGBOOST_KEY,
                DISTILLED_XGBOOST_KEY,
                "蒸馏XGBoost自OOF门控对其原始模型",
            ),
            paired_comparison(
                cells,
                GATED_SUPERVISED_XGBOOST_KEY,
                SUPERVISED_XGBOOST_KEY,
                "监督XGBoost自OOF门控对其原始模型",
            ),
            paired_comparison(
                cells,
                GATED_DISTILLED_XGBOOST_KEY,
                PROPOSED_GRU_KEY,
                "自门控蒸馏XGBoost对门控蒸馏GRU",
            ),
        ],
        ignore_index=True,
    )
    output_dir = config.output_dir_for_split("val") / "蒸馏XGBoost实验"
    write_report(output_dir, cells, coefficients, runtimes, comparisons)
    console.done(output_dir)


if __name__ == "__main__":
    main()
