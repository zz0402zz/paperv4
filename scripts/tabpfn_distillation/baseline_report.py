#!/usr/bin/env python3
"""Compare same-protocol LSTM/XGBoost with the gated distilled GRU."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io
from scripts.tabpfn_distillation.inertia_gate import apply_persistence_gate, load_gate
from scripts.tabpfn_distillation import inertia_report
from scripts.tabpfn_distillation.protocol_baselines import (
    BASELINE_EXPERIMENT_ID,
    BASELINE_KEYS,
    BASELINE_LABELS,
    LSTM_KEY,
    XGBOOST_KEY,
    baseline_prediction_path,
)
from scripts.tabpfn_distillation.report import metrics
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import select_tasks


GATED_BASELINE_KEYS = {
    model_key: f"oof_gated_{model_key}" for model_key in BASELINE_KEYS
}
BASELINE_REPORT_LABELS = {
    **inertia_report.VARIANT_LABELS,
    **BASELINE_LABELS,
    **{
        GATED_BASELINE_KEYS[model_key]: f"OOF惯性门控{label}"
        for model_key, label in BASELINE_LABELS.items()
    },
}
PROPOSED_KEY = inertia_report.GATED_STUDENT_KEYS[
    config.DISTILLED_DELTA_KEY
]
FOCUS_KEYS = (
    inertia_report.PERSISTENCE_KEY,
    inertia_report.GATED_TEACHER_KEY,
    inertia_report.GATED_STUDENT_KEYS[config.SUPERVISED_DELTA_KEY],
    PROPOSED_KEY,
    LSTM_KEY,
    GATED_BASELINE_KEYS[LSTM_KEY],
    XGBOOST_KEY,
    GATED_BASELINE_KEYS[XGBOOST_KEY],
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
                "variant_label": BASELINE_REPORT_LABELS[variant],
                "seed": int(seed),
                "horizon_hours": hours,
                **metrics(
                    prediction[:, horizon], truth[:, horizon], mask[:, horizon]
                ),
            }
        )


def _scalar(arrays: dict[str, np.ndarray], key: str) -> float:
    value = np.asarray(arrays.get(key, np.nan))
    return float(value.item()) if value.size == 1 else np.nan


def task_rows(
    split: dict[str, np.ndarray],
    station: str,
    target: str,
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows, _ = inertia_report.task_rows(split, station, target, seeds)
    runtime_rows: list[dict[str, object]] = []
    gate, _ = load_gate(station, target)
    alpha = np.asarray(gate["alpha"], dtype=float)
    current = np.asarray(split["current"], dtype=float)
    for model_key in BASELINE_KEYS:
        for seed in seeds:
            path = baseline_prediction_path(model_key, seed, station, target)
            if not path.exists():
                raise FileNotFoundError(f"缺少同协议基线预测: {path}")
            arrays, metadata = io.load_archive(path)
            if (
                metadata.get("experiment") != BASELINE_EXPERIMENT_ID
                or metadata.get("kind") != "same_protocol_validation_prediction"
                or metadata.get("model") != model_key
                or int(metadata.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"基线预测身份不一致: {path}")
            if bool(metadata.get("validation_labels_used_for_fit", True)):
                raise RuntimeError(f"基线声明使用了验证标签: {path}")
            if bool(metadata.get("test_labels_used", True)):
                raise RuntimeError(f"基线声明使用了测试标签: {path}")
            if not np.array_equal(arrays["target_start"], split["target_start"]):
                raise RuntimeError(f"基线预测时间轴不一致: {path}")
            prediction = np.asarray(arrays["pred"], dtype=float)
            truth = np.asarray(arrays["true"], dtype=float)
            mask = np.asarray(arrays["mask"], dtype=bool)
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=model_key,
                seed=seed,
                prediction=prediction,
                truth=truth,
                mask=mask,
            )
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=GATED_BASELINE_KEYS[model_key],
                seed=seed,
                prediction=apply_persistence_gate(prediction, current, alpha),
                truth=truth,
                mask=mask,
            )
            runtime_rows.append(
                {
                    "station": station,
                    "target": target,
                    "variant": model_key,
                    "variant_label": BASELINE_REPORT_LABELS[model_key],
                    "seed": seed,
                    "training_seconds": _scalar(arrays, "training_seconds"),
                    "inference_seconds": _scalar(arrays, "inference_seconds"),
                    "parameter_count": int(_scalar(arrays, "parameter_count")),
                    "tree_count": int(_scalar(arrays, "tree_count")),
                }
            )
    return rows, runtime_rows


def paired_comparison(
    cells: pd.DataFrame, left: str, right: str, label: str
) -> pd.DataFrame:
    keys = ["station", "target", "seed", "horizon_hours"]
    left_frame = cells.loc[cells["variant"].eq(left), keys + ["rmse"]].rename(
        columns={"rmse": "left_rmse"}
    )
    right_frame = cells.loc[cells["variant"].eq(right), keys + ["rmse"]].rename(
        columns={"rmse": "right_rmse"}
    )
    paired = left_frame.merge(
        right_frame, on=keys, how="inner", validate="one_to_one"
    )
    paired.insert(0, "comparison", label)
    paired.insert(1, "left_variant", left)
    paired.insert(2, "right_variant", right)
    paired["difference"] = paired["left_rmse"] - paired["right_rmse"]
    paired["relative_pct"] = np.where(
        paired["right_rmse"].ne(0),
        (paired["left_rmse"] / paired["right_rmse"] - 1.0) * 100.0,
        np.nan,
    )
    paired["left_wins"] = paired["difference"].lt(0)
    return paired


def write_report(output_dir, cells, runtimes, comparisons) -> None:
    focus = cells.loc[cells["variant"].isin(FOCUS_KEYS)].copy()
    overall = (
        focus.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            macro_rmse=("rmse", "mean"),
            macro_mae=("mae", "mean"),
            macro_nse=("nse", "mean"),
            cells=("rmse", "size"),
        )
        .sort_values("macro_rmse")
    )
    by_horizon = (
        focus.groupby(
            ["target", "variant", "variant_label", "horizon_hours"],
            as_index=False,
        )
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), nse=("nse", "mean"))
        .sort_values(["target", "horizon_hours", "rmse"])
    )
    runtime_summary = (
        runtimes.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            training_seconds_mean=("training_seconds", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
            parameter_count=("parameter_count", "max"),
            tree_count=("tree_count", "max"),
        )
        .sort_values("variant")
    )
    comparison_summary = (
        comparisons.groupby(
            ["comparison", "left_variant", "right_variant"], as_index=False
        )
        .agg(
            left_rmse=("left_rmse", "mean"),
            right_rmse=("right_rmse", "mean"),
            relative_pct=("relative_pct", "mean"),
            left_wins=("left_wins", "sum"),
            total_cells=("left_wins", "size"),
        )
    )
    best = overall.iloc[0]
    lines = [
        "# 4–72小时同协议基线验证报告",
        "",
        "- 所有模型使用同一站点、过去24小时输入、18个直接预测时距和相同标签掩码。",
        "- LSTM与GRU共享隐层宽度、当前值分支、轮数、批量和学习率。",
        "- XGBoost对18个时距分别直接拟合，不使用验证集早停。",
        "- 门控版本共享由训练OOF拟合的惯性系数，不读取验证标签。",
        "",
        f"- 当前最佳：{best['variant_label']}，RMSE={float(best['macro_rmse']):.6f}。",
        "",
        "## 总体结果",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## 运行时间与复杂度",
        "",
        "```text",
        runtime_summary.to_string(index=False),
        "```",
        "",
        "## 配对比较",
        "",
        "```text",
        comparison_summary.to_string(index=False),
        "```",
        "",
        "## 分时距结果",
        "",
        "```text",
        by_horizon.to_string(index=False),
        "```",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "同协议基线报告.md").write_text("\n".join(lines), encoding="utf-8")
    focus.to_csv(output_dir / "同协议基线逐任务时距指标.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "同协议基线总体结果.csv", index=False, encoding="utf-8-sig")
    by_horizon.to_csv(output_dir / "同协议基线分时距结果.csv", index=False, encoding="utf-8-sig")
    runtimes.to_csv(output_dir / "同协议基线运行时间.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output_dir / "同协议基线配对比较.csv", index=False, encoding="utf-8-sig")


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
    runtime_rows: list[dict[str, object]] = []
    for station in stations:
        for target in targets:
            split = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["val"]
            task_metrics, task_runtimes = task_rows(split, station, target, seeds)
            rows.extend(task_metrics)
            runtime_rows.extend(task_runtimes)
    cells = pd.DataFrame(rows)
    runtimes = pd.DataFrame(runtime_rows)
    comparisons = pd.concat(
        [
            paired_comparison(
                cells,
                GATED_BASELINE_KEYS[LSTM_KEY],
                LSTM_KEY,
                "LSTM：OOF惯性门控对原模型",
            ),
            paired_comparison(
                cells,
                GATED_BASELINE_KEYS[XGBOOST_KEY],
                XGBOOST_KEY,
                "XGBoost：OOF惯性门控对原模型",
            ),
            paired_comparison(
                cells,
                PROPOSED_KEY,
                GATED_BASELINE_KEYS[LSTM_KEY],
                "提出模型对门控LSTM",
            ),
            paired_comparison(
                cells,
                PROPOSED_KEY,
                GATED_BASELINE_KEYS[XGBOOST_KEY],
                "提出模型对门控XGBoost",
            ),
        ],
        ignore_index=True,
    )
    output_dir = config.output_dir_for_split("val") / "同协议基线"
    write_report(output_dir, cells, runtimes, comparisons)
    console.done(output_dir)


if __name__ == "__main__":
    main()
