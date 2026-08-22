#!/usr/bin/env python3
"""Report OOF-fitted persistence gating for TabPFN and GRU forecasts."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io
from scripts.tabpfn_distillation.inertia_gate import (
    apply_persistence_gate,
    load_gate,
)
from scripts.tabpfn_distillation.report import metrics
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import select_tasks


PERSISTENCE_KEY = "persistence"
TEACHER_KEY = "delta_tabpfn_teacher"
GATED_TEACHER_KEY = "oof_gated_delta_tabpfn_teacher"
GATED_STUDENT_KEYS = {
    variant: f"oof_gated_{variant}" for variant in config.STUDENT_KEYS
}
VARIANT_LABELS = {
    PERSISTENCE_KEY: "持续性",
    TEACHER_KEY: "Delta-TabPFN教师",
    GATED_TEACHER_KEY: "OOF惯性门控TabPFN",
    **config.STUDENT_FILE_LABELS,
    **{
        GATED_STUDENT_KEYS[variant]: f"OOF惯性门控{label}"
        for variant, label in config.STUDENT_FILE_LABELS.items()
    },
}


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
                "variant_label": VARIANT_LABELS[variant],
                "seed": int(seed),
                "horizon_hours": hours,
                **metrics(
                    prediction[:, horizon], truth[:, horizon], mask[:, horizon]
                ),
            }
        )


def _load_validation_teacher(
    split: dict[str, np.ndarray], station: str, target: str
) -> tuple[np.ndarray, np.ndarray]:
    path = io.teacher_cache_path("验证集", station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少验证集教师缓存: {path}")
    arrays, metadata = io.load_archive(path)
    if metadata.get("kind") != "validation":
        raise RuntimeError(f"教师缓存类型不正确: {path}")
    completed = np.asarray(arrays.get("completed"), dtype=bool)
    if completed.shape != (1, config.OUTPUT_STEPS) or not completed.all():
        raise RuntimeError(f"验证集教师缓存尚未完成18个时距: {path}")
    if not np.array_equal(arrays["target_start"], split["target_start"]):
        raise RuntimeError(f"验证集教师时间轴不一致: {path}")
    prediction = data.to_absolute(
        np.asarray(arrays["pred_delta"], dtype=float),
        split["current"],
        "delta",
    )
    return prediction, np.asarray(arrays["pred_mask"], dtype=bool)


def task_rows(
    split: dict[str, np.ndarray],
    station: str,
    target: str,
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    gate, gate_metadata = load_gate(station, target)
    if bool(gate_metadata.get("uses_validation_labels", True)):
        raise RuntimeError("门控参数声明使用了验证标签，拒绝汇总。")
    alpha = np.asarray(gate["alpha"], dtype=float)
    truth = np.asarray(split["y_abs"], dtype=float)
    mask = np.asarray(split["y_mask"], dtype=bool)
    current = np.asarray(split["current"], dtype=float)
    persistence = np.repeat(current, config.OUTPUT_STEPS, axis=1)
    _append_rows(
        rows,
        station=station,
        target=target,
        variant=PERSISTENCE_KEY,
        seed=0,
        prediction=persistence,
        truth=truth,
        mask=mask,
    )

    teacher, teacher_mask = _load_validation_teacher(split, station, target)
    _append_rows(
        rows,
        station=station,
        target=target,
        variant=TEACHER_KEY,
        seed=config.TEACHER_SEED,
        prediction=teacher,
        truth=truth,
        mask=mask & teacher_mask,
    )
    _append_rows(
        rows,
        station=station,
        target=target,
        variant=GATED_TEACHER_KEY,
        seed=config.TEACHER_SEED,
        prediction=apply_persistence_gate(teacher, current, alpha),
        truth=truth,
        mask=mask & teacher_mask,
    )

    for horizon, hours in enumerate(config.HORIZON_HOURS):
        gate_rows.append(
            {
                "station": station,
                "target": target,
                "horizon_hours": hours,
                "alpha": alpha[horizon],
                "oof_valid_points": int(gate["valid_count"][horizon]),
                "oof_rmse_persistence": gate["oof_rmse_persistence"][horizon],
                "oof_rmse_tabpfn": gate["oof_rmse_tabpfn"][horizon],
                "oof_rmse_gated_tabpfn": gate["oof_rmse_gated_tabpfn"][horizon],
            }
        )

    for variant in config.STUDENT_KEYS:
        for seed in seeds:
            path = io.student_prediction_path("val", variant, seed, station, target)
            if not path.exists():
                raise FileNotFoundError(f"缺少学生预测: {path}")
            arrays, metadata = io.load_archive(path)
            if metadata.get("variant") != variant or int(metadata.get("seed", -1)) != seed:
                raise RuntimeError(f"学生预测身份不一致: {path}")
            if not np.array_equal(arrays["target_start"], split["target_start"]):
                raise RuntimeError(f"学生预测时间轴不一致: {path}")
            prediction = np.asarray(arrays["pred"], dtype=float)
            prediction_truth = np.asarray(arrays["true"], dtype=float)
            prediction_mask = np.asarray(arrays["mask"], dtype=bool)
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=variant,
                seed=seed,
                prediction=prediction,
                truth=prediction_truth,
                mask=prediction_mask,
            )
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=GATED_STUDENT_KEYS[variant],
                seed=seed,
                prediction=apply_persistence_gate(prediction, current, alpha),
                truth=prediction_truth,
                mask=prediction_mask,
            )
    return rows, gate_rows


def paired_improvements(cells: pd.DataFrame) -> pd.DataFrame:
    pairs = [(TEACHER_KEY, GATED_TEACHER_KEY)] + [
        (variant, GATED_STUDENT_KEYS[variant]) for variant in config.STUDENT_KEYS
    ]
    frames = []
    keys = ["station", "target", "seed", "horizon_hours"]
    for source, gated in pairs:
        source_frame = cells.loc[
            cells["variant"].eq(source), keys + ["rmse"]
        ].rename(columns={"rmse": "source_rmse"})
        gated_frame = cells.loc[
            cells["variant"].eq(gated), keys + ["rmse"]
        ].rename(columns={"rmse": "gated_rmse"})
        paired = source_frame.merge(
            gated_frame, on=keys, how="inner", validate="one_to_one"
        )
        paired.insert(0, "source_variant", source)
        paired.insert(1, "gated_variant", gated)
        paired["difference"] = paired["gated_rmse"] - paired["source_rmse"]
        paired["relative_pct"] = np.where(
            paired["source_rmse"].ne(0),
            (paired["gated_rmse"] / paired["source_rmse"] - 1.0) * 100.0,
            np.nan,
        )
        paired["gate_improved"] = paired["difference"].lt(0)
        frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def write_report(
    output_dir,
    cells: pd.DataFrame,
    gate_rows: pd.DataFrame,
    paired: pd.DataFrame,
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
    improvement = (
        paired.groupby(["source_variant", "gated_variant"], as_index=False)
        .agg(
            source_rmse=("source_rmse", "mean"),
            gated_rmse=("gated_rmse", "mean"),
            relative_pct=("relative_pct", "mean"),
            improved_cells=("gate_improved", "sum"),
            total_cells=("gate_improved", "size"),
        )
        .sort_values("gated_rmse")
    )
    persistence_rmse = float(
        overall.loc[overall["variant"].eq(PERSISTENCE_KEY), "macro_rmse"].iloc[0]
    )
    best = overall.iloc[0]
    lines = [
        "# OOF惯性门控TabPFN因果蒸馏验证报告",
        "",
        "- 门控系数只使用训练集严格因果OOF预测和训练标签拟合。",
        "- 验证标签只用于评价，没有进入门控系数。",
        "- 公式：预测 = 当前值 + alpha_h * (原预测 - 当前值)，alpha_h在[0,1]内。",
        "- 相同门控同时应用到教师、监督GRU和蒸馏GRU，便于识别蒸馏的独立贡献。",
        "",
        f"- 持续性宏平均RMSE：{persistence_rmse:.6f}。",
        f"- 当前最佳模型：{best['variant_label']}，RMSE={float(best['macro_rmse']):.6f}。",
        "",
        "## 总体结果",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## OOF门控系数",
        "",
        "```text",
        gate_rows.to_string(index=False),
        "```",
        "",
        "## 门控前后配对改进",
        "",
        "```text",
        improvement.to_string(index=False),
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
    (output_dir / "惯性门控报告.md").write_text("\n".join(lines), encoding="utf-8")
    cells.to_csv(output_dir / "惯性门控逐任务时距指标.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "惯性门控总体结果.csv", index=False, encoding="utf-8-sig")
    by_horizon.to_csv(output_dir / "惯性门控分时距结果.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output_dir / "惯性门控配对改进.csv", index=False, encoding="utf-8-sig")
    gate_rows.to_csv(output_dir / "惯性门控系数.csv", index=False, encoding="utf-8-sig")


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
    gate_rows: list[dict[str, object]] = []
    for station in stations:
        for target in targets:
            split = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["val"]
            task_metrics, task_gate = task_rows(split, station, target, seeds)
            rows.extend(task_metrics)
            gate_rows.extend(task_gate)
    cells = pd.DataFrame(rows)
    gates = pd.DataFrame(gate_rows)
    paired = paired_improvements(cells)
    output_dir = config.output_dir_for_split("val") / "惯性门控"
    write_report(output_dir, cells, gates, paired)
    console.done(output_dir)


if __name__ == "__main__":
    main()
