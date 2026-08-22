#!/usr/bin/env python3
"""Report 4--72 hour teacher, student, and target-representation comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import select_tasks


PERSISTENCE_KEY = "persistence"
TEACHER_KEY = "delta_tabpfn_teacher"
VARIANT_LABELS = {
    PERSISTENCE_KEY: "持续性",
    TEACHER_KEY: "Delta-TabPFN教师",
    **config.STUDENT_FILE_LABELS,
}


def metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return {"valid_points": 0, "mae": np.nan, "rmse": np.nan, "nse": np.nan}
    prediction = np.asarray(pred, dtype=float)[valid]
    truth = np.asarray(true, dtype=float)[valid]
    error = prediction - truth
    denominator = float(np.square(truth - truth.mean()).sum())
    return {
        "valid_points": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "nse": np.nan
        if denominator <= np.finfo(float).eps
        else float(1.0 - np.square(error).sum() / denominator),
    }


def _append_rows(
    rows: list[dict[str, object]],
    *,
    station: str,
    target: str,
    variant: str,
    seed: int,
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> None:
    for horizon, hours in enumerate(config.HORIZON_HOURS):
        rows.append(
            {
                "station": station,
                "target": target,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "seed": seed,
                "horizon_hours": hours,
                **metrics(pred[:, horizon], true[:, horizon], mask[:, horizon]),
            }
        )


def task_rows(
    split: dict[str, np.ndarray],
    station: str,
    target: str,
    seeds: tuple[int, ...],
    *,
    allow_partial: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    true = np.asarray(split["y_abs"], dtype=float)
    mask = np.asarray(split["y_mask"], dtype=bool)
    persistence = np.repeat(np.asarray(split["current"], dtype=float), config.OUTPUT_STEPS, axis=1)
    _append_rows(
        rows,
        station=station,
        target=target,
        variant=PERSISTENCE_KEY,
        seed=0,
        pred=persistence,
        true=true,
        mask=mask,
    )

    teacher_path = io.teacher_cache_path("验证集", station, target)
    if not teacher_path.exists():
        if not allow_partial:
            raise FileNotFoundError(f"缺少验证集教师缓存: {teacher_path}")
    else:
        teacher, teacher_metadata = io.load_archive(teacher_path)
        if teacher_metadata.get("kind") != "validation":
            raise RuntimeError(f"教师缓存类型不正确: {teacher_path}")
        completed = np.asarray(teacher.get("completed"), dtype=bool)
        if completed.shape != (1, config.OUTPUT_STEPS) or not completed.all():
            if not allow_partial:
                raise RuntimeError(f"验证集教师缓存尚未完成18个时距: {teacher_path}")
        else:
            if not np.array_equal(teacher["target_start"], split["target_start"]):
                raise RuntimeError(f"验证集教师时间轴不一致: {teacher_path}")
            teacher_abs = data.to_absolute(
                teacher["pred_delta"], split["current"], "delta"
            )
            teacher_mask = mask & np.asarray(teacher["pred_mask"], dtype=bool)
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=TEACHER_KEY,
                seed=config.TEACHER_SEED,
                pred=teacher_abs,
                true=true,
                mask=teacher_mask,
            )

    for variant in config.STUDENT_KEYS:
        for seed in seeds:
            path = io.student_prediction_path("val", variant, seed, station, target)
            if not path.exists():
                if allow_partial:
                    continue
                raise FileNotFoundError(f"缺少学生预测: {path}")
            arrays, metadata = io.load_archive(path)
            if metadata.get("variant") != variant or int(metadata.get("seed", -1)) != seed:
                raise RuntimeError(f"学生预测身份不一致: {path}")
            if not np.array_equal(arrays["target_start"], split["target_start"]):
                raise RuntimeError(f"学生预测时间轴不一致: {path}")
            _append_rows(
                rows,
                station=station,
                target=target,
                variant=variant,
                seed=seed,
                pred=np.asarray(arrays["pred"], dtype=float),
                true=np.asarray(arrays["true"], dtype=float),
                mask=np.asarray(arrays["mask"], dtype=bool),
            )
    return rows


def paired_comparison(
    cells: pd.DataFrame, left: str, right: str, comparison: str
) -> pd.DataFrame:
    keys = ["station", "target", "seed", "horizon_hours"]
    left_frame = cells.loc[cells["variant"].eq(left), keys + ["rmse"]].rename(
        columns={"rmse": "left_rmse"}
    )
    right_frame = cells.loc[cells["variant"].eq(right), keys + ["rmse"]].rename(
        columns={"rmse": "right_rmse"}
    )
    paired = left_frame.merge(right_frame, on=keys, how="inner", validate="one_to_one")
    paired.insert(0, "comparison", comparison)
    paired.insert(1, "left_variant", left)
    paired.insert(2, "right_variant", right)
    paired["difference"] = paired["left_rmse"] - paired["right_rmse"]
    paired["relative_pct"] = np.where(
        paired["right_rmse"].ne(0),
        (paired["left_rmse"] / paired["right_rmse"] - 1.0) * 100.0,
        np.nan,
    )
    paired["winner"] = np.where(
        paired["difference"].lt(0), VARIANT_LABELS[left], VARIANT_LABELS[right]
    )
    return paired


def write_report(
    output_dir: Path,
    cells: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    allow_partial: bool,
) -> None:
    mean_by_horizon = (
        cells.groupby(["target", "variant", "variant_label", "horizon_hours"], as_index=False)
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), nse=("nse", "mean"))
        .sort_values(["target", "horizon_hours", "variant"])
    )
    comparison_summary = (
        comparisons.groupby(
            ["comparison", "left_variant", "right_variant", "target", "horizon_hours"],
            as_index=False,
        )
        .agg(
            left_rmse=("left_rmse", "mean"),
            right_rmse=("right_rmse", "mean"),
            relative_pct=("relative_pct", "mean"),
        )
    )
    lines = [
        "# TabPFN因果蒸馏4–72小时" + ("阶段性报告" if allow_partial else "验证报告"),
        "",
        *(
            ["- **阶段性结果：仅汇总当前已经完成的模型，不能作为最终论文结论。**", ""]
            if allow_partial
            else []
        ),
        "- 输入：本站过去24小时（6个4小时时间步）的9项水质观测。",
        "- 输出：未来4、8、…、72小时，共18个直接预测时距。",
        "- 教师：严格时间前向OOF的Delta-TabPFN；学生训练不读取验证标签。",
        "- 原值/变化量消融共享同一教师、网络容量、随机种子和损失权重。",
        "- 不跨指标平均原始RMSE；所有比较均在同一指标和时距内进行。",
        "",
        "## 分指标和时距结果",
        "",
        "```text",
        mean_by_horizon.to_string(index=False),
        "```",
        "",
        "## 配对消融",
        "",
        "```text",
        comparison_summary.to_string(index=False),
        "```",
        "",
    ]
    report_name = "阶段性报告.md" if allow_partial else "实验报告.md"
    prefix = "阶段性" if allow_partial else ""
    (output_dir / report_name).write_text("\n".join(lines), encoding="utf-8")
    mean_by_horizon.to_csv(
        output_dir / f"{prefix}分指标时距平均结果.csv", index=False, encoding="utf-8-sig"
    )
    comparison_summary.to_csv(
        output_dir / f"{prefix}消融汇总.csv", index=False, encoding="utf-8-sig"
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize completed stages without treating missing teacher/student files as final.",
    )
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

    rows = []
    for station in stations:
        for target in targets:
            split = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["val"]
            rows.extend(
                task_rows(
                    split,
                    station,
                    target,
                    seeds,
                    allow_partial=args.allow_partial,
                )
            )
    cells = pd.DataFrame(rows)
    comparisons = pd.concat(
        [
            paired_comparison(
                cells,
                config.SUPERVISED_ABSOLUTE_KEY,
                config.SUPERVISED_DELTA_KEY,
                "监督GRU：原值对变化量",
            ),
            paired_comparison(
                cells,
                config.DISTILLED_ABSOLUTE_KEY,
                config.DISTILLED_DELTA_KEY,
                "因果蒸馏GRU：原值对变化量",
            ),
            paired_comparison(
                cells,
                config.DISTILLED_ABSOLUTE_KEY,
                config.SUPERVISED_ABSOLUTE_KEY,
                "原值：因果蒸馏对无蒸馏",
            ),
            paired_comparison(
                cells,
                config.DISTILLED_DELTA_KEY,
                config.SUPERVISED_DELTA_KEY,
                "变化量：因果蒸馏对无蒸馏",
            ),
        ],
        ignore_index=True,
    )
    output_dir = config.output_dir_for_split("val")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "阶段性" if args.allow_partial else ""
    cells.to_csv(
        output_dir / f"{prefix}逐任务时距指标.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        output_dir / f"{prefix}逐任务配对消融.csv", index=False, encoding="utf-8-sig"
    )
    write_report(
        output_dir, cells, comparisons, allow_partial=args.allow_partial
    )
    console.done(output_dir)


if __name__ == "__main__":
    main()
