#!/usr/bin/env python3
"""Fit an OOF-only persistence gate for long-horizon TabPFN changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io
from scripts.tabpfn_distillation.teacher import select_tasks


GATE_EXPERIMENT_ID = "tabpfn_oof_persistence_gate_4_72h_v1"
GATE_METHOD = "per_horizon_clipped_ols_through_origin"
GATE_LOWER = 0.0
GATE_UPPER = 1.0


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def gate_cache_path(station: str, target: str) -> Path:
    filename = "__".join(
        (io.safe_filename(station), io.safe_filename(target), "OOF惯性门控")
    )
    return config.OUTPUT_DIR / "惯性门控" / "门控参数" / f"{filename}.npz"


def _masked_rmse_by_horizon(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(truth)
    )
    result = np.full(prediction.shape[1], np.nan, dtype=float)
    for horizon in range(prediction.shape[1]):
        horizon_valid = valid[:, horizon]
        if horizon_valid.any():
            error = prediction[horizon_valid, horizon] - truth[horizon_valid, horizon]
            result[horizon] = float(np.sqrt(np.square(error).mean()))
    return result


def fit_persistence_gate(
    teacher_delta: np.ndarray,
    true_delta: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit alpha in current + alpha * predicted_delta using OOF rows only."""

    teacher_delta = np.asarray(teacher_delta, dtype=float)
    true_delta = np.asarray(true_delta, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if teacher_delta.shape != true_delta.shape or teacher_delta.shape != mask.shape:
        raise ValueError("教师变化量、真实变化量和掩码形状必须一致。")
    if teacher_delta.ndim != 2 or teacher_delta.shape[1] != config.OUTPUT_STEPS:
        raise ValueError("惯性门控必须接收18个直接预测时距。")

    valid = mask & np.isfinite(teacher_delta) & np.isfinite(true_delta)
    numerator = np.zeros(config.OUTPUT_STEPS, dtype=float)
    denominator = np.zeros(config.OUTPUT_STEPS, dtype=float)
    valid_count = valid.sum(axis=0).astype(np.int64)
    for horizon in range(config.OUTPUT_STEPS):
        horizon_valid = valid[:, horizon]
        if not horizon_valid.any():
            continue
        prediction = teacher_delta[horizon_valid, horizon]
        truth = true_delta[horizon_valid, horizon]
        numerator[horizon] = float(np.dot(prediction, truth))
        denominator[horizon] = float(np.dot(prediction, prediction))

    alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )
    alpha = np.clip(alpha, GATE_LOWER, GATE_UPPER)
    gated_delta = teacher_delta * alpha.reshape(1, -1)
    persistence_delta = np.zeros_like(true_delta, dtype=float)
    return {
        "alpha": alpha,
        "numerator": numerator,
        "denominator": denominator,
        "valid_count": valid_count,
        "oof_rmse_persistence": _masked_rmse_by_horizon(
            persistence_delta, true_delta, valid
        ),
        "oof_rmse_tabpfn": _masked_rmse_by_horizon(
            teacher_delta, true_delta, valid
        ),
        "oof_rmse_gated_tabpfn": _masked_rmse_by_horizon(
            gated_delta, true_delta, valid
        ),
    }


def apply_persistence_gate(
    prediction_absolute: np.ndarray,
    current: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Shrink an absolute forecast toward persistence at each horizon."""

    prediction_absolute = np.asarray(prediction_absolute, dtype=float)
    current = np.asarray(current, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if prediction_absolute.ndim != 2 or prediction_absolute.shape[1] != config.OUTPUT_STEPS:
        raise ValueError("门控预测必须是 [样本, 18时距] 矩阵。")
    if current.shape != (len(prediction_absolute), 1):
        raise ValueError("当前值必须是 [样本, 1] 矩阵。")
    if alpha.shape != (config.OUTPUT_STEPS,):
        raise ValueError("门控系数必须包含18个时距。")
    if not np.isfinite(alpha).all() or (alpha < GATE_LOWER).any() or (alpha > GATE_UPPER).any():
        raise ValueError("门控系数必须是[0, 1]内的有限值。")
    return current + alpha.reshape(1, -1) * (prediction_absolute - current)


def _load_oof_teacher(
    train: dict[str, np.ndarray], station: str, target: str
) -> tuple[dict[str, np.ndarray], str]:
    path = io.teacher_cache_path("训练OOF", station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少因果OOF教师缓存: {path}")
    arrays, metadata = io.load_archive(path)
    if metadata.get("experiment") != config.EXPERIMENT_ID:
        raise RuntimeError(f"教师缓存实验身份不一致: {path}")
    if metadata.get("kind") != "causal_oof":
        raise RuntimeError(f"不是训练OOF教师缓存: {path}")
    if metadata.get("station") != station or metadata.get("target") != target:
        raise RuntimeError(f"教师缓存任务身份不一致: {path}")
    if not np.array_equal(
        arrays.get("target_start"),
        np.asarray(train["target_start"], dtype="datetime64[ns]"),
    ):
        raise RuntimeError(f"教师缓存时间轴与训练集不一致: {path}")
    completed = np.asarray(arrays.get("completed"), dtype=bool)
    if completed.shape != (len(config.OOF_FOLDS), config.OUTPUT_STEPS) or not completed.all():
        raise RuntimeError(f"教师OOF缓存尚未完成全部18个时距: {path}")
    return arrays, io.file_sha256(path)


def _gate_metadata(
    station: str,
    target: str,
    rows: int,
    teacher_cache_sha256: str,
) -> dict[str, object]:
    return {
        "experiment": GATE_EXPERIMENT_ID,
        "kind": "oof_persistence_gate",
        "method": GATE_METHOD,
        "station": station,
        "target": target,
        "rows": int(rows),
        "horizon_hours": list(config.HORIZON_HOURS),
        "bounds": [GATE_LOWER, GATE_UPPER],
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "teacher_cache_sha256": teacher_cache_sha256,
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            ("config.py", "data.py", "io.py", "inertia_gate.py")
        ),
    }


def fit_task_gate(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    *,
    force: bool,
) -> Path:
    teacher, teacher_sha256 = _load_oof_teacher(train, station, target)
    expected = _gate_metadata(
        station, target, len(train["target_start"]), teacher_sha256
    )
    path = gate_cache_path(station, target)
    existing = None if force else io.load_exact(path, expected)
    if existing is not None:
        console.info("resume", gate="already complete", station=station, target=target)
        return path

    teacher_delta = np.asarray(teacher["pred_delta"], dtype=float)
    teacher_mask = np.asarray(teacher["pred_mask"], dtype=bool)
    true_delta = np.asarray(train["y_delta"], dtype=float)
    true_mask = np.asarray(train["y_mask"], dtype=bool)
    arrays = fit_persistence_gate(
        teacher_delta,
        true_delta,
        teacher_mask & true_mask,
    )
    arrays["target_start"] = np.asarray(
        train["target_start"], dtype="datetime64[ns]"
    )
    io.save_archive(path, arrays, expected)
    console.info(
        "saved gate",
        station=station,
        target=target,
        alpha=",".join(f"{value:.3f}" for value in arrays["alpha"]),
    )
    return path


def load_gate(station: str, target: str) -> tuple[dict[str, np.ndarray], dict]:
    path = gate_cache_path(station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少OOF惯性门控参数: {path}")
    arrays, metadata = io.load_archive(path)
    if metadata.get("experiment") != GATE_EXPERIMENT_ID:
        raise RuntimeError(f"门控实验身份不一致: {path}")
    if metadata.get("station") != station or metadata.get("target") != target:
        raise RuntimeError(f"门控任务身份不一致: {path}")
    alpha = np.asarray(arrays.get("alpha"), dtype=float)
    if alpha.shape != (config.OUTPUT_STEPS,):
        raise RuntimeError(f"门控系数形状不正确: {path}")
    teacher_path = io.teacher_cache_path("训练OOF", station, target)
    if metadata.get("teacher_cache_sha256") != io.file_sha256(teacher_path):
        raise RuntimeError(f"门控参数对应的OOF教师缓存已变化: {path}")
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = data.load_v2_panel()
    try:
        stations, targets = select_tasks(
            panel, args.stations, args.targets, args.all_stations, args.all_targets
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            train = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["train"]
            fit_task_gate(train, station, target, force=args.force)
    console.done(config.OUTPUT_DIR / "惯性门控")


if __name__ == "__main__":
    main()
