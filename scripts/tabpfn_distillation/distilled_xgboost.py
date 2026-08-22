#!/usr/bin/env python3
"""Train causal TabPFN-distilled XGBoost and its own strict OOF gate."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io, models
from scripts.tabpfn_distillation.inertia_gate import fit_persistence_gate
from scripts.tabpfn_distillation.protocol_baselines import (
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_N_JOBS,
    XGBOOST_REG_ALPHA,
    XGBOOST_REG_LAMBDA,
    XGBOOST_SUBSAMPLE,
    _require_xgboost,
)
from scripts.tabpfn_distillation.student import (
    _parse_seeds,
    teacher_targets,
)
from scripts.tabpfn_distillation.teacher import (
    _parse_horizons,
    select_tasks,
)


DISTILLED_XGBOOST_EXPERIMENT_ID = "causal_tabpfn_distilled_xgboost_4_72h_v1"
DISTILLED_XGBOOST_KEY = "causal_distilled_delta_xgboost"
DISTILLED_XGBOOST_LABEL = "变化量因果蒸馏XGBoost"
TEACHER_WEIGHT = config.DISTILLATION_WEIGHT


def validation_prediction_path(seed: int, station: str, target: str) -> Path:
    filename = "__".join(
        (
            DISTILLED_XGBOOST_LABEL,
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return (
        config.output_dir_for_split("val")
        / "蒸馏XGBoost实验"
        / "预测结果"
        / f"{filename}.npz"
    )


def oof_cache_path(seed: int, station: str, target: str) -> Path:
    filename = "__".join(
        (
            "因果蒸馏XGBoost训练OOF",
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return config.OUTPUT_DIR / "蒸馏XGBoost实验" / "训练OOF" / f"{filename}.npz"


def self_gate_path(seed: int, station: str, target: str) -> Path:
    filename = "__".join(
        (
            "因果蒸馏XGBoost自OOF惯性门控",
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return config.OUTPUT_DIR / "蒸馏XGBoost实验" / "自门控参数" / f"{filename}.npz"


def build_distillation_targets(
    true_values: np.ndarray,
    true_mask: np.ndarray,
    teacher_values: np.ndarray,
    teacher_mask: np.ndarray,
    teacher_weight: float = TEACHER_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse true-MSE + lambda*teacher-MSE into weighted XGBoost labels."""

    true_values = np.asarray(true_values, dtype=float)
    teacher_values = np.asarray(teacher_values, dtype=float)
    true_valid = np.asarray(true_mask, dtype=bool) & np.isfinite(true_values)
    teacher_valid = np.asarray(teacher_mask, dtype=bool) & np.isfinite(
        teacher_values
    )
    if true_values.shape != teacher_values.shape:
        raise ValueError("真实标签与教师标签形状必须一致。")
    if true_valid.shape != true_values.shape or teacher_valid.shape != true_values.shape:
        raise ValueError("蒸馏标签掩码形状必须与标签一致。")
    if not np.isfinite(teacher_weight) or teacher_weight < 0:
        raise ValueError("教师蒸馏权重必须是非负有限数。")
    weight = true_valid.astype(float) + teacher_weight * teacher_valid.astype(
        float
    )
    numerator = np.where(true_valid, true_values, 0.0) + teacher_weight * np.where(
        teacher_valid, teacher_values, 0.0
    )
    target = np.divide(
        numerator,
        weight,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=weight > 0,
    )
    return target, weight, true_valid, teacher_valid


def _xgboost_parameters() -> dict[str, object]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": XGBOOST_N_ESTIMATORS,
        "max_depth": XGBOOST_MAX_DEPTH,
        "learning_rate": XGBOOST_LEARNING_RATE,
        "subsample": XGBOOST_SUBSAMPLE,
        "colsample_bytree": XGBOOST_COLSAMPLE_BYTREE,
        "min_child_weight": XGBOOST_MIN_CHILD_WEIGHT,
        "reg_alpha": XGBOOST_REG_ALPHA,
        "reg_lambda": XGBOOST_REG_LAMBDA,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": XGBOOST_N_JOBS,
        "verbosity": 0,
    }


def _make_xgboost(seed: int):
    _require_xgboost()
    from xgboost import XGBRegressor

    return XGBRegressor(random_state=int(seed), **_xgboost_parameters())


def _code_identity() -> dict[str, str]:
    return io.code_sha256(
        (
            "config.py",
            "data.py",
            "io.py",
            "models.py",
            "inertia_gate.py",
            "protocol_baselines.py",
            "student.py",
            "distilled_xgboost.py",
        )
    )


def _common_metadata(
    kind: str,
    seed: int,
    station: str,
    target: str,
    rows: int,
    teacher_sha256: str,
) -> dict[str, object]:
    return {
        "experiment": DISTILLED_XGBOOST_EXPERIMENT_ID,
        "kind": kind,
        "model": DISTILLED_XGBOOST_KEY,
        "model_label": DISTILLED_XGBOOST_LABEL,
        "seed": int(seed),
        "station": station,
        "target": target,
        "rows": int(rows),
        "target_mode": "delta",
        "teacher_target_mode": "delta",
        "teacher_weight": TEACHER_WEIGHT,
        "distillation_objective": (
            "true_squared_error_plus_teacher_weight_times_"
            "causal_oof_teacher_squared_error"
        ),
        "teacher_cache_sha256": teacher_sha256,
        "input_steps": config.INPUT_STEPS,
        "input_features": list(config.INPUT_FEATURES),
        "horizon_hours": list(config.HORIZON_HOURS),
        "xgboost_version": _require_xgboost(),
        "xgboost_parameters": _xgboost_parameters(),
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "validation_early_stopping": False,
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": _code_identity(),
    }


def _fit_predict(
    fit_features: np.ndarray,
    fit_target: np.ndarray,
    fit_weight: np.ndarray,
    fit_rows: np.ndarray,
    prediction_features: np.ndarray,
    prediction_rows: np.ndarray,
    median_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float, int]:
    fit_x = np.asarray(fit_features[fit_rows], dtype=float)
    fit_y = np.asarray(fit_target[fit_rows], dtype=float)
    sample_weight = np.asarray(fit_weight[fit_rows], dtype=float)
    predict_x = np.asarray(prediction_features[prediction_rows], dtype=float)
    median_x = np.asarray(fit_features[median_rows], dtype=float)
    medians = models.finite_feature_medians(median_x)
    fit_x = models.apply_feature_medians(fit_x, medians)
    predict_x = models.apply_feature_medians(predict_x, medians)
    model = _make_xgboost(seed)
    begin = time.perf_counter()
    model.fit(fit_x, fit_y, sample_weight=sample_weight)
    training_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    predicted = np.asarray(model.predict(predict_x), dtype=float)
    inference_seconds = time.perf_counter() - begin
    tree_count = len(model.get_booster().get_dump())
    del model
    gc.collect()
    return predicted, training_seconds, inference_seconds, tree_count


def _empty_validation_checkpoint(
    validation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows = len(validation["target_start"])
    return {
        "pred_delta": np.full((rows, config.OUTPUT_STEPS), np.nan, dtype=float),
        "pred_mask": np.zeros((rows, config.OUTPUT_STEPS), dtype=bool),
        "completed": np.zeros((1, config.OUTPUT_STEPS), dtype=bool),
        "fit_rows": np.zeros(config.OUTPUT_STEPS, dtype=np.int64),
        "true_rows": np.zeros(config.OUTPUT_STEPS, dtype=np.int64),
        "teacher_rows": np.zeros(config.OUTPUT_STEPS, dtype=np.int64),
        "training_seconds": np.zeros(config.OUTPUT_STEPS, dtype=float),
        "inference_seconds": np.zeros(config.OUTPUT_STEPS, dtype=float),
        "tree_count": np.zeros(config.OUTPUT_STEPS, dtype=np.int64),
        "target_start": np.asarray(
            validation["target_start"], dtype="datetime64[ns]"
        ),
    }


def _empty_oof_checkpoint(
    train: dict[str, np.ndarray], fold_count: int
) -> dict[str, np.ndarray]:
    rows = len(train["target_start"])
    shape = (fold_count, config.OUTPUT_STEPS)
    return {
        "pred_delta": np.full((rows, config.OUTPUT_STEPS), np.nan, dtype=float),
        "pred_mask": np.zeros((rows, config.OUTPUT_STEPS), dtype=bool),
        "fold_index": np.full(rows, -1, dtype=np.int16),
        "completed": np.zeros(shape, dtype=bool),
        "fit_rows": np.zeros(shape, dtype=np.int64),
        "true_rows": np.zeros(shape, dtype=np.int64),
        "teacher_rows": np.zeros(shape, dtype=np.int64),
        "prediction_rows": np.zeros(shape, dtype=np.int64),
        "training_seconds": np.zeros(shape, dtype=float),
        "inference_seconds": np.zeros(shape, dtype=float),
        "tree_count": np.zeros(shape, dtype=np.int64),
        "target_start": np.asarray(train["target_start"], dtype="datetime64[ns]"),
    }


def _load_checkpoint(
    path: Path,
    expected_metadata: dict[str, object],
    empty: dict[str, np.ndarray],
    *,
    force: bool,
) -> dict[str, np.ndarray]:
    arrays = None if force else io.load_exact(path, expected_metadata)
    if arrays is None:
        return empty
    if set(empty).difference(arrays):
        raise RuntimeError(f"蒸馏XGBoost缓存字段不完整: {path}")
    for key in ("pred_delta", "completed", "target_start"):
        if arrays[key].shape != empty[key].shape:
            raise RuntimeError(f"蒸馏XGBoost缓存形状不一致: {path}/{key}")
    if not np.array_equal(arrays["target_start"], empty["target_start"]):
        raise RuntimeError(f"蒸馏XGBoost缓存时间轴不一致: {path}")
    return arrays


def generate_validation_prediction(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    station: str,
    target: str,
    seed: int,
    distilled_target: np.ndarray,
    distilled_weight: np.ndarray,
    true_valid: np.ndarray,
    teacher_valid: np.ndarray,
    teacher_sha256: str,
    *,
    horizon_indices: tuple[int, ...],
    force: bool,
) -> Path:
    path = validation_prediction_path(seed, station, target)
    expected = _common_metadata(
        "distilled_xgboost_validation_prediction",
        seed,
        station,
        target,
        len(validation["target_start"]),
        teacher_sha256,
    )
    arrays = _load_checkpoint(
        path,
        expected,
        _empty_validation_checkpoint(validation),
        force=force,
    )
    train_x = data.tabpfn_features(train)
    validation_x = data.tabpfn_features(validation)
    prediction_rows = np.ones(len(validation_x), dtype=bool)
    median_rows = np.ones(len(train_x), dtype=bool)
    for horizon in horizon_indices:
        if bool(arrays["completed"][0, horizon]):
            continue
        fit_rows = (
            distilled_weight[:, horizon] > 0
        ) & np.isfinite(distilled_target[:, horizon])
        if int(fit_rows.sum()) < config.MIN_TEACHER_TRAIN_ROWS:
            raise ValueError(
                f"蒸馏XGBoost验证训练样本不足: {station}/{target}/"
                f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
            )
        console.info(
            "distilled XGBoost validation",
            seed=seed,
            horizon=f"{config.HORIZON_HOURS[horizon]}h",
            fit_rows=int(fit_rows.sum()),
            true_rows=int(true_valid[:, horizon].sum()),
            teacher_rows=int(teacher_valid[:, horizon].sum()),
        )
        predicted, train_s, infer_s, trees = _fit_predict(
            train_x,
            distilled_target[:, horizon],
            distilled_weight[:, horizon],
            fit_rows,
            validation_x,
            prediction_rows,
            median_rows,
            seed,
        )
        arrays["pred_delta"][:, horizon] = predicted
        arrays["pred_mask"][:, horizon] = np.isfinite(predicted)
        arrays["fit_rows"][horizon] = int(fit_rows.sum())
        arrays["true_rows"][horizon] = int(true_valid[:, horizon].sum())
        arrays["teacher_rows"][horizon] = int(teacher_valid[:, horizon].sum())
        arrays["training_seconds"][horizon] = train_s
        arrays["inference_seconds"][horizon] = infer_s
        arrays["tree_count"][horizon] = trees
        arrays["completed"][0, horizon] = True
        absolute = data.to_absolute(
            arrays["pred_delta"], validation["current"], "delta"
        )
        saved = {**arrays, **data.prediction_arrays(validation, absolute)}
        io.save_archive(path, saved, expected)
    return path


def generate_oof(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    seed: int,
    distilled_target: np.ndarray,
    distilled_weight: np.ndarray,
    true_valid: np.ndarray,
    teacher_valid: np.ndarray,
    teacher_sha256: str,
    *,
    horizon_indices: tuple[int, ...],
    force: bool,
) -> tuple[Path, dict[str, np.ndarray]]:
    folds = data.causal_oof_folds(train)
    path = oof_cache_path(seed, station, target)
    expected = {
        **_common_metadata(
            "causal_distilled_xgboost_oof",
            seed,
            station,
            target,
            len(train["target_start"]),
            teacher_sha256,
        ),
        "strictly_causal": True,
        "oof_folds": [list(fold) for fold in config.OOF_FOLDS],
    }
    arrays = _load_checkpoint(
        path,
        expected,
        _empty_oof_checkpoint(train, len(folds)),
        force=force,
    )
    features = data.tabpfn_features(train)
    current_valid = np.asarray(train["current_mask"], dtype=bool)[:, 0]
    for fold in folds:
        fold_index = int(fold["index"])
        fold_fit = np.asarray(fold["fit_mask"], dtype=bool)
        fold_prediction = np.asarray(fold["prediction_mask"], dtype=bool)
        prediction_rows = fold_prediction & current_valid
        arrays["fold_index"][fold_prediction] = fold_index
        if not prediction_rows.any():
            raise ValueError(
                f"蒸馏XGBoost OOF折没有预测行: "
                f"{station}/{target}/{fold['name']}"
            )
        for horizon in horizon_indices:
            if bool(arrays["completed"][fold_index, horizon]):
                continue
            fit_rows = (
                fold_fit
                & (distilled_weight[:, horizon] > 0)
                & np.isfinite(distilled_target[:, horizon])
            )
            median_rows = fold_fit & true_valid[:, horizon]
            if int(fit_rows.sum()) < config.MIN_TEACHER_TRAIN_ROWS:
                raise ValueError(
                    f"蒸馏XGBoost OOF训练样本不足: "
                    f"{station}/{target}/{fold['name']}/"
                    f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
                )
            if not median_rows.any():
                raise ValueError(
                    f"蒸馏XGBoost OOF缺少因果中位数拟合行: "
                    f"{station}/{target}/{fold['name']}"
                )
            fit_label_end = np.asarray(train["target_end"])[fit_rows].max()
            prediction_start = np.asarray(train["target_start"])[
                prediction_rows
            ].min()
            if not fit_label_end < prediction_start:
                raise RuntimeError(
                    f"蒸馏XGBoost OOF因果性检查失败: "
                    f"{station}/{target}/{fold['name']}/"
                    f"{config.HORIZON_HOURS[horizon]}h"
                )
            console.info(
                "distilled XGBoost causal OOF",
                seed=seed,
                fold=fold["name"],
                horizon=f"{config.HORIZON_HOURS[horizon]}h",
                fit_rows=int(fit_rows.sum()),
                predict_rows=int(prediction_rows.sum()),
            )
            predicted, train_s, infer_s, trees = _fit_predict(
                features,
                distilled_target[:, horizon],
                distilled_weight[:, horizon],
                fit_rows,
                features,
                prediction_rows,
                median_rows,
                seed,
            )
            row_indices = np.flatnonzero(prediction_rows)
            arrays["pred_delta"][row_indices, horizon] = predicted
            arrays["pred_mask"][row_indices, horizon] = np.isfinite(predicted)
            arrays["fit_rows"][fold_index, horizon] = int(fit_rows.sum())
            arrays["true_rows"][fold_index, horizon] = int(
                (fold_fit & true_valid[:, horizon]).sum()
            )
            arrays["teacher_rows"][fold_index, horizon] = int(
                (fold_fit & teacher_valid[:, horizon]).sum()
            )
            arrays["prediction_rows"][fold_index, horizon] = int(
                prediction_rows.sum()
            )
            arrays["training_seconds"][fold_index, horizon] = train_s
            arrays["inference_seconds"][fold_index, horizon] = infer_s
            arrays["tree_count"][fold_index, horizon] = trees
            arrays["completed"][fold_index, horizon] = True
            io.save_archive(path, arrays, expected)
    return path, arrays


def _gate_metadata(
    seed: int,
    station: str,
    target: str,
    rows: int,
    teacher_sha256: str,
    oof_sha256: str,
) -> dict[str, object]:
    return {
        "experiment": DISTILLED_XGBOOST_EXPERIMENT_ID,
        "kind": "distilled_xgboost_self_oof_persistence_gate",
        "method": "per_horizon_clipped_ols_through_origin",
        "source_model": DISTILLED_XGBOOST_KEY,
        "seed": int(seed),
        "station": station,
        "target": target,
        "rows": int(rows),
        "teacher_weight": TEACHER_WEIGHT,
        "horizon_hours": list(config.HORIZON_HOURS),
        "bounds": [0.0, 1.0],
        "strictly_causal": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "teacher_cache_sha256": teacher_sha256,
        "distilled_xgboost_oof_sha256": oof_sha256,
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": _code_identity(),
    }


def fit_self_gate(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    seed: int,
    teacher_sha256: str,
    oof_path: Path,
    oof_arrays: dict[str, np.ndarray],
    *,
    force: bool,
) -> Path | None:
    completed = np.asarray(oof_arrays["completed"], dtype=bool)
    if not completed.all():
        console.info(
            "partial distilled XGBoost OOF",
            seed=seed,
            completed=int(completed.sum()),
            total=int(completed.size),
        )
        return None
    oof_sha256 = io.file_sha256(oof_path)
    expected = _gate_metadata(
        seed,
        station,
        target,
        len(train["target_start"]),
        teacher_sha256,
        oof_sha256,
    )
    path = self_gate_path(seed, station, target)
    existing = None if force else io.load_exact(path, expected)
    if existing is not None:
        console.info("resume", gate="distilled XGBoost self OOF complete", seed=seed)
        return path
    arrays = fit_persistence_gate(
        np.asarray(oof_arrays["pred_delta"], dtype=float),
        np.asarray(train["y_delta"], dtype=float),
        np.asarray(oof_arrays["pred_mask"], dtype=bool)
        & np.asarray(train["y_mask"], dtype=bool),
    )
    arrays["target_start"] = np.asarray(
        train["target_start"], dtype="datetime64[ns]"
    )
    io.save_archive(path, arrays, expected)
    console.info(
        "saved distilled XGBoost self gate",
        seed=seed,
        alpha=",".join(f"{value:.3f}" for value in arrays["alpha"]),
    )
    return path


def load_validation_prediction(
    seed: int, station: str, target: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = validation_prediction_path(seed, station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少蒸馏XGBoost验证预测: {path}")
    arrays, metadata = io.load_archive(path)
    if (
        metadata.get("experiment") != DISTILLED_XGBOOST_EXPERIMENT_ID
        or metadata.get("kind") != "distilled_xgboost_validation_prediction"
        or int(metadata.get("seed", -1)) != seed
        or metadata.get("station") != station
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"蒸馏XGBoost验证预测身份不一致: {path}")
    completed = np.asarray(arrays.get("completed"), dtype=bool)
    if completed.shape != (1, config.OUTPUT_STEPS) or not completed.all():
        raise RuntimeError(f"蒸馏XGBoost验证预测未完成18个时距: {path}")
    if bool(metadata.get("validation_labels_used_for_fit", True)):
        raise RuntimeError(f"蒸馏XGBoost声明使用了验证标签: {path}")
    if bool(metadata.get("test_labels_used", True)):
        raise RuntimeError(f"蒸馏XGBoost声明使用了测试标签: {path}")
    teacher_path = io.teacher_cache_path("训练OOF", station, target)
    if metadata.get("teacher_cache_sha256") != io.file_sha256(teacher_path):
        raise RuntimeError(f"蒸馏XGBoost对应的TabPFN OOF缓存已变化: {path}")
    return arrays, metadata


def load_self_gate(
    seed: int, station: str, target: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = self_gate_path(seed, station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少蒸馏XGBoost自OOF门控: {path}")
    arrays, metadata = io.load_archive(path)
    if (
        metadata.get("experiment") != DISTILLED_XGBOOST_EXPERIMENT_ID
        or metadata.get("kind")
        != "distilled_xgboost_self_oof_persistence_gate"
        or int(metadata.get("seed", -1)) != seed
        or metadata.get("station") != station
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控身份不一致: {path}")
    if bool(metadata.get("uses_validation_labels", True)):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控使用了验证标签: {path}")
    if bool(metadata.get("uses_test_labels", True)):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控使用了测试标签: {path}")
    source_path = oof_cache_path(seed, station, target)
    if metadata.get("distilled_xgboost_oof_sha256") != io.file_sha256(
        source_path
    ):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控源缓存已变化: {path}")
    teacher_path = io.teacher_cache_path("训练OOF", station, target)
    if metadata.get("teacher_cache_sha256") != io.file_sha256(teacher_path):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控的教师缓存已变化: {path}")
    alpha = np.asarray(arrays.get("alpha"), dtype=float)
    if alpha.shape != (config.OUTPUT_STEPS,):
        raise RuntimeError(f"蒸馏XGBoost自OOF门控系数形状不正确: {path}")
    return arrays, metadata


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
        "--horizons",
        default=",".join(map(str, config.HORIZON_HOURS)),
        help="Comma-separated subset of 4,8,...,72. Progress shares one cache.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = data.load_v2_panel()
    try:
        stations, targets = select_tasks(
            panel, args.stations, args.targets, args.all_stations, args.all_targets
        )
        seeds = _parse_seeds(args.seeds)
        horizon_indices = _parse_horizons(args.horizons)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            splits = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )
            train = splits["train"]
            teacher_values, teacher_mask, teacher_sha256 = teacher_targets(
                train, station, target, "delta"
            )
            distilled = build_distillation_targets(
                train["y_delta"],
                train["y_mask"],
                teacher_values,
                teacher_mask,
                TEACHER_WEIGHT,
            )
            distilled_target, distilled_weight, true_valid, teacher_valid = distilled
            for seed in seeds:
                generate_validation_prediction(
                    train,
                    splits["val"],
                    station,
                    target,
                    seed,
                    distilled_target,
                    distilled_weight,
                    true_valid,
                    teacher_valid,
                    teacher_sha256,
                    horizon_indices=horizon_indices,
                    force=args.force,
                )
                source_path, oof_arrays = generate_oof(
                    train,
                    station,
                    target,
                    seed,
                    distilled_target,
                    distilled_weight,
                    true_valid,
                    teacher_valid,
                    teacher_sha256,
                    horizon_indices=horizon_indices,
                    force=args.force,
                )
                fit_self_gate(
                    train,
                    station,
                    target,
                    seed,
                    teacher_sha256,
                    source_path,
                    oof_arrays,
                    force=args.force,
                )
    console.done(config.OUTPUT_DIR / "蒸馏XGBoost实验")


if __name__ == "__main__":
    main()
