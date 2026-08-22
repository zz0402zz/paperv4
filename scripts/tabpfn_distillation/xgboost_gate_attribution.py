#!/usr/bin/env python3
"""Generate causal XGBoost OOF predictions and fit a self-OOF inertia gate."""

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
from scripts.tabpfn_distillation.student import _parse_seeds
from scripts.tabpfn_distillation.teacher import (
    _parse_horizons,
    select_tasks,
)


ATTRIBUTION_EXPERIMENT_ID = "xgboost_self_oof_gate_attribution_4_72h_v1"


def xgboost_oof_cache_path(seed: int, station: str, target: str) -> Path:
    filename = "__".join(
        (
            "变化量XGBoost训练OOF",
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return config.OUTPUT_DIR / "门控归因实验" / "XGBoost训练OOF" / f"{filename}.npz"


def xgboost_self_gate_path(seed: int, station: str, target: str) -> Path:
    filename = "__".join(
        (
            "XGBoost自OOF惯性门控",
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return config.OUTPUT_DIR / "门控归因实验" / "XGBoost自门控参数" / f"{filename}.npz"


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


def _oof_metadata(
    seed: int, station: str, target: str, rows: int
) -> dict[str, object]:
    return {
        "experiment": ATTRIBUTION_EXPERIMENT_ID,
        "kind": "causal_xgboost_oof",
        "model": "delta_xgboost",
        "seed": int(seed),
        "station": station,
        "target": target,
        "rows": int(rows),
        "target_mode": "delta",
        "input_steps": config.INPUT_STEPS,
        "input_features": list(config.INPUT_FEATURES),
        "horizon_hours": list(config.HORIZON_HOURS),
        "oof_folds": [list(fold) for fold in config.OOF_FOLDS],
        "strictly_causal": True,
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "validation_early_stopping": False,
        "xgboost_version": _require_xgboost(),
        "xgboost_parameters": _xgboost_parameters(),
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            (
                "config.py",
                "data.py",
                "io.py",
                "models.py",
                "protocol_baselines.py",
                "xgboost_gate_attribution.py",
            )
        ),
    }


def _empty_oof_checkpoint(
    rows: int, fold_count: int, target_start: np.ndarray
) -> dict[str, np.ndarray]:
    shape = (fold_count, config.OUTPUT_STEPS)
    return {
        "pred_delta": np.full((rows, config.OUTPUT_STEPS), np.nan, dtype=float),
        "pred_mask": np.zeros((rows, config.OUTPUT_STEPS), dtype=bool),
        "fold_index": np.full(rows, -1, dtype=np.int16),
        "completed": np.zeros(shape, dtype=bool),
        "fit_rows": np.zeros(shape, dtype=np.int64),
        "prediction_rows": np.zeros(shape, dtype=np.int64),
        "training_seconds": np.zeros(shape, dtype=float),
        "inference_seconds": np.zeros(shape, dtype=float),
        "tree_count": np.zeros(shape, dtype=np.int64),
        "target_start": np.asarray(target_start, dtype="datetime64[ns]"),
    }


def _load_or_initialize_oof(
    path: Path,
    expected_metadata: dict[str, object],
    *,
    rows: int,
    fold_count: int,
    target_start: np.ndarray,
    force: bool,
) -> dict[str, np.ndarray]:
    arrays = None if force else io.load_exact(path, expected_metadata)
    if arrays is None:
        return _empty_oof_checkpoint(rows, fold_count, target_start)
    required = {
        "pred_delta",
        "pred_mask",
        "fold_index",
        "completed",
        "fit_rows",
        "prediction_rows",
        "training_seconds",
        "inference_seconds",
        "tree_count",
        "target_start",
    }
    if required.difference(arrays):
        raise RuntimeError(f"XGBoost OOF缓存字段不完整: {path}")
    expected_time = np.asarray(target_start, dtype="datetime64[ns]")
    if not np.array_equal(arrays["target_start"], expected_time):
        raise RuntimeError(f"XGBoost OOF缓存时间轴不一致: {path}")
    if arrays["pred_delta"].shape != (rows, config.OUTPUT_STEPS):
        raise RuntimeError(f"XGBoost OOF预测形状不一致: {path}")
    expected_progress = (fold_count, config.OUTPUT_STEPS)
    if arrays["completed"].shape != expected_progress:
        raise RuntimeError(f"XGBoost OOF进度形状不一致: {path}")
    return arrays


def _fit_predict(
    features: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    prediction_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float, int]:
    fit_x = np.asarray(features[fit_rows], dtype=float)
    predict_x = np.asarray(features[prediction_rows], dtype=float)
    medians = models.finite_feature_medians(fit_x)
    fit_x = models.apply_feature_medians(fit_x, medians)
    predict_x = models.apply_feature_medians(predict_x, medians)
    model = _make_xgboost(seed)
    begin = time.perf_counter()
    model.fit(fit_x, np.asarray(labels[fit_rows], dtype=float))
    training_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    predicted = np.asarray(model.predict(predict_x), dtype=float)
    inference_seconds = time.perf_counter() - begin
    tree_count = len(model.get_booster().get_dump())
    del model
    gc.collect()
    return predicted, training_seconds, inference_seconds, tree_count


def generate_xgboost_oof(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    seed: int,
    *,
    horizon_indices: tuple[int, ...],
    force: bool,
) -> tuple[Path, dict[str, np.ndarray]]:
    folds = data.causal_oof_folds(train)
    path = xgboost_oof_cache_path(seed, station, target)
    expected = _oof_metadata(seed, station, target, len(train["target_start"]))
    arrays = _load_or_initialize_oof(
        path,
        expected,
        rows=len(train["target_start"]),
        fold_count=len(folds),
        target_start=train["target_start"],
        force=force,
    )
    features = data.tabpfn_features(train)
    labels = np.asarray(train["y_delta"], dtype=float)
    label_mask = np.asarray(train["y_mask"], dtype=bool)
    current_valid = np.asarray(train["current_mask"], dtype=bool)[:, 0]

    for fold in folds:
        fold_index = int(fold["index"])
        fold_prediction_rows = np.asarray(fold["prediction_mask"], dtype=bool)
        prediction_rows = fold_prediction_rows & current_valid
        arrays["fold_index"][fold_prediction_rows] = fold_index
        if not prediction_rows.any():
            raise ValueError(
                f"XGBoost OOF折没有预测行: {station}/{target}/{fold['name']}"
            )
        for horizon in horizon_indices:
            if bool(arrays["completed"][fold_index, horizon]):
                continue
            fit_rows = (
                np.asarray(fold["fit_mask"], dtype=bool)
                & label_mask[:, horizon]
                & np.isfinite(labels[:, horizon])
            )
            if int(fit_rows.sum()) < config.MIN_TEACHER_TRAIN_ROWS:
                raise ValueError(
                    f"XGBoost OOF训练样本不足: {station}/{target}/{fold['name']}/"
                    f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
                )
            fit_label_end = np.asarray(train["target_end"])[fit_rows].max()
            prediction_start = np.asarray(train["target_start"])[
                prediction_rows
            ].min()
            if not fit_label_end < prediction_start:
                raise RuntimeError(
                    f"XGBoost OOF因果性检查失败: {station}/{target}/"
                    f"{fold['name']}/{config.HORIZON_HOURS[horizon]}h"
                )
            console.info(
                "XGBoost causal OOF",
                seed=seed,
                fold=fold["name"],
                horizon=f"{config.HORIZON_HOURS[horizon]}h",
                fit_rows=int(fit_rows.sum()),
                predict_rows=int(prediction_rows.sum()),
            )
            predicted, train_s, infer_s, trees = _fit_predict(
                features,
                labels[:, horizon],
                fit_rows,
                prediction_rows,
                seed,
            )
            row_indices = np.flatnonzero(prediction_rows)
            arrays["pred_delta"][row_indices, horizon] = predicted
            arrays["pred_mask"][row_indices, horizon] = np.isfinite(predicted)
            arrays["fit_rows"][fold_index, horizon] = int(fit_rows.sum())
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
    oof_sha256: str,
) -> dict[str, object]:
    return {
        "experiment": ATTRIBUTION_EXPERIMENT_ID,
        "kind": "xgboost_self_oof_persistence_gate",
        "method": "per_horizon_clipped_ols_through_origin",
        "source_model": "delta_xgboost",
        "seed": int(seed),
        "station": station,
        "target": target,
        "rows": int(rows),
        "horizon_hours": list(config.HORIZON_HOURS),
        "bounds": [0.0, 1.0],
        "strictly_causal": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "xgboost_oof_sha256": oof_sha256,
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            (
                "config.py",
                "data.py",
                "io.py",
                "inertia_gate.py",
                "protocol_baselines.py",
                "xgboost_gate_attribution.py",
            )
        ),
    }


def fit_xgboost_self_gate(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    seed: int,
    oof_path: Path,
    oof_arrays: dict[str, np.ndarray],
    *,
    force: bool,
) -> Path | None:
    completed = np.asarray(oof_arrays["completed"], dtype=bool)
    if not completed.all():
        console.info(
            "partial XGBoost OOF",
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
        oof_sha256,
    )
    path = xgboost_self_gate_path(seed, station, target)
    existing = None if force else io.load_exact(path, expected)
    if existing is not None:
        console.info("resume", gate="XGBoost self OOF complete", seed=seed)
        return path
    true_delta = np.asarray(train["y_delta"], dtype=float)
    true_mask = np.asarray(train["y_mask"], dtype=bool)
    prediction = np.asarray(oof_arrays["pred_delta"], dtype=float)
    prediction_mask = np.asarray(oof_arrays["pred_mask"], dtype=bool)
    arrays = fit_persistence_gate(
        prediction,
        true_delta,
        prediction_mask & true_mask,
    )
    arrays["target_start"] = np.asarray(
        train["target_start"], dtype="datetime64[ns]"
    )
    io.save_archive(path, arrays, expected)
    console.info(
        "saved XGBoost self gate",
        seed=seed,
        alpha=",".join(f"{value:.3f}" for value in arrays["alpha"]),
    )
    return path


def load_xgboost_self_gate(
    seed: int, station: str, target: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = xgboost_self_gate_path(seed, station, target)
    if not path.exists():
        raise FileNotFoundError(f"缺少XGBoost自OOF门控参数: {path}")
    arrays, metadata = io.load_archive(path)
    if (
        metadata.get("experiment") != ATTRIBUTION_EXPERIMENT_ID
        or metadata.get("kind") != "xgboost_self_oof_persistence_gate"
        or int(metadata.get("seed", -1)) != seed
        or metadata.get("station") != station
        or metadata.get("target") != target
    ):
        raise RuntimeError(f"XGBoost自OOF门控身份不一致: {path}")
    if bool(metadata.get("uses_validation_labels", True)):
        raise RuntimeError(f"XGBoost自OOF门控声明使用了验证标签: {path}")
    if bool(metadata.get("uses_test_labels", True)):
        raise RuntimeError(f"XGBoost自OOF门控声明使用了测试标签: {path}")
    oof_path = xgboost_oof_cache_path(seed, station, target)
    if metadata.get("xgboost_oof_sha256") != io.file_sha256(oof_path):
        raise RuntimeError(f"XGBoost自OOF门控对应的OOF缓存已变化: {path}")
    alpha = np.asarray(arrays.get("alpha"), dtype=float)
    if alpha.shape != (config.OUTPUT_STEPS,):
        raise RuntimeError(f"XGBoost自OOF门控系数形状不正确: {path}")
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
            train = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )["train"]
            for seed in seeds:
                oof_path, oof_arrays = generate_xgboost_oof(
                    train,
                    station,
                    target,
                    seed,
                    horizon_indices=horizon_indices,
                    force=args.force,
                )
                fit_xgboost_self_gate(
                    train,
                    station,
                    target,
                    seed,
                    oof_path,
                    oof_arrays,
                    force=args.force,
                )
    console.done(config.OUTPUT_DIR / "门控归因实验")


if __name__ == "__main__":
    main()
