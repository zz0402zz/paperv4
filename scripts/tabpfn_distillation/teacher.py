#!/usr/bin/env python3
"""Generate causal OOF and validation Delta-TabPFN teacher predictions."""

from __future__ import annotations

import argparse
import gc
from importlib import metadata as package_metadata

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io, models


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_horizons(value: str) -> tuple[int, ...]:
    hours = tuple(int(item) for item in _parse_csv(value))
    if not hours:
        raise ValueError("至少需要一个预测时距。")
    unknown = set(hours).difference(config.HORIZON_HOURS)
    if unknown:
        raise ValueError(f"时距必须来自4、8、…、72小时: {sorted(unknown)}")
    if len(set(hours)) != len(hours):
        raise ValueError("预测时距不能重复。")
    return tuple(config.HORIZON_HOURS.index(hour) for hour in hours)


def select_tasks(
    panel,
    stations_arg: str | None,
    targets_arg: str | None,
    all_stations: bool,
    all_targets: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = data.available_stations(panel)
    stations = available if all_stations else _parse_csv(stations_arg or "")
    targets = config.TARGETS if all_targets else _parse_csv(targets_arg or "")
    if not stations or not targets:
        raise ValueError("必须明确选择站点和指标，或使用 --all-stations/--all-targets。")
    unknown_stations = set(stations).difference(available)
    unknown_targets = set(targets).difference(config.TARGETS)
    if unknown_stations:
        raise ValueError(f"未知站点: {sorted(unknown_stations)}")
    if unknown_targets:
        raise ValueError(f"未知指标: {sorted(unknown_targets)}")
    return tuple(stations), tuple(targets)


def _teacher_metadata(kind: str, station: str, target: str, rows: int) -> dict[str, object]:
    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": kind,
        "station": station,
        "target": target,
        "teacher_target_mode": "delta",
        "teacher_seed": config.TEACHER_SEED,
        "model_identity": models.MODEL_IDENTITY,
        "tabpfn_version": package_metadata.version("tabpfn"),
        "tabpfn_fit_mode": config.TABPFN_FIT_MODE,
        "tabpfn_prediction_batch_size": config.TABPFN_PREDICTION_BATCH_SIZE,
        "input_steps": config.INPUT_STEPS,
        "horizon_hours": list(config.HORIZON_HOURS),
        "rows": int(rows),
        "oof_folds": [list(fold) for fold in config.OOF_FOLDS],
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            ("config.py", "data.py", "io.py", "models.py", "teacher.py")
        ),
    }


def _empty_checkpoint(rows: int, fold_count: int, target_start: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pred_delta": np.full((rows, config.OUTPUT_STEPS), np.nan, dtype=float),
        "pred_mask": np.zeros((rows, config.OUTPUT_STEPS), dtype=bool),
        "fold_index": np.full(rows, -1, dtype=np.int16),
        "completed": np.zeros((fold_count, config.OUTPUT_STEPS), dtype=bool),
        "target_start": np.asarray(target_start, dtype="datetime64[ns]"),
    }


def _load_or_initialize(
    path,
    expected_metadata: dict,
    *,
    rows: int,
    fold_count: int,
    target_start: np.ndarray,
    force: bool,
) -> dict[str, np.ndarray]:
    arrays = None if force else io.load_exact(path, expected_metadata)
    if arrays is None:
        return _empty_checkpoint(rows, fold_count, target_start)
    required = {"pred_delta", "pred_mask", "fold_index", "completed", "target_start"}
    if required.difference(arrays):
        raise RuntimeError(f"教师缓存字段不完整: {path}")
    if not np.array_equal(arrays["target_start"], np.asarray(target_start, dtype="datetime64[ns]")):
        raise RuntimeError(f"教师缓存时间轴与当前数据不一致: {path}")
    if arrays["pred_delta"].shape != (rows, config.OUTPUT_STEPS):
        raise RuntimeError(f"教师缓存形状与当前协议不一致: {path}")
    if arrays["completed"].shape != (fold_count, config.OUTPUT_STEPS):
        raise RuntimeError(f"教师缓存进度形状不一致: {path}")
    return arrays


def _release_teacher(regressor) -> None:
    del regressor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fit_predict(
    fit_features: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    prediction_features: np.ndarray,
    prediction_rows: np.ndarray,
) -> np.ndarray:
    fit_features = np.asarray(fit_features)
    labels = np.asarray(labels)
    fit_rows = np.asarray(fit_rows, dtype=bool)
    prediction_features = np.asarray(prediction_features)
    prediction_rows = np.asarray(prediction_rows, dtype=bool)
    if len(fit_features) != len(labels) or len(fit_features) != len(fit_rows):
        raise ValueError("教师拟合特征、标签和拟合掩码的行数必须一致。")
    if len(prediction_features) != len(prediction_rows):
        raise ValueError("教师预测特征和预测掩码的行数必须一致。")
    fit_x = np.asarray(fit_features[fit_rows], dtype=float)
    predict_x = np.asarray(prediction_features[prediction_rows], dtype=float)
    medians = models.finite_feature_medians(fit_x)
    fit_x = models.apply_feature_medians(fit_x, medians)
    predict_x = models.apply_feature_medians(predict_x, medians)
    regressor = models.make_teacher()
    try:
        regressor.fit(fit_x, np.asarray(labels[fit_rows], dtype=float))
        batches = []
        for begin in range(0, len(predict_x), config.TABPFN_PREDICTION_BATCH_SIZE):
            end = begin + config.TABPFN_PREDICTION_BATCH_SIZE
            batches.append(np.asarray(regressor.predict(predict_x[begin:end]), dtype=float))
        return np.concatenate(batches) if batches else np.asarray([], dtype=float)
    finally:
        _release_teacher(regressor)


def generate_oof(
    train: dict[str, np.ndarray],
    station: str,
    target: str,
    *,
    horizon_indices: tuple[int, ...],
    force: bool,
) -> None:
    models.require_tabpfn()
    folds = data.causal_oof_folds(train)
    path = io.teacher_cache_path("训练OOF", station, target)
    expected = _teacher_metadata("causal_oof", station, target, len(train["target_start"]))
    arrays = _load_or_initialize(
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
        prediction_rows = np.asarray(fold["prediction_mask"], dtype=bool) & current_valid
        arrays["fold_index"][np.asarray(fold["prediction_mask"], dtype=bool)] = fold_index
        if not prediction_rows.any():
            raise ValueError(f"OOF折没有预测行: {station}/{target}/{fold['name']}")
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
                    f"教师训练样本不足: {station}/{target}/{fold['name']}/"
                    f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
                )
            console.info(
                "teacher OOF",
                fold=fold["name"],
                horizon=f"{config.HORIZON_HOURS[horizon]}h",
                fit_rows=int(fit_rows.sum()),
                predict_rows=int(prediction_rows.sum()),
            )
            predicted = _fit_predict(
                features,
                labels[:, horizon],
                fit_rows,
                features,
                prediction_rows,
            )
            row_indices = np.flatnonzero(prediction_rows)
            arrays["pred_delta"][row_indices, horizon] = predicted
            arrays["pred_mask"][row_indices, horizon] = np.isfinite(predicted)
            arrays["completed"][fold_index, horizon] = True
            io.save_archive(path, arrays, expected)


def generate_validation(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    station: str,
    target: str,
    *,
    horizon_indices: tuple[int, ...],
    force: bool,
) -> None:
    models.require_tabpfn()
    path = io.teacher_cache_path("验证集", station, target)
    expected = _teacher_metadata("validation", station, target, len(validation["target_start"]))
    arrays = _load_or_initialize(
        path,
        expected,
        rows=len(validation["target_start"]),
        fold_count=1,
        target_start=validation["target_start"],
        force=force,
    )
    train_x = data.tabpfn_features(train)
    validation_x = data.tabpfn_features(validation)
    labels = np.asarray(train["y_delta"], dtype=float)
    label_mask = np.asarray(train["y_mask"], dtype=bool)
    prediction_rows = np.asarray(validation["current_mask"], dtype=bool)[:, 0]
    for horizon in horizon_indices:
        if bool(arrays["completed"][0, horizon]):
            continue
        fit_rows = label_mask[:, horizon] & np.isfinite(labels[:, horizon])
        if int(fit_rows.sum()) < config.MIN_TEACHER_TRAIN_ROWS:
            raise ValueError(
                f"验证教师训练样本不足: {station}/{target}/"
                f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
            )
        console.info(
            "teacher validation",
            horizon=f"{config.HORIZON_HOURS[horizon]}h",
            fit_rows=int(fit_rows.sum()),
            predict_rows=int(prediction_rows.sum()),
        )
        predicted = _fit_predict(
            train_x,
            labels[:, horizon],
            fit_rows,
            validation_x,
            prediction_rows,
        )
        row_indices = np.flatnonzero(prediction_rows)
        arrays["pred_delta"][row_indices, horizon] = predicted
        arrays["pred_mask"][row_indices, horizon] = np.isfinite(predicted)
        arrays["completed"][0, horizon] = True
        io.save_archive(path, arrays, expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--cache", choices=("all", "oof", "validation"), default="all")
    parser.add_argument(
        "--horizons",
        default=",".join(map(str, config.HORIZON_HOURS)),
        help="Comma-separated subset of 4,8,...,72. Progress shares the same full cache.",
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
        horizon_indices = _parse_horizons(args.horizons)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            splits = data.split_by_time(data.build_station_target_dataset(panel, station, target))
            if args.cache in {"all", "oof"}:
                generate_oof(
                    splits["train"],
                    station,
                    target,
                    horizon_indices=horizon_indices,
                    force=args.force,
                )
            if args.cache in {"all", "validation"}:
                generate_validation(
                    splits["train"],
                    splits["val"],
                    station,
                    target,
                    horizon_indices=horizon_indices,
                    force=args.force,
                )
    console.done(config.OUTPUT_DIR / "教师缓存")


if __name__ == "__main__":
    main()
