#!/usr/bin/env python3
"""Generate strict forward-OOF predictions for candidate teachers."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import data as base_data
from scripts.multitarget_forecasting import io
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations
from scripts.teacher_screening import config, data, models


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_models(value: str) -> tuple[str, ...]:
    selected = _csv(value)
    if not selected:
        raise ValueError("至少需要一个教师候选。")
    unknown = set(selected).difference(config.MODELS)
    if unknown:
        raise ValueError(f"未知教师候选: {sorted(unknown)}")
    return selected


def parse_horizons(value: str) -> tuple[int, ...]:
    hours = tuple(int(item) for item in _csv(value))
    if not hours:
        raise ValueError("至少需要一个预测时距。")
    if len(set(hours)) != len(hours):
        raise ValueError("预测时距不能重复。")
    data.horizon_indices(hours)
    return hours


def representative_stations() -> tuple[str, ...]:
    path = config.REPRESENTATIVE_STATIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"缺少代表站点冻结文件: {path}。请先运行 "
            "python -m scripts.multitarget_forecasting.station_screening"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    stations = tuple(str(item) for item in payload.get("stations", ()))
    if not stations:
        raise RuntimeError(f"代表站点文件没有站点: {path}")
    if payload.get("selection_uses_2024_labels") is not False:
        raise RuntimeError("代表站点必须只根据训练期数据筛选。")
    return stations


def select_screening_stations(
    panel,
    *,
    stations: str | None,
    all_stations: bool,
    representatives: bool,
) -> tuple[str, ...]:
    if representatives:
        selected = representative_stations()
        available = set(base_data.available_stations(panel))
        unknown = set(selected).difference(available)
        if unknown:
            raise ValueError(f"代表站点不在当前数据中: {sorted(unknown)}")
        return selected
    return select_stations(panel, stations, all_stations)


@lru_cache(maxsize=1)
def frozen_identity() -> dict[str, object]:
    paths = (
        Path("scripts/multitarget_forecasting/config.py"),
        Path("scripts/multitarget_forecasting/data.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_config.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_data.py"),
        Path("scripts/teacher_screening/config.py"),
        Path("scripts/teacher_screening/data.py"),
        Path("scripts/teacher_screening/models.py"),
        Path("scripts/teacher_screening/run.py"),
    )
    return {
        **io.data_identity(),
        "code_sha256": {str(path): io.file_sha256(path) for path in paths},
    }


def expected_metadata(
    station: str,
    model: str,
    seed: int,
    device: str,
    horizon_hours: tuple[int, ...],
) -> dict[str, object]:
    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": "causal_forward_oof_teacher_prediction",
        "station": station,
        "seed": int(seed),
        "device_requested": device,
        "horizon_hours": list(horizon_hours),
        "targets": list(config.TARGETS),
        "target_output_modes": config.TARGET_OUTPUT_MODES,
        "log_targets": list(config.LOG_TARGETS),
        "input_history_hours": 24,
        "oof_folds": [list(fold) for fold in config.OOF_FOLDS],
        "uses_2024_labels": False,
        "uses_2025_labels": False,
        "scaler": "per_fold_training_only_median_iqr",
        **models.candidate_identity(model),
        **frozen_identity(),
    }


def _empty_arrays(
    train: dict[str, np.ndarray], horizon_indices: tuple[int, ...]
) -> dict[str, np.ndarray]:
    rows = len(train["target_start"])
    horizon_count = len(horizon_indices)
    fold_count = len(config.OOF_FOLDS)
    truth, mask, current = data.oof_truth(train, horizon_indices)
    return {
        "pred": np.full((rows, horizon_count, len(config.TARGETS)), np.nan),
        "true": truth,
        "mask": mask,
        "current": current,
        "target_start": np.asarray(train["target_start"], dtype="datetime64[ns]"),
        "fold_index": np.full(rows, -1, dtype=np.int16),
        "completed_folds": np.zeros(fold_count, dtype=bool),
        "training_seconds_by_fold": np.zeros(fold_count, dtype=float),
        "inference_seconds_by_fold": np.zeros(fold_count, dtype=float),
        "fitted_models_by_fold": np.zeros(fold_count, dtype=np.int64),
        "selected_epoch_by_fold": np.full(fold_count, -1, dtype=np.int64),
        "warning_lower_by_fold": np.full(
            (fold_count, len(config.TARGETS)), np.nan, dtype=float
        ),
        "warning_upper_by_fold": np.full(
            (fold_count, len(config.TARGETS)), np.nan, dtype=float
        ),
    }


def _load_or_initialize(
    path: Path,
    expected: dict[str, object],
    train: dict[str, np.ndarray],
    horizon_indices: tuple[int, ...],
    *,
    force: bool,
) -> dict[str, np.ndarray]:
    if force or not path.exists():
        return _empty_arrays(train, horizon_indices)
    arrays, metadata = io.load_archive(path)
    if metadata != expected:
        raise RuntimeError(
            f"已有教师筛选缓存与当前协议不一致: {path}。"
            "请审阅后显式使用 --force。"
        )
    expected_shape = (
        len(train["target_start"]),
        len(horizon_indices),
        len(config.TARGETS),
    )
    if arrays.get("pred", np.empty(0)).shape != expected_shape:
        raise RuntimeError(f"教师筛选缓存形状错误: {path}")
    return arrays


def run(
    *,
    stations: str | None,
    all_stations: bool,
    representatives: bool,
    selected_models: tuple[str, ...],
    horizon_hours: tuple[int, ...],
    seeds: tuple[int, ...],
    device: str,
    force: bool,
) -> None:
    panel = data.load_training_panel()
    selected_stations = select_screening_stations(
        panel,
        stations=stations,
        all_stations=all_stations,
        representatives=representatives,
    )
    horizon_indices = data.horizon_indices(horizon_hours)
    total = len(selected_stations) * len(selected_models) * len(seeds)
    progress = 0
    for station in selected_stations:
        train = data.build_training_dataset(panel, station)
        folds = data.causal_oof_folds(train)
        if not len(train["target_start"]):
            raise ValueError(f"站点没有训练窗口: {station}")
        for model in selected_models:
            models.require_candidate(model)
            for seed in seeds:
                progress += 1
                path = config.prediction_path(station, model, seed, horizon_hours)
                expected = expected_metadata(
                    station, model, seed, device, horizon_hours
                )
                arrays = _load_or_initialize(
                    path,
                    expected,
                    train,
                    horizon_indices,
                    force=force,
                )
                console.phase(
                    f"{station} / {config.MODEL_LABELS[model]}",
                    current=progress,
                    total=total,
                )
                for fold in folds:
                    fold_index = int(fold["index"])
                    prediction_mask = np.asarray(
                        fold["prediction_mask"], dtype=bool
                    )
                    arrays["fold_index"][prediction_mask] = fold_index
                    if bool(arrays["completed_folds"][fold_index]):
                        console.info("resume_fold", fold=fold["name"])
                        continue
                    fit_mask = np.asarray(fold["fit_mask"], dtype=bool)
                    if int(fit_mask.sum()) < config.MIN_TRAIN_ROWS:
                        raise ValueError(
                            f"OOF拟合行不足: {station}/{fold['name']}="
                            f"{int(fit_mask.sum())}"
                        )
                    console.info(
                        "fit_oof_fold",
                        fold=fold["name"],
                        fit_rows=int(fit_mask.sum()),
                        prediction_rows=int(prediction_mask.sum()),
                        horizons=horizon_hours,
                    )
                    prepared = data.prepare_fold(
                        train, fit_mask, prediction_mask, horizon_indices
                    )
                    fit_split = base_data.subset(train, fit_mask)
                    warning_lower, warning_upper = base_data.warning_thresholds(
                        fit_split
                    )
                    arrays["warning_lower_by_fold"][fold_index] = warning_lower
                    arrays["warning_upper_by_fold"][fold_index] = warning_upper
                    predicted, diagnostics = models.fit_predict(
                        model,
                        prepared,
                        seed=seed,
                        device=device,
                    )
                    row_indices = np.flatnonzero(prediction_mask)
                    if predicted.shape != arrays["pred"][row_indices].shape:
                        raise RuntimeError(
                            f"教师预测形状错误: expected="
                            f"{arrays['pred'][row_indices].shape}, actual={predicted.shape}"
                        )
                    arrays["pred"][row_indices] = predicted
                    arrays["training_seconds_by_fold"][fold_index] = float(
                        diagnostics["training_seconds"]
                    )
                    arrays["inference_seconds_by_fold"][fold_index] = float(
                        diagnostics["inference_seconds"]
                    )
                    arrays["fitted_models_by_fold"][fold_index] = int(
                        diagnostics["fitted_models"]
                    )
                    if "selected_epoch" in diagnostics:
                        arrays["selected_epoch_by_fold"][fold_index] = int(
                            diagnostics["selected_epoch"]
                        )
                    arrays["completed_folds"][fold_index] = True
                    io.save_archive(path, arrays, expected)
                    console.info(
                        "saved_fold",
                        fold=fold["name"],
                        train_s=float(diagnostics["training_seconds"]),
                        inference_s=float(diagnostics["inference_seconds"]),
                    )
                console.info("saved", output=path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="逗号分隔的站点")
    station_group.add_argument("--all-stations", action="store_true")
    station_group.add_argument("--representative-stations", action="store_true")
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument(
        "--horizons", default=",".join(map(str, config.ANCHOR_HOURS))
    )
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        stations=args.stations,
        all_stations=args.all_stations,
        representatives=args.representative_stations,
        selected_models=parse_models(args.models),
        horizon_hours=parse_horizons(args.horizons),
        seeds=_parse_seeds(args.seeds),
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
