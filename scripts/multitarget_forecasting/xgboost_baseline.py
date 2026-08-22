#!/usr/bin/env python3
"""Same-sample XGBoost baselines for all five targets and 18 horizons."""

from __future__ import annotations

import argparse
from functools import lru_cache
import gc
from importlib import metadata as package_metadata
from pathlib import Path
import time

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config, data, io
from scripts.multitarget_forecasting.run import (
    _parse_seeds,
    _parse_target_modes,
    select_stations,
)


EXPERIMENT_ID = "joint_five_target_same_sample_xgboost_4_72h_v2"
OUTPUT_DIR = config.OUTPUT_DIR / "验证集" / "同协议XGBoost基线"
CONTEXT = "24h"
DEVICES = ("cpu", "cuda")
XGBOOST_VERSION = "3.2.0"
N_ESTIMATORS = 400
MAX_DEPTH = 6
LEARNING_RATE = 0.03
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
MIN_CHILD_WEIGHT = 1.0
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
N_JOBS = -1
MIN_TRAIN_ROWS = 256


def device_output_dir(device: str) -> Path:
    if device not in DEVICES:
        raise ValueError(f"未知XGBoost设备: {device}")
    return OUTPUT_DIR / ("GPU" if device == "cuda" else "CPU")


def prediction_path(
    station: str, target_mode: str, seed: int, device: str
) -> Path:
    return device_output_dir(device) / "预测结果" / (
        "__".join(
            (
                f"{config.TARGET_MODE_LABELS[target_mode]}XGBoost",
                f"种子{seed}",
                io.safe_filename(station),
                "五指标18时距",
            )
        )
        + ".npz"
    )


def require_xgboost() -> str:
    try:
        installed = package_metadata.version("xgboost")
    except package_metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "缺少xgboost==3.2.0，请先在.venv-tabpfn环境中安装。"
        ) from exc
    if installed != XGBOOST_VERSION:
        raise SystemExit(
            f"XGBoost版本不符合冻结协议: installed={installed}, "
            f"required={XGBOOST_VERSION}"
        )
    return installed


@lru_cache(maxsize=1)
def frozen_identity() -> dict[str, object]:
    paths = (
        Path("scripts/multitarget_forecasting/config.py"),
        Path("scripts/multitarget_forecasting/data.py"),
        Path("scripts/multitarget_forecasting/xgboost_baseline.py"),
    )
    return {
        **io.data_identity(),
        "code_sha256": {
            str(path): io.file_sha256(path) for path in paths
        },
    }


def expected_metadata(
    station: str, target_mode: str, seed: int, device: str
) -> dict[str, object]:
    return {
        "experiment": EXPERIMENT_ID,
        "kind": "same_sample_five_target_xgboost_validation_prediction",
        "station": station,
        "model": "xgboost",
        "target_mode": target_mode,
        "context": CONTEXT,
        "seed": int(seed),
        "targets": list(config.TARGETS),
        "horizon_hours": list(config.HORIZON_HOURS),
        "train_end": config.TRAIN_END,
        "validation_end": config.VAL_END,
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "xgboost_version": require_xgboost(),
        "n_estimators_per_output": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_alpha": REG_ALPHA,
        "reg_lambda": REG_LAMBDA,
        "tree_method": "hist",
        "device_type": device,
        "n_jobs": N_JOBS,
        **frozen_identity(),
    }


def load_exact(path: Path, expected: dict[str, object]):
    if not path.exists():
        return None
    arrays, actual = io.load_archive(path)
    if actual != expected:
        raise RuntimeError(
            f"已有XGBoost结果与当前协议不一致: {path}。"
            "请审阅后显式使用 --force。"
        )
    return arrays


def prepare_features(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Use exactly the information visible to the 24-hour joint GRU."""

    train_sequence, train_context, val_sequence, val_context, _ = (
        data.prepare_inputs(train, validation, CONTEXT)
    )
    train_x = np.concatenate(
        (train_sequence.reshape(len(train_sequence), -1), train_context), axis=1
    )
    val_x = np.concatenate(
        (val_sequence.reshape(len(val_sequence), -1), val_context), axis=1
    )
    return train_x.astype(np.float32), val_x.astype(np.float32)


def target_values(
    split: dict[str, np.ndarray], target_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    key = "y_abs" if target_mode == "absolute" else "y_delta"
    values = np.asarray(split[key], dtype=float)
    mask = np.asarray(split["y_mask"], dtype=bool) & np.isfinite(values)
    return values, mask


def to_absolute_prediction(
    prediction: np.ndarray,
    current: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    if target_mode == "absolute":
        return prediction
    return prediction + np.asarray(current, dtype=float)[:, None, :]


def _make_model(seed: int, device: str):
    require_xgboost()
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        tree_method="hist",
        device=device,
        n_jobs=N_JOBS,
        random_state=int(seed),
        verbosity=0,
    )


def train_xgboost(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    target_mode: str,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_x, validation_x = prepare_features(train, validation)
    labels, label_mask = target_values(train, target_mode)
    native_prediction = np.full(
        (
            len(validation_x),
            config.OUTPUT_STEPS,
            len(config.TARGETS),
        ),
        np.nan,
        dtype=float,
    )
    training_by_target = np.zeros(len(config.TARGETS), dtype=float)
    inference_by_target = np.zeros(len(config.TARGETS), dtype=float)
    trees_by_target = np.zeros(len(config.TARGETS), dtype=np.int64)
    for target_index, target in enumerate(config.TARGETS):
        console.info(
            "xgboost_target",
            target=target,
            target_mode=config.TARGET_MODE_LABELS[target_mode],
            outputs=config.OUTPUT_STEPS,
        )
        for horizon_index, horizon_hours in enumerate(config.HORIZON_HOURS):
            fit_rows = label_mask[:, horizon_index, target_index]
            if int(fit_rows.sum()) < MIN_TRAIN_ROWS:
                raise ValueError(
                    f"XGBoost训练样本不足: {target}/{horizon_hours}h="
                    f"{int(fit_rows.sum())}"
                )
            model = _make_model(seed, device)
            begin = time.perf_counter()
            model.fit(
                train_x[fit_rows], labels[fit_rows, horizon_index, target_index]
            )
            training_by_target[target_index] += time.perf_counter() - begin
            begin = time.perf_counter()
            native_prediction[:, horizon_index, target_index] = np.asarray(
                model.predict(validation_x), dtype=float
            )
            inference_by_target[target_index] += time.perf_counter() - begin
            trees_by_target[target_index] += len(model.get_booster().get_dump())
            del model
        gc.collect()
    prediction = to_absolute_prediction(
        native_prediction, validation["current"], target_mode
    )
    diagnostics = {
        "training_seconds": np.asarray(training_by_target.sum(), dtype=float),
        "inference_seconds": np.asarray(inference_by_target.sum(), dtype=float),
        "training_seconds_by_target": training_by_target,
        "inference_seconds_by_target": inference_by_target,
        "tree_count": np.asarray(trees_by_target.sum(), dtype=np.int64),
        "tree_count_by_target": trees_by_target,
        "feature_count": np.asarray(train_x.shape[1], dtype=np.int64),
        "fitted_output_count": np.asarray(
            config.OUTPUT_STEPS * len(config.TARGETS), dtype=np.int64
        ),
    }
    return prediction, diagnostics


def run(
    *,
    stations: str | None,
    all_stations: bool,
    target_modes: tuple[str, ...],
    seeds: tuple[int, ...],
    device: str,
    force: bool,
) -> None:
    require_xgboost()
    if device not in DEVICES:
        raise ValueError(f"未知XGBoost设备: {device}")
    panel = data.load_development_panel()
    selected_stations = select_stations(panel, stations, all_stations)
    total = len(selected_stations) * len(target_modes) * len(seeds)
    completed = 0
    for station in selected_stations:
        dataset = data.build_station_dataset(panel, station)
        splits = data.split_by_time(dataset)
        train = splits["train"]
        validation = splits["val"]
        if not len(train["target_start"]) or not len(validation["target_start"]):
            raise ValueError(f"站点缺少训练集或验证集窗口: {station}")
        lower, upper = data.warning_thresholds(train)
        for target_mode in target_modes:
            for seed in seeds:
                completed += 1
                expected = expected_metadata(station, target_mode, seed, device)
                path = prediction_path(station, target_mode, seed, device)
                existing = None if force else load_exact(path, expected)
                if existing is not None:
                    console.info(
                        "resume",
                        progress=f"{completed}/{total}",
                        station=station,
                        target_mode=config.TARGET_MODE_LABELS[target_mode],
                        device=device,
                    )
                    continue
                console.info(
                    "train",
                    progress=f"{completed}/{total}",
                    station=station,
                    model="XGBoost",
                    device=device,
                    target_mode=config.TARGET_MODE_LABELS[target_mode],
                    fitted_outputs=config.OUTPUT_STEPS * len(config.TARGETS),
                )
                prediction, diagnostics = train_xgboost(
                    train,
                    validation,
                    target_mode=target_mode,
                    seed=seed,
                    device=device,
                )
                arrays = {
                    "pred": prediction,
                    "true": np.asarray(validation["y_abs"], dtype=float),
                    "mask": np.asarray(validation["y_mask"], dtype=bool),
                    "current": np.asarray(validation["current"], dtype=float),
                    "target_start": np.asarray(
                        validation["target_start"], dtype="datetime64[ns]"
                    ),
                    "warning_lower": lower,
                    "warning_upper": upper,
                    **diagnostics,
                }
                io.save_archive(path, arrays, expected)
                console.info(
                    "saved",
                    station=station,
                    target_mode=config.TARGET_MODE_LABELS[target_mode],
                    rows=len(prediction),
                    train_s=float(diagnostics["training_seconds"]),
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="25站五指标18时距同样本XGBoost强基线"
    )
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="逗号分隔的站点")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--target-modes", default="absolute,delta")
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        stations=args.stations,
        all_stations=args.all_stations,
        target_modes=_parse_target_modes(args.target_modes),
        seeds=_parse_seeds(args.seeds),
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
