#!/usr/bin/env python3
"""Train same-protocol delta LSTM and XGBoost 4--72 hour baselines."""

from __future__ import annotations

import argparse
import gc
from importlib import metadata as package_metadata
import time

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_distillation import config, data, io, models
from scripts.tabpfn_distillation.student import (
    _parse_seeds,
    _set_seed,
    masked_mse,
    prepare_inputs,
)
from scripts.tabpfn_distillation.teacher import select_tasks


BASELINE_EXPERIMENT_ID = "same_protocol_long_horizon_baselines_v1"
LSTM_KEY = "delta_lstm"
XGBOOST_KEY = "delta_xgboost"
BASELINE_KEYS = (LSTM_KEY, XGBOOST_KEY)
BASELINE_LABELS = {
    LSTM_KEY: "变化量LSTM",
    XGBOOST_KEY: "变化量XGBoost",
}

LSTM_HIDDEN_SIZE = config.GRU_HIDDEN_SIZE
LSTM_CURRENT_HIDDEN_SIZE = config.GRU_CURRENT_HIDDEN_SIZE
LSTM_BATCH_SIZE = config.GRU_BATCH_SIZE
LSTM_EPOCHS = config.GRU_EPOCHS
LSTM_LEARNING_RATE = config.GRU_LEARNING_RATE

# Frozen without validation early stopping. Stochastic row/column sampling makes
# the five formal seeds meaningful while keeping every fit inside the train split.
XGBOOST_VERSION = "3.2.0"
XGBOOST_N_ESTIMATORS = 400
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.03
XGBOOST_SUBSAMPLE = 0.9
XGBOOST_COLSAMPLE_BYTREE = 0.9
XGBOOST_MIN_CHILD_WEIGHT = 1.0
XGBOOST_REG_ALPHA = 0.0
XGBOOST_REG_LAMBDA = 1.0
XGBOOST_N_JOBS = -1


def baseline_prediction_path(
    model_key: str, seed: int, station: str, target: str
):
    if model_key not in BASELINE_KEYS:
        raise ValueError(f"Unknown baseline: {model_key}")
    filename = "__".join(
        (
            io.safe_filename(BASELINE_LABELS[model_key]),
            f"种子{seed}",
            io.safe_filename(station),
            io.safe_filename(target),
        )
    )
    return (
        config.output_dir_for_split("val")
        / "同协议基线"
        / "预测结果"
        / f"{filename}.npz"
    )


def _require_xgboost() -> str:
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


def _baseline_metadata(
    model_key: str,
    seed: int,
    station: str,
    target: str,
) -> dict[str, object]:
    common: dict[str, object] = {
        "experiment": BASELINE_EXPERIMENT_ID,
        "kind": "same_protocol_validation_prediction",
        "model": model_key,
        "model_label": BASELINE_LABELS[model_key],
        "seed": int(seed),
        "station": station,
        "target": target,
        "target_mode": "delta",
        "input_steps": config.INPUT_STEPS,
        "input_features": list(config.INPUT_FEATURES),
        "horizon_hours": list(config.HORIZON_HOURS),
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "target_policy": "approved_original_observations_only",
        **io.data_identity(),
        "code_sha256": io.code_sha256(
            (
                "config.py",
                "data.py",
                "io.py",
                "models.py",
                "student.py",
                "protocol_baselines.py",
            )
        ),
    }
    if model_key == LSTM_KEY:
        import torch

        common.update(
            {
                "torch_version": package_metadata.version("torch"),
                "device_type": "cuda" if torch.cuda.is_available() else "cpu",
                "hidden_size": LSTM_HIDDEN_SIZE,
                "current_hidden_size": LSTM_CURRENT_HIDDEN_SIZE,
                "batch_size": LSTM_BATCH_SIZE,
                "epochs": LSTM_EPOCHS,
                "learning_rate": LSTM_LEARNING_RATE,
            }
        )
    elif model_key == XGBOOST_KEY:
        common.update(
            {
                "xgboost_version": _require_xgboost(),
                "n_estimators_per_horizon": XGBOOST_N_ESTIMATORS,
                "max_depth": XGBOOST_MAX_DEPTH,
                "learning_rate": XGBOOST_LEARNING_RATE,
                "subsample": XGBOOST_SUBSAMPLE,
                "colsample_bytree": XGBOOST_COLSAMPLE_BYTREE,
                "min_child_weight": XGBOOST_MIN_CHILD_WEIGHT,
                "reg_alpha": XGBOOST_REG_ALPHA,
                "reg_lambda": XGBOOST_REG_LAMBDA,
                "tree_method": "hist",
                "device_type": "cpu",
                "n_jobs": XGBOOST_N_JOBS,
                "validation_early_stopping": False,
            }
        )
    else:
        raise ValueError(f"Unknown baseline: {model_key}")
    return common


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_lstm(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    true_delta = np.asarray(train["y_delta"], dtype=float)
    true_mask = np.asarray(train["y_mask"], dtype=bool) & np.isfinite(true_delta)
    target_scaler = models.MaskedScaler.fit(true_delta, true_mask)
    target_scaled = np.nan_to_num(
        target_scaler.transform(true_delta), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    usable_rows = true_mask.any(axis=1)
    if not usable_rows.any():
        raise ValueError("LSTM没有可用训练标签。")
    train_sequence, train_current, val_sequence, val_current = prepare_inputs(
        train, validation
    )
    _set_seed(torch, seed)

    class LongHorizonLSTM(torch.nn.Module):
        def __init__(self, sequence_dim: int) -> None:
            super().__init__()
            self.sequence_encoder = torch.nn.LSTM(
                input_size=sequence_dim,
                hidden_size=LSTM_HIDDEN_SIZE,
                batch_first=True,
            )
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, LSTM_CURRENT_HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Linear(LSTM_CURRENT_HIDDEN_SIZE, LSTM_HIDDEN_SIZE),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(LSTM_HIDDEN_SIZE * 2, config.OUTPUT_STEPS)

        def forward(self, sequence_x, current_x):
            encoded, _ = self.sequence_encoder(sequence_x)
            current_state = self.current_encoder(current_x)
            return self.head(
                torch.cat((encoded[:, -1, :], current_state), dim=1)
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongHorizonLSTM(train_sequence.shape[-1]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_sequence[usable_rows]),
        torch.as_tensor(train_current[usable_rows]),
        torch.as_tensor(target_scaled[usable_rows]),
        torch.as_tensor(true_mask[usable_rows]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(LSTM_BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    _synchronize(torch, device)
    train_begin = time.perf_counter()
    model.train()
    for _ in range(LSTM_EPOCHS):
        for sequence_x, current_x, truth_y, truth_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(sequence_x.to(device), current_x.to(device))
            loss = masked_mse(
                torch, predicted, truth_y.to(device), truth_mask.to(device)
            )
            loss.backward()
            optimizer.step()
    _synchronize(torch, device)
    training_seconds = time.perf_counter() - train_begin

    model.eval()
    batches = []
    _synchronize(torch, device)
    inference_begin = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, len(val_sequence), LSTM_BATCH_SIZE):
            end = begin + LSTM_BATCH_SIZE
            batches.append(
                model(
                    torch.as_tensor(val_sequence[begin:end], device=device),
                    torch.as_tensor(val_current[begin:end], device=device),
                )
                .cpu()
                .numpy()
            )
    _synchronize(torch, device)
    inference_seconds = time.perf_counter() - inference_begin
    predicted_scaled = (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, config.OUTPUT_STEPS), dtype=float)
    )
    predicted_delta = target_scaler.inverse_transform(predicted_scaled)
    prediction = data.to_absolute(predicted_delta, validation["current"], "delta")
    diagnostics = {
        "training_seconds": np.asarray(training_seconds, dtype=float),
        "inference_seconds": np.asarray(inference_seconds, dtype=float),
        "parameter_count": np.asarray(parameter_count, dtype=np.int64),
        "tree_count": np.asarray(0, dtype=np.int64),
    }
    del model, optimizer, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, diagnostics


def train_xgboost(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _require_xgboost()
    from xgboost import XGBRegressor

    train_x = data.tabpfn_features(train)
    validation_x = data.tabpfn_features(validation)
    medians = models.finite_feature_medians(train_x)
    train_x = models.apply_feature_medians(train_x, medians)
    validation_x = models.apply_feature_medians(validation_x, medians)
    labels = np.asarray(train["y_delta"], dtype=float)
    label_mask = np.asarray(train["y_mask"], dtype=bool)
    predicted_delta = np.full(
        (len(validation_x), config.OUTPUT_STEPS), np.nan, dtype=float
    )
    training_seconds = 0.0
    inference_seconds = 0.0
    tree_count = 0
    for horizon in range(config.OUTPUT_STEPS):
        fit_rows = label_mask[:, horizon] & np.isfinite(labels[:, horizon])
        if int(fit_rows.sum()) < config.MIN_TEACHER_TRAIN_ROWS:
            raise ValueError(
                f"XGBoost训练样本不足: "
                f"{config.HORIZON_HOURS[horizon]}h={int(fit_rows.sum())}"
            )
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=XGBOOST_N_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
            min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
            reg_alpha=XGBOOST_REG_ALPHA,
            reg_lambda=XGBOOST_REG_LAMBDA,
            tree_method="hist",
            device="cpu",
            n_jobs=XGBOOST_N_JOBS,
            random_state=int(seed),
            verbosity=0,
        )
        begin = time.perf_counter()
        model.fit(train_x[fit_rows], labels[fit_rows, horizon])
        training_seconds += time.perf_counter() - begin
        begin = time.perf_counter()
        predicted_delta[:, horizon] = np.asarray(
            model.predict(validation_x), dtype=float
        )
        inference_seconds += time.perf_counter() - begin
        tree_count += len(model.get_booster().get_dump())
        del model
        gc.collect()
    prediction = data.to_absolute(predicted_delta, validation["current"], "delta")
    diagnostics = {
        "training_seconds": np.asarray(training_seconds, dtype=float),
        "inference_seconds": np.asarray(inference_seconds, dtype=float),
        "parameter_count": np.asarray(0, dtype=np.int64),
        "tree_count": np.asarray(tree_count, dtype=np.int64),
    }
    return prediction, diagnostics


def run_task(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    model_key: str,
    seed: int,
    station: str,
    target: str,
    force: bool,
) -> None:
    expected = _baseline_metadata(model_key, seed, station, target)
    path = baseline_prediction_path(model_key, seed, station, target)
    existing = None if force else io.load_exact(path, expected)
    if existing is not None:
        if existing.get("pred", np.empty(0)).shape != (
            len(validation["target_start"]),
            config.OUTPUT_STEPS,
        ):
            raise RuntimeError(f"已有基线预测形状不正确: {path}")
        console.info("resume", model=model_key, seed=seed, status="already complete")
        return
    if model_key == LSTM_KEY:
        prediction, diagnostics = train_lstm(train, validation, seed)
    elif model_key == XGBOOST_KEY:
        prediction, diagnostics = train_xgboost(train, validation, seed)
    else:
        raise ValueError(f"Unknown baseline: {model_key}")
    arrays = data.prediction_arrays(validation, prediction)
    arrays.update(diagnostics)
    io.save_archive(path, arrays, expected)
    console.info(
        "saved baseline",
        model=model_key,
        seed=seed,
        rows=len(prediction),
        train_s=f"{float(diagnostics['training_seconds']):.2f}",
        infer_s=f"{float(diagnostics['inference_seconds']):.2f}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *BASELINE_KEYS), default="all")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets")
    target_group.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, config.STUDENT_SEEDS)))
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
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected_models = BASELINE_KEYS if args.model == "all" else (args.model,)
    total = len(stations) * len(targets)
    current = 0
    for station in stations:
        for target in targets:
            current += 1
            console.phase(f"{station} / {target}", current=current, total=total)
            splits = data.split_by_time(
                data.build_station_target_dataset(panel, station, target)
            )
            for model_key in selected_models:
                for seed in seeds:
                    run_task(
                        splits["train"],
                        splits["val"],
                        model_key=model_key,
                        seed=seed,
                        station=station,
                        target=target,
                        force=args.force,
                    )
    console.done(config.output_dir_for_split("val") / "同协议基线")


if __name__ == "__main__":
    main()
