"""Shared-linear-head GRU with robust, log, and quality-aware preprocessing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data as base_data
from scripts.multitarget_forecasting import head_ablation_model
from scripts.multitarget_forecasting import preprocessing_ablation_config as config
from scripts.multitarget_forecasting import preprocessing_ablation_data as prep_data
from scripts.multitarget_forecasting.model import (
    _make_loader,
    _predict_scaled,
    _synchronize,
    set_seed,
)


def balanced_masked_huber(torch, prediction, target, mask, delta: float):
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    absolute = torch.abs(prediction - target)
    loss = torch.where(
        absolute <= delta,
        0.5 * torch.square(absolute),
        delta * (absolute - 0.5 * delta),
    )
    loss = torch.where(valid, loss, 0.0)
    counts = valid.sum(dim=0)
    available = counts > 0
    per_output = loss.sum(dim=0) / counts.clamp_min(1)
    return (per_output * available).sum() / available.sum().clamp_min(1)


def balanced_masked_huber_numpy(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, delta: float
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    absolute = np.abs(prediction - target)
    loss = np.where(
        absolute <= delta,
        0.5 * np.square(absolute),
        delta * (absolute - 0.5 * delta),
    )
    counts = valid.sum(axis=0)
    sums = np.where(valid, loss, 0.0).sum(axis=0)
    available = counts > 0
    return float(np.mean(sums[available] / counts[available])) if available.any() else np.nan


def _train_epoch(torch, network, loader, optimizer, device) -> float:
    network.train()
    losses: list[float] = []
    for sequence_x, context_x, truth_y, truth_mask in loader:
        optimizer.zero_grad(set_to_none=True)
        predicted = network(sequence_x.to(device), context_x.to(device))
        loss = balanced_masked_huber(
            torch,
            predicted,
            truth_y.to(device),
            truth_mask.to(device),
            config.HUBER_DELTA,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def _select_epoch(
    torch,
    train: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    device,
) -> tuple[int, dict[str, np.ndarray]]:
    spec = config.VARIANT_SPECS[variant]
    fit, internal_validation = base_data.internal_time_split(train)
    if not len(fit["target_start"]) or not len(internal_validation["target_start"]):
        raise ValueError("训练期内部时间切分为空，无法选择轮数。")
    fit_sequence, fit_context, val_sequence, val_context, _ = prep_data.prepare_inputs(
        fit,
        internal_validation,
        log_targets=bool(spec["log_targets"]),
        quality_aware=bool(spec["quality_aware"]),
    )
    transformer = prep_data.TargetTransform.fit(fit, bool(spec["log_targets"]))
    fit_values, fit_mask = transformer.training_values(fit, bool(spec["quality_aware"]))
    val_values, val_mask = transformer.training_values(
        internal_validation, bool(spec["quality_aware"])
    )
    scaler = prep_data.RobustScaler.fit(fit_values, fit_mask)
    fit_scaled = np.nan_to_num(scaler.transform(fit_values), nan=0.0).astype(np.float32)
    val_scaled = np.nan_to_num(scaler.transform(val_values), nan=0.0).astype(np.float32)

    set_seed(torch, seed)
    network = head_ablation_model.build_model(
        torch, fit_sequence.shape[-1], fit_context.shape[-1], "mixed_linear"
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=base_config.LEARNING_RATE)
    loader = _make_loader(torch, fit_sequence, fit_context, fit_scaled, fit_mask, seed)
    curve_epochs: list[int] = []
    curve_train_loss: list[float] = []
    curve_val_loss: list[float] = []
    best_epoch = base_config.EVALUATION_EVERY
    best_loss = np.inf
    stale_evaluations = 0

    _synchronize(torch, device)
    begin = time.perf_counter()
    for epoch in range(1, base_config.MAX_EPOCHS + 1):
        train_loss = _train_epoch(torch, network, loader, optimizer, device)
        if epoch % base_config.EVALUATION_EVERY:
            continue
        predicted = _predict_scaled(torch, network, val_sequence, val_context, device)
        val_loss = balanced_masked_huber_numpy(
            predicted, val_scaled, val_mask, config.HUBER_DELTA
        )
        curve_epochs.append(epoch)
        curve_train_loss.append(train_loss)
        curve_val_loss.append(val_loss)
        improved = np.isfinite(val_loss) and (
            val_loss < best_loss - base_config.EARLY_STOPPING_MIN_DELTA
        )
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        console.info(
            "internal_validation",
            preprocessing=config.VARIANT_LABELS[variant],
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
        )
        if (
            epoch >= base_config.MIN_EPOCHS
            and stale_evaluations >= base_config.EARLY_STOPPING_PATIENCE
        ):
            break
    _synchronize(torch, device)
    selection_seconds = time.perf_counter() - begin
    if not np.isfinite(best_loss):
        raise ValueError("训练期内部验证没有可用标签。")
    diagnostics = {
        "selection_epochs": np.asarray(curve_epochs, dtype=np.int64),
        "selection_train_loss": np.asarray(curve_train_loss, dtype=float),
        "selection_val_loss": np.asarray(curve_val_loss, dtype=float),
        "selected_epoch": np.asarray(best_epoch, dtype=np.int64),
        "best_internal_val_loss": np.asarray(best_loss, dtype=float),
        "selection_training_seconds": np.asarray(selection_seconds, dtype=float),
    }
    del network, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, diagnostics


def train_preprocessing_ablation(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    if variant not in config.VARIANTS:
        raise ValueError(f"未知预处理变体: {variant}")
    spec = config.VARIANT_SPECS[variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_epoch, selection_diagnostics = _select_epoch(
        torch, train, variant=variant, seed=seed, device=device
    )
    console.info("refit", selected_epoch=selected_epoch, rows=len(train["target_start"]))
    train_sequence, train_context, val_sequence, val_context, input_scalers = (
        prep_data.prepare_inputs(
            train,
            validation,
            log_targets=bool(spec["log_targets"]),
            quality_aware=bool(spec["quality_aware"]),
        )
    )
    transformer = prep_data.TargetTransform.fit(train, bool(spec["log_targets"]))
    train_values, train_mask = transformer.training_values(
        train, bool(spec["quality_aware"])
    )
    target_scaler = prep_data.RobustScaler.fit(train_values, train_mask)
    train_scaled = np.nan_to_num(
        target_scaler.transform(train_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    set_seed(torch, seed)
    network = head_ablation_model.build_model(
        torch, train_sequence.shape[-1], train_context.shape[-1], "mixed_linear"
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=base_config.LEARNING_RATE)
    loader = _make_loader(
        torch, train_sequence, train_context, train_scaled, train_mask, seed
    )
    _synchronize(torch, device)
    begin = time.perf_counter()
    final_loss = np.nan
    for epoch in range(1, selected_epoch + 1):
        final_loss = _train_epoch(torch, network, loader, optimizer, device)
        if epoch == 1 or epoch == selected_epoch or epoch % 50 == 0:
            console.info("refit_epoch", epoch=epoch, total=selected_epoch, loss=final_loss)
    _synchronize(torch, device)
    refit_seconds = time.perf_counter() - begin

    _synchronize(torch, device)
    begin = time.perf_counter()
    predicted_scaled = _predict_scaled(torch, network, val_sequence, val_context, device)
    _synchronize(torch, device)
    inference_seconds = time.perf_counter() - begin
    predicted_transformed = target_scaler.inverse_transform(predicted_scaled)
    prediction = transformer.to_absolute(predicted_transformed, validation["current"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": network.state_dict(),
            "experiment": config.EXPERIMENT_ID,
            "variant": variant,
            "variant_spec": dict(spec),
            "context": config.CONTEXT,
            "target_output_modes": config.TARGET_OUTPUT_MODES,
            "seed": int(seed),
            "sequence_dim": int(train_sequence.shape[-1]),
            "context_dim": int(train_context.shape[-1]),
            "targets": list(config.TARGETS),
            "horizon_hours": list(config.HORIZON_HOURS),
            "selected_epoch": int(selected_epoch),
            "target_center": torch.as_tensor(target_scaler.center),
            "target_scale": torch.as_tensor(target_scaler.scale),
            "target_log_scales": torch.as_tensor(transformer.log_scales),
            "input_scalers": {
                name: torch.as_tensor(values) for name, values in input_scalers.items()
            },
        },
        temporary,
    )
    temporary.replace(model_path)
    diagnostics = {
        "training_seconds": np.asarray(
            float(selection_diagnostics["selection_training_seconds"]) + refit_seconds,
            dtype=float,
        ),
        "refit_training_seconds": np.asarray(refit_seconds, dtype=float),
        "inference_seconds": np.asarray(inference_seconds, dtype=float),
        "parameter_count": np.asarray(
            sum(parameter.numel() for parameter in network.parameters()), dtype=np.int64
        ),
        "final_training_loss": np.asarray(final_loss, dtype=float),
        "target_scale": np.asarray(target_scaler.scale, dtype=float),
        "target_log_scales": np.asarray(transformer.log_scales, dtype=float),
        "training_label_cells": np.asarray(train_mask.sum(), dtype=np.int64),
        **selection_diagnostics,
    }
    return prediction, diagnostics
