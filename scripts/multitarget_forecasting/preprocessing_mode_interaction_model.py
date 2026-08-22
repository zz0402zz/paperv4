"""Joint GRU training for one-target mode flips under A/C/D/E preprocessing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data as base_data
from scripts.multitarget_forecasting import head_ablation_model
from scripts.multitarget_forecasting import preprocessing_ablation_data as robust_data
from scripts.multitarget_forecasting import preprocessing_mode_interaction_config as config
from scripts.multitarget_forecasting.model import (
    _make_loader,
    _predict_scaled,
    _synchronize,
    set_seed,
)
from scripts.multitarget_forecasting.preprocessing_component_model import (
    _fit_target_scaler,
    _loss_numpy,
    _train_epoch,
)


@dataclass(frozen=True)
class ModeTargetTransform:
    modes: dict[str, str]
    log_scales: np.ndarray
    log_enabled: bool

    @classmethod
    def fit(
        cls,
        split: dict[str, np.ndarray],
        modes: dict[str, str],
        log_enabled: bool,
    ) -> "ModeTargetTransform":
        scales = np.ones(len(config.TARGETS), dtype=float)
        if log_enabled:
            current = np.asarray(split["current"], dtype=float)
            future = np.asarray(split["y_abs"], dtype=float)
            current_mask = np.asarray(split["current_mask"], dtype=bool)
            future_mask = np.asarray(split["y_mask"], dtype=bool)
            for target in config.LOG_TARGETS:
                index = config.TARGETS.index(target)
                values = np.concatenate(
                    (
                        current[current_mask[:, index], index],
                        future[:, :, index][future_mask[:, :, index]],
                    )
                )
                positive = values[np.isfinite(values) & (values > 0)]
                if len(positive):
                    candidate = float(np.median(positive))
                    scales[index] = candidate if candidate > 0 else 1.0
        return cls(modes=dict(modes), log_scales=scales, log_enabled=log_enabled)

    def _absolute(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        if not self.log_enabled:
            return result
        for target in config.LOG_TARGETS:
            index = config.TARGETS.index(target)
            valid = np.isfinite(result[..., index]) & (result[..., index] >= 0)
            result[..., index] = np.where(
                valid,
                np.log1p(np.maximum(result[..., index], 0.0) / self.log_scales[index]),
                np.nan,
            )
        return result

    def training_values(
        self, split: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        absolute = self._absolute(split["y_abs"])
        current = self._absolute(split["current"])
        values = np.empty_like(absolute)
        for index, target in enumerate(config.TARGETS):
            if self.modes[target] == "delta":
                values[:, :, index] = absolute[:, :, index] - current[:, None, index]
            else:
                values[:, :, index] = absolute[:, :, index]
        mask = np.asarray(split["y_mask"], dtype=bool) & np.isfinite(values)
        return values, mask

    def to_absolute(
        self, transformed_prediction: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        prediction = np.asarray(transformed_prediction, dtype=float).copy()
        transformed_current = self._absolute(current)
        for index, target in enumerate(config.TARGETS):
            if self.modes[target] == "delta":
                prediction[:, :, index] += transformed_current[:, None, index]
            if self.log_enabled and target in config.LOG_TARGETS:
                prediction[:, :, index] = (
                    np.expm1(np.clip(prediction[:, :, index], -30.0, 30.0))
                    * self.log_scales[index]
                )
                prediction[:, :, index] = np.maximum(prediction[:, :, index], 0.0)
        return prediction


def _prepare_inputs(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    spec: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if spec["input_scaler"] == "mean_std":
        if spec["log_targets"]:
            raise ValueError("当前协议不包含均值标准化下的对数输入。")
        return base_data.prepare_inputs(train, evaluation, config.CONTEXT)
    if spec["input_scaler"] == "median_iqr":
        return robust_data.prepare_inputs(
            train,
            evaluation,
            log_targets=bool(spec["log_targets"]),
            quality_aware=False,
        )
    raise ValueError(f"未知输入标准化: {spec['input_scaler']}")


def _select_epoch(
    torch,
    train: dict[str, np.ndarray],
    *,
    preprocessing: str,
    flip: str,
    seed: int,
    device,
) -> tuple[int, dict[str, np.ndarray]]:
    spec = config.PREPROCESSING_SPECS[preprocessing]
    modes = config.flipped_modes(flip)
    fit, internal_validation = base_data.internal_time_split(train)
    if not len(fit["target_start"]) or not len(internal_validation["target_start"]):
        raise ValueError("训练期内部时间切分为空，无法选择轮数。")
    fit_sequence, fit_context, val_sequence, val_context, _ = _prepare_inputs(
        fit, internal_validation, spec
    )
    transformer = ModeTargetTransform.fit(
        fit, modes, bool(spec["log_targets"])
    )
    fit_values, fit_mask = transformer.training_values(fit)
    val_values, val_mask = transformer.training_values(internal_validation)
    scaler = _fit_target_scaler(
        fit_values, fit_mask, str(spec["target_scaler"])
    )
    fit_scaled = np.nan_to_num(
        scaler.transform(fit_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    val_scaled = np.nan_to_num(
        scaler.transform(val_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

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
        train_loss = _train_epoch(
            torch, network, loader, optimizer, device, str(spec["loss"])
        )
        if epoch % base_config.EVALUATION_EVERY:
            continue
        predicted = _predict_scaled(torch, network, val_sequence, val_context, device)
        val_loss = _loss_numpy(predicted, val_scaled, val_mask, str(spec["loss"]))
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
            preprocessing=spec["label"],
            flip=config.flip_label(flip),
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


def train_mode_interaction(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    preprocessing: str,
    flip: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    spec = config.PREPROCESSING_SPECS[preprocessing]
    modes = config.flipped_modes(flip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_epoch, selection_diagnostics = _select_epoch(
        torch,
        train,
        preprocessing=preprocessing,
        flip=flip,
        seed=seed,
        device=device,
    )
    console.info("refit", selected_epoch=selected_epoch, rows=len(train["target_start"]))
    train_sequence, train_context, val_sequence, val_context, _ = _prepare_inputs(
        train, validation, spec
    )
    transformer = ModeTargetTransform.fit(
        train, modes, bool(spec["log_targets"])
    )
    train_values, train_mask = transformer.training_values(train)
    target_scaler = _fit_target_scaler(
        train_values, train_mask, str(spec["target_scaler"])
    )
    train_scaled = np.nan_to_num(
        target_scaler.transform(train_values),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
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
        final_loss = _train_epoch(
            torch, network, loader, optimizer, device, str(spec["loss"])
        )
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
