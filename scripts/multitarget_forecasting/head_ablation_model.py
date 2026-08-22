"""Parameter-controlled shared-head versus target-specific-head models."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data
from scripts.multitarget_forecasting import head_ablation_config as config
from scripts.multitarget_forecasting.model import (
    _make_loader,
    _predict_scaled,
    _synchronize,
    _train_epoch,
    balanced_masked_mse_numpy,
    set_seed,
)


def build_model(torch, sequence_dim: int, context_dim: int, variant: str):
    if variant not in config.VARIANTS:
        raise ValueError(f"未知预测头变体: {variant}")

    class HeadAblationGRU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_dim,
                hidden_size=base_config.GRU_HIDDEN_SIZE,
                batch_first=True,
            )
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(context_dim, base_config.CONTEXT_HIDDEN_SIZE),
                torch.nn.GELU(),
                torch.nn.LayerNorm(base_config.CONTEXT_HIDDEN_SIZE),
            )
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(
                    base_config.GRU_HIDDEN_SIZE
                    + base_config.CONTEXT_HIDDEN_SIZE,
                    base_config.FUSION_HIDDEN_SIZE,
                ),
                torch.nn.GELU(),
            )
            if variant == "mixed_linear":
                self.output_head = torch.nn.Linear(
                    base_config.FUSION_HIDDEN_SIZE,
                    config.OUTPUT_STEPS * len(config.TARGETS),
                )
            elif variant == "mixed_shared_mlp":
                self.output_head = torch.nn.Sequential(
                    torch.nn.Linear(
                        base_config.FUSION_HIDDEN_SIZE,
                        config.SHARED_MLP_HIDDEN_SIZE,
                    ),
                    torch.nn.GELU(),
                    torch.nn.Linear(
                        config.SHARED_MLP_HIDDEN_SIZE,
                        config.OUTPUT_STEPS * len(config.TARGETS),
                    ),
                )
            else:
                self.output_head = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(
                                base_config.FUSION_HIDDEN_SIZE,
                                config.TARGET_HEAD_HIDDEN_SIZE,
                            ),
                            torch.nn.GELU(),
                            torch.nn.Linear(
                                config.TARGET_HEAD_HIDDEN_SIZE,
                                config.OUTPUT_STEPS,
                            ),
                        )
                        for _ in config.TARGETS
                    ]
                )

        def forward(self, sequence_x, context_x):
            encoded, _ = self.sequence_encoder(sequence_x)
            context_state = self.context_encoder(context_x)
            fused = self.fusion(
                torch.cat((encoded[:, -1, :], context_state), dim=1)
            )
            if variant == "mixed_target_heads":
                return torch.stack(
                    [head(fused) for head in self.output_head], dim=2
                )
            return self.output_head(fused).reshape(
                -1, config.OUTPUT_STEPS, len(config.TARGETS)
            )

    return HeadAblationGRU()


def mixed_targets(
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.asarray(split["y_abs"], dtype=float)
    delta = np.asarray(split["y_delta"], dtype=float)
    values = np.empty_like(absolute)
    for target_index, target in enumerate(config.TARGETS):
        source = delta if config.TARGET_OUTPUT_MODES[target] == "delta" else absolute
        values[:, :, target_index] = source[:, :, target_index]
    mask = np.asarray(split["y_mask"], dtype=bool) & np.isfinite(values)
    return values, mask


def to_absolute_prediction(
    mixed_prediction: np.ndarray, current: np.ndarray
) -> np.ndarray:
    prediction = np.asarray(mixed_prediction, dtype=float).copy()
    current = np.asarray(current, dtype=float)
    for target_index, target in enumerate(config.TARGETS):
        if config.TARGET_OUTPUT_MODES[target] == "delta":
            prediction[:, :, target_index] += current[:, None, target_index]
    return prediction


def _select_epoch(
    torch,
    train: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    device,
) -> tuple[int, dict[str, np.ndarray]]:
    fit, internal_validation = data.internal_time_split(train)
    if not len(fit["target_start"]) or not len(
        internal_validation["target_start"]
    ):
        raise ValueError("训练期内部时间切分为空，无法选择训练轮数。")
    fit_sequence, fit_context, val_sequence, val_context, _ = data.prepare_inputs(
        fit, internal_validation, config.CONTEXT
    )
    fit_values, fit_mask = mixed_targets(fit)
    val_values, val_mask = mixed_targets(internal_validation)
    scaler = data.Standardizer.fit(fit_values, fit_mask)
    fit_scaled = np.nan_to_num(
        scaler.transform(fit_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    val_scaled = np.nan_to_num(
        scaler.transform(val_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    set_seed(torch, seed)
    network = build_model(
        torch, fit_sequence.shape[-1], fit_context.shape[-1], variant
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=base_config.LEARNING_RATE)
    loader = _make_loader(
        torch, fit_sequence, fit_context, fit_scaled, fit_mask, seed
    )
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
        predicted = _predict_scaled(
            torch, network, val_sequence, val_context, device
        )
        val_loss = balanced_masked_mse_numpy(predicted, val_scaled, val_mask)
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
            variant=config.VARIANT_LABELS[variant],
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


def train_head_ablation(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    variant: str,
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_epoch, selection_diagnostics = _select_epoch(
        torch, train, variant=variant, seed=seed, device=device
    )
    console.info("refit", selected_epoch=selected_epoch, rows=len(train["target_start"]))
    train_sequence, train_context, val_sequence, val_context, input_scalers = (
        data.prepare_inputs(train, validation, config.CONTEXT)
    )
    train_values, train_mask = mixed_targets(train)
    target_scaler = data.Standardizer.fit(train_values, train_mask)
    train_scaled = np.nan_to_num(
        target_scaler.transform(train_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    set_seed(torch, seed)
    network = build_model(
        torch, train_sequence.shape[-1], train_context.shape[-1], variant
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
            console.info(
                "refit_epoch", epoch=epoch, total=selected_epoch, loss=final_loss
            )
    _synchronize(torch, device)
    refit_seconds = time.perf_counter() - begin

    _synchronize(torch, device)
    begin = time.perf_counter()
    predicted_scaled = _predict_scaled(
        torch, network, val_sequence, val_context, device
    )
    _synchronize(torch, device)
    inference_seconds = time.perf_counter() - begin
    predicted_mixed = target_scaler.inverse_transform(predicted_scaled)
    prediction = to_absolute_prediction(predicted_mixed, validation["current"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": network.state_dict(),
            "experiment": config.EXPERIMENT_ID,
            "variant": variant,
            "context": config.CONTEXT,
            "target_output_modes": config.TARGET_OUTPUT_MODES,
            "seed": int(seed),
            "sequence_dim": int(train_sequence.shape[-1]),
            "context_dim": int(train_context.shape[-1]),
            "targets": list(config.TARGETS),
            "horizon_hours": list(config.HORIZON_HOURS),
            "selected_epoch": int(selected_epoch),
            "target_mean": torch.as_tensor(target_scaler.mean),
            "target_scale": torch.as_tensor(target_scaler.scale),
            "input_scalers": {
                name: torch.as_tensor(values)
                for name, values in input_scalers.items()
            },
        },
        temporary,
    )
    temporary.replace(model_path)
    diagnostics = {
        "training_seconds": np.asarray(
            float(selection_diagnostics["selection_training_seconds"])
            + refit_seconds,
            dtype=float,
        ),
        "refit_training_seconds": np.asarray(refit_seconds, dtype=float),
        "inference_seconds": np.asarray(inference_seconds, dtype=float),
        "parameter_count": np.asarray(
            sum(parameter.numel() for parameter in network.parameters()),
            dtype=np.int64,
        ),
        "final_training_loss": np.asarray(final_loss, dtype=float),
        "target_scale": np.asarray(target_scaler.scale, dtype=float),
        **selection_diagnostics,
    }
    return prediction, diagnostics
