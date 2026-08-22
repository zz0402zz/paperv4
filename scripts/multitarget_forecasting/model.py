"""One-pass GRU for 18 horizons and five water-quality targets."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config, data


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(torch, sequence_dim: int, context_dim: int):
    class JointFiveTargetGRU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_dim,
                hidden_size=config.GRU_HIDDEN_SIZE,
                batch_first=True,
            )
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(context_dim, config.CONTEXT_HIDDEN_SIZE),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.CONTEXT_HIDDEN_SIZE),
            )
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(
                    config.GRU_HIDDEN_SIZE + config.CONTEXT_HIDDEN_SIZE,
                    config.FUSION_HIDDEN_SIZE,
                ),
                torch.nn.GELU(),
            )
            self.head = torch.nn.Linear(
                config.FUSION_HIDDEN_SIZE,
                config.OUTPUT_STEPS * len(config.TARGETS),
            )

        def forward(self, sequence_x, context_x):
            encoded, _ = self.sequence_encoder(sequence_x)
            context_state = self.context_encoder(context_x)
            fused = self.fusion(
                torch.cat((encoded[:, -1, :], context_state), dim=1)
            )
            return self.head(fused).reshape(
                -1, config.OUTPUT_STEPS, len(config.TARGETS)
            )

    return JointFiveTargetGRU()


def balanced_masked_mse(torch, prediction, target, mask):
    """Give every available target-by-horizon output equal loss weight."""

    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    squared = torch.where(valid, torch.square(prediction - target), 0.0)
    counts = valid.sum(dim=0)
    sums = squared.sum(dim=0)
    available = counts > 0
    per_output = sums / counts.clamp_min(1)
    return (per_output * available).sum() / available.sum().clamp_min(1)


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target_data(
    split: dict[str, np.ndarray], target_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    label_key = "y_abs" if target_mode == "absolute" else "y_delta"
    values = np.asarray(split[label_key], dtype=float)
    mask = np.asarray(split["y_mask"], dtype=bool) & np.isfinite(values)
    return values, mask


def _make_loader(
    torch,
    sequence: np.ndarray,
    context_values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    seed: int,
):
    usable = np.asarray(mask, dtype=bool).any(axis=(1, 2))
    if not usable.any():
        raise ValueError("五指标联合模型没有可用训练标签。")
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(sequence[usable]),
        torch.as_tensor(context_values[usable]),
        torch.as_tensor(target[usable]),
        torch.as_tensor(mask[usable]),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_epoch(torch, network, loader, optimizer, device) -> float:
    network.train()
    losses = []
    for sequence_x, context_x, truth_y, truth_mask in loader:
        optimizer.zero_grad(set_to_none=True)
        predicted = network(sequence_x.to(device), context_x.to(device))
        loss = balanced_masked_mse(
            torch, predicted, truth_y.to(device), truth_mask.to(device)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def _predict_scaled(
    torch,
    network,
    sequence: np.ndarray,
    context_values: np.ndarray,
    device,
) -> np.ndarray:
    network.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(sequence), config.BATCH_SIZE):
            end = start + config.BATCH_SIZE
            batches.append(
                network(
                    torch.as_tensor(sequence[start:end], device=device),
                    torch.as_tensor(context_values[start:end], device=device),
                )
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, config.OUTPUT_STEPS, len(config.TARGETS)), dtype=float)
    )


def balanced_masked_mse_numpy(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
    )
    counts = valid.sum(axis=0)
    squared = np.where(valid, np.square(prediction - target), 0.0).sum(axis=0)
    available = counts > 0
    if not available.any():
        return np.nan
    return float(np.mean(squared[available] / counts[available]))


def _select_epoch(
    torch,
    train: dict[str, np.ndarray],
    *,
    context: str,
    target_mode: str,
    seed: int,
    device,
) -> tuple[int, dict[str, np.ndarray]]:
    fit, internal_validation = data.internal_time_split(train)
    if not len(fit["target_start"]) or not len(internal_validation["target_start"]):
        raise ValueError("训练期内部时间切分为空，无法选择训练轮数。")
    fit_sequence, fit_context, val_sequence, val_context, _ = data.prepare_inputs(
        fit, internal_validation, context
    )
    fit_values, fit_mask = _target_data(fit, target_mode)
    val_values, val_mask = _target_data(internal_validation, target_mode)
    scaler = data.Standardizer.fit(fit_values, fit_mask)
    fit_scaled = np.nan_to_num(
        scaler.transform(fit_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    val_scaled = np.nan_to_num(
        scaler.transform(val_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    set_seed(torch, seed)
    network = build_model(
        torch, sequence_dim=fit_sequence.shape[-1], context_dim=fit_context.shape[-1]
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    loader = _make_loader(
        torch, fit_sequence, fit_context, fit_scaled, fit_mask, seed
    )
    curve_epochs = []
    curve_train_loss = []
    curve_val_loss = []
    best_epoch = config.EVALUATION_EVERY
    best_loss = np.inf
    stale_evaluations = 0

    _synchronize(torch, device)
    begin = time.perf_counter()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = _train_epoch(torch, network, loader, optimizer, device)
        should_evaluate = epoch % config.EVALUATION_EVERY == 0
        if not should_evaluate:
            continue
        predicted = _predict_scaled(
            torch, network, val_sequence, val_context, device
        )
        val_loss = balanced_masked_mse_numpy(predicted, val_scaled, val_mask)
        curve_epochs.append(epoch)
        curve_train_loss.append(train_loss)
        curve_val_loss.append(val_loss)
        improved = np.isfinite(val_loss) and (
            val_loss < best_loss - config.EARLY_STOPPING_MIN_DELTA
        )
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        console.info(
            "internal_validation",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
        )
        if (
            epoch >= config.MIN_EPOCHS
            and stale_evaluations >= config.EARLY_STOPPING_PATIENCE
        ):
            break
    _synchronize(torch, device)
    selection_seconds = time.perf_counter() - begin
    if not np.isfinite(best_loss):
        raise ValueError("训练期内部验证没有可用标签，无法选择训练轮数。")
    diagnostics = {
        "selection_epochs": np.asarray(curve_epochs, dtype=np.int64),
        "selection_train_loss": np.asarray(curve_train_loss, dtype=float),
        "selection_val_loss": np.asarray(curve_val_loss, dtype=float),
        "selected_epoch": np.asarray(best_epoch, dtype=np.int64),
        "best_internal_val_loss": np.asarray(best_loss, dtype=float),
        "selection_training_seconds": np.asarray(selection_seconds, dtype=float),
        "internal_fit_rows": np.asarray(len(fit_sequence), dtype=np.int64),
        "internal_validation_rows": np.asarray(len(val_sequence), dtype=np.int64),
    }
    del network, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, diagnostics


def train_joint_gru(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    context: str,
    target_mode: str,
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    if target_mode not in config.TARGET_MODES:
        raise ValueError(f"未知输出表示: {target_mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_epoch, selection_diagnostics = _select_epoch(
        torch,
        train,
        context=context,
        target_mode=target_mode,
        seed=seed,
        device=device,
    )
    console.info("refit", selected_epoch=selected_epoch, rows=len(train["target_start"]))

    train_sequence, train_context, val_sequence, val_context, input_scalers = (
        data.prepare_inputs(train, validation, context)
    )
    true_values, true_mask = _target_data(train, target_mode)
    target_scaler = data.Standardizer.fit(true_values, true_mask)
    target_scaled = np.nan_to_num(
        target_scaler.transform(true_values), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    set_seed(torch, seed)
    network = build_model(
        torch, sequence_dim=train_sequence.shape[-1], context_dim=train_context.shape[-1]
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    loader = _make_loader(
        torch, train_sequence, train_context, target_scaled, true_mask, seed
    )

    _synchronize(torch, device)
    begin = time.perf_counter()
    final_loss = np.nan
    for epoch in range(1, selected_epoch + 1):
        final_loss = _train_epoch(torch, network, loader, optimizer, device)
        if epoch == 1 or epoch == selected_epoch or epoch % 50 == 0:
            console.info(
                "refit_epoch",
                epoch=epoch,
                total=selected_epoch,
                loss=final_loss,
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
    predicted_values = target_scaler.inverse_transform(predicted_scaled)
    prediction = (
        predicted_values
        if target_mode == "absolute"
        else predicted_values + np.asarray(validation["current"])[:, None, :]
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": network.state_dict(),
            "experiment": config.EXPERIMENT_ID,
            "context": context,
            "target_mode": target_mode,
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
