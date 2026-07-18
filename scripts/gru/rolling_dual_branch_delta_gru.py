#!/usr/bin/env python3
"""One-step D model with prediction-fed rolling inference."""

from __future__ import annotations

from scripts.common.terminal_output import console

import numpy as np

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_wentu_diff_delta_gru as delta
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_physical_lag_gru as lag


ONE_STEP = 1


def build_one_step_training_splits(
    data,
    stations: tuple[str, ...],
    input_steps: int,
    input_columns: tuple[str, ...],
    target_feature: str,
):
    """Build one-step delta training data with train-only fitted scalers."""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=input_steps,
        output_steps=ONE_STEP,
        input_columns=input_columns,
        target_columns=(target_feature,),
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    scaled_splits, scalers = lag.scale_physical_lag_splits(raw_splits)
    scaled_splits, current_scaler = dual.attach_scaled_current_level(raw_splits, scaled_splits)
    return raw_splits, scaled_splits, scalers, current_scaler


def prepare_rollout_splits(
    raw_splits: dict[str, dict[str, np.ndarray]],
    input_scaler,
) -> dict[str, dict[str, np.ndarray]]:
    """Scale fixed forecast-origin histories with the one-step training scaler."""
    prepared = {}
    for split_name, split in raw_splits.items():
        sequence_scaled = input_scaler.transform(split["self_x"])
        prepared[split_name] = {
            **split,
            "self_x": lag.prepare_model_inputs(sequence_scaled, split["self_mask"]),
            "y_mask": split["y_mask"].astype(bool),
        }
    return prepared


def make_rollout_loader(torch, split: dict[str, np.ndarray]):
    """Load fixed history, initial state and complete multi-horizon truth."""
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(split["self_x"], dtype=torch.float32),
        torch.as_tensor(split["last_target"], dtype=torch.float32),
        torch.as_tensor(split["y_abs"], dtype=torch.float32),
        torch.as_tensor(split["y_mask"], dtype=torch.bool),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=paper.BATCH_SIZE, shuffle=False)


def _scaler_value(scaler, field: str) -> float:
    values = getattr(scaler, field)
    if values is None or len(values) != 1:
        raise ValueError("rolling D requires a fitted single-target scaler")
    return float(values[0])


def collect_rolling_target_arrays(
    torch,
    model,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    current_scaler,
    output_steps: int,
    target_input_index: int,
    device,
) -> dict[str, np.ndarray]:
    """Feed each prediction to the next current MLP and target-diff history step."""
    model.eval()
    current_mean = _scaler_value(current_scaler, "mean_")
    current_scale = _scaler_value(current_scaler, "scale_")
    delta_mean = _scaler_value(scalers.target_scaler, "mean_")
    delta_scale = _scaler_value(scalers.target_scaler, "scale_")
    input_means = np.asarray(scalers.input_scaler.mean_, dtype=float)
    input_scales = np.asarray(scalers.input_scaler.scale_, dtype=float)
    if target_input_index < 0 or target_input_index >= len(input_means):
        raise IndexError("target_input_index is outside the raw history channels")
    target_input_mean = float(input_means[target_input_index])
    target_input_scale = float(input_scales[target_input_index])
    raw_input_dim = len(input_means)
    loader = make_rollout_loader(torch, split)
    prediction_batches = []
    truth_batches = []
    mask_batches = []

    with torch.no_grad():
        for sequence_x, initial_level, truth, target_mask in loader:
            sequence_x = sequence_x.to(device)
            current_level = initial_level.to(device)
            current_valid = torch.isfinite(current_level)
            current_level = torch.nan_to_num(current_level, nan=0.0, posinf=0.0, neginf=0.0)
            horizon_predictions = []

            for _ in range(output_steps):
                current_scaled = (current_level - current_mean) / current_scale
                current_input = torch.cat(
                    [
                        torch.nan_to_num(current_scaled, nan=0.0, posinf=0.0, neginf=0.0),
                        current_valid.to(dtype=current_scaled.dtype),
                    ],
                    dim=-1,
                )
                delta_scaled = model(sequence_x, current_input)[:, 0]
                delta_raw = delta_scaled * delta_scale + delta_mean
                next_level = current_level + delta_raw
                next_level = torch.where(current_valid, next_level, torch.full_like(next_level, float("nan")))
                horizon_predictions.append(next_level)
                next_history_step = torch.zeros(
                    (*sequence_x.shape[:1], sequence_x.shape[2], sequence_x.shape[3]),
                    dtype=sequence_x.dtype,
                    device=sequence_x.device,
                )
                next_history_step[..., target_input_index] = (
                    delta_raw[..., 0] - target_input_mean
                ) / target_input_scale
                next_history_step[..., raw_input_dim + target_input_index] = current_valid[..., 0].to(
                    dtype=sequence_x.dtype
                )
                sequence_x = torch.cat([sequence_x[:, 1:], next_history_step[:, None]], dim=1)
                current_level = next_level
                current_valid = current_valid & torch.isfinite(current_level)

            prediction = torch.stack(horizon_predictions, dim=1).cpu().numpy()
            truth_np = truth.numpy()
            mask_np = target_mask.numpy().astype(bool)
            mask_np &= np.isfinite(prediction) & np.isfinite(truth_np)
            prediction_batches.append(prediction)
            truth_batches.append(truth_np)
            mask_batches.append(mask_np)

    if not prediction_batches:
        return {
            "pred": split["y_abs"][:0],
            "true": split["y_abs"][:0],
            "mask": split["y_mask"][:0],
        }
    return {
        "pred": np.concatenate(prediction_batches),
        "true": np.concatenate(truth_batches),
        "mask": np.concatenate(mask_batches),
    }


def _target_metrics(
    arrays: dict[str, np.ndarray],
    stations: tuple[str, ...],
    target_feature: str,
) -> dict[str, object]:
    return base.masked_error_metrics(
        arrays["pred"] - arrays["true"],
        arrays["mask"],
        (target_feature,),
        stations,
        truth=arrays["true"],
    )


def fit_target_model(
    torch,
    data,
    stations: tuple[str, ...],
    target_feature: str,
    input_steps: int,
    input_columns: tuple[str, ...],
    rollout_raw_splits: dict[str, dict[str, np.ndarray]],
    output_steps: int,
    seed: int,
    device,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    """Train on one-step deltas and select the checkpoint by rolling validation RMSE."""
    torch.manual_seed(int(seed))
    _, one_step_splits, scalers, current_scaler = build_one_step_training_splits(
        data,
        stations,
        input_steps,
        input_columns,
        target_feature,
    )
    rollout_splits = prepare_rollout_splits(rollout_raw_splits, scalers.input_scaler)
    train_loader = dual.make_dual_loader(torch, one_step_splits["train"], shuffle=True)
    model = dual.make_dual_branch_model(
        torch,
        sequence_input_dim=one_step_splits["train"]["self_x"].shape[-1],
        current_input_dim=one_step_splits["train"]["current_level"].shape[-1],
        target_dim=1,
        output_steps=ONE_STEP,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=paper.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=paper.LR_DECAY_FACTOR,
        patience=paper.LR_DECAY_PATIENCE,
    )
    loss_fn = paper.make_loss_fn(torch)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = dual.train_dual_epoch(torch, model, train_loader, optimizer, loss_fn, device)
        val_arrays = collect_rolling_target_arrays(
            torch,
            model,
            rollout_splits["val"],
            scalers,
            current_scaler,
            output_steps,
            input_columns.index(f"{target_feature}_diff1"),
            device,
        )
        val_metrics = _target_metrics(val_arrays, stations, target_feature)
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        improved = val_rmse < best_rmse - paper.MIN_DELTA
        console.print(
            f"rolling_D/{target_feature}/seed={seed} epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_roll_rmse={val_rmse:.6f}",
            flush=True,
        )
        if improved:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"no valid rolling checkpoint for {target_feature}/seed={seed}")
    model.load_state_dict(best_state)
    arrays_by_split = {
        split: collect_rolling_target_arrays(
            torch,
            model,
            rollout_split,
            scalers,
            current_scaler,
            output_steps,
            input_columns.index(f"{target_feature}_diff1"),
            device,
        )
        for split, rollout_split in rollout_splits.items()
    }
    metrics = {
        split: _target_metrics(arrays, stations, target_feature)
        for split, arrays in arrays_by_split.items()
    }
    console.model_result(
        target_feature,
        best_epoch=best_epoch,
        val_rmse=float(metrics["val"]["rmse"]),
        test_rmse=float(metrics["test"]["rmse"]),
    )
    del model
    return {
        "target": target_feature,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "metrics": metrics,
        "input_columns": list(input_columns),
        "rollout": (
            "previous predicted target becomes the next current level and its predicted delta is appended "
            "to history; unknown future non-target channels are masked"
        ),
    }, arrays_by_split
