#!/usr/bin/env python3
"""Reusable models and controls for the 24h history-branch ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

from dataclasses import dataclass

import numpy as np

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_wentu_diff_delta_gru as delta
from scripts.gru import run_wentu_dual_branch_delta_gru as dual


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    model_mode: str
    history_control: str = "observed"


NEURAL_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("current_only_mlp", "Current only MLP", "current_only"),
    VariantSpec("diff_only_gru", "Diff history only GRU", "diff_only"),
    VariantSpec("full_D", "Full D", "full"),
    VariantSpec("full_D_zero_history", "Full D with zero history", "full", "zero"),
    VariantSpec("full_D_mismatched_history", "Full D with mismatched history", "full", "mismatched"),
)


def controlled_splits(
    scaled_splits: dict[str, dict[str, np.ndarray]],
    control: str,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Apply a split-local history negative control without touching labels/current state."""
    if control not in {"observed", "zero", "mismatched"}:
        raise ValueError(f"Unknown history control: {control}")
    if control == "observed":
        return scaled_splits, {}

    output: dict[str, dict[str, np.ndarray]] = {}
    shifts: dict[str, int] = {}
    split_offsets = {"train": 101, "val": 211, "test": 307}
    for split_name, split in scaled_splits.items():
        sequence = split["self_x"]
        if control == "zero":
            controlled = np.zeros_like(sequence)
            shifts[split_name] = 0
        else:
            count = len(sequence)
            if count < 2:
                raise ValueError(f"Need at least two {split_name} windows for mismatching.")
            rng = np.random.default_rng(int(seed) + split_offsets[split_name])
            shift = int(rng.integers(1, count))
            controlled = np.roll(sequence, shift=shift, axis=0).copy()
            shifts[split_name] = shift
        output[split_name] = {**split, "self_x": controlled}
    return output, shifts


def make_model(
    torch,
    mode: str,
    sequence_input_dim: int,
    current_input_dim: int,
    output_steps: int,
):
    """Build one branch variant while keeping the shared hidden width fixed."""
    if mode == "full":
        return dual.make_dual_branch_model(
            torch,
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            target_dim=1,
            output_steps=output_steps,
        )
    if mode not in {"current_only", "diff_only"}:
        raise ValueError(f"Unknown model mode: {mode}")

    hidden_size = paper.HIDDEN_SIZE

    class HistoryBranchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = paper.GRU_DROPOUT if paper.NUM_GRU_LAYERS > 1 else 0.0
            if mode == "diff_only":
                self.sequence_encoder = torch.nn.GRU(
                    input_size=sequence_input_dim,
                    hidden_size=hidden_size,
                    num_layers=paper.NUM_GRU_LAYERS,
                    batch_first=True,
                    dropout=recurrent_dropout,
                )
                head_input_dim = hidden_size
            else:
                self.current_encoder = torch.nn.Sequential(
                    torch.nn.Linear(current_input_dim, dual.CURRENT_HIDDEN_SIZE),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dual.CURRENT_HIDDEN_SIZE, hidden_size),
                    torch.nn.ReLU(),
                )
                head_input_dim = hidden_size
            self.dropout = torch.nn.Dropout(paper.HEAD_DROPOUT)
            self.head = torch.nn.Linear(head_input_dim, output_steps)

        def forward(self, sequence_x, current_level):
            batch_size, steps, node_count, _ = sequence_x.shape
            if mode == "diff_only":
                encoded_input = sequence_x.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, -1)
                encoded, _ = self.sequence_encoder(encoded_input)
                state = encoded[:, -1, :].reshape(batch_size, node_count, hidden_size)
            else:
                state = self.current_encoder(current_level)
            prediction = self.head(self.dropout(state)).reshape(batch_size, node_count, output_steps, 1)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return HistoryBranchModel()


def _metric_from_arrays(
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
    variant: VariantSpec,
    target_feature: str,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    output_steps: int,
    seed: int,
    device,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]], list[dict[str, object]], dict[str, int]]:
    """Train one target/variant and retain the validation-selected state in memory."""
    torch.manual_seed(int(seed))
    controlled, shifts = controlled_splits(scaled_splits, variant.history_control, seed)
    train_loader = dual.make_dual_loader(torch, controlled["train"], shuffle=True)
    eval_loaders = {
        split: dual.make_dual_loader(torch, values, shuffle=False)
        for split, values in controlled.items()
    }
    model = make_model(
        torch,
        variant.model_mode,
        sequence_input_dim=controlled["train"]["self_x"].shape[-1],
        current_input_dim=controlled["train"]["current_level"].shape[-1],
        output_steps=output_steps,
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
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = dual.train_dual_epoch(torch, model, train_loader, optimizer, loss_fn, device)
        val_arrays = dual.collect_dual_target_arrays(
            torch,
            model,
            eval_loaders["val"],
            controlled["val"],
            scalers,
            device,
        )
        val_metrics = _metric_from_arrays(val_arrays, stations, target_feature)
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        improved = val_rmse < best_rmse - paper.MIN_DELTA
        history.append(
            {
                "epoch": epoch,
                "target": target_feature,
                "variant": variant.key,
                "seed": int(seed),
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
            }
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
        raise RuntimeError(f"No valid checkpoint for {variant.key}/{target_feature}/seed={seed}")
    model.load_state_dict(best_state)
    arrays_by_split = {
        split: dual.collect_dual_target_arrays(
            torch,
            model,
            eval_loaders[split],
            controlled[split],
            scalers,
            device,
        )
        for split in ("train", "val", "test")
    }
    metrics = {
        split: _metric_from_arrays(arrays, stations, target_feature)
        for split, arrays in arrays_by_split.items()
    }
    result = {
        "variant": variant.key,
        "target": target_feature,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "metrics": metrics,
    }
    console.print(
        f"{variant.key}/{target_feature}/seed={seed}: epoch={best_epoch} "
        f"val_rmse={metrics['val']['rmse']:.6f} test_rmse={metrics['test']['rmse']:.6f}",
        flush=True,
    )
    del model
    return result, arrays_by_split, history, shifts


def persistence_arrays(split: dict[str, np.ndarray], output_steps: int) -> dict[str, np.ndarray]:
    """Repeat the forecast-origin target level across all direct horizons."""
    prediction = np.repeat(split["last_target"][:, None, :, :], output_steps, axis=1)
    truth = split["y_abs"]
    mask = split["y_mask"].astype(bool) & np.isfinite(prediction) & np.isfinite(truth)
    return {"pred": prediction, "true": truth, "mask": mask}


def _ridge_design(split: dict[str, np.ndarray]) -> np.ndarray:
    sequence = split["self_x"]
    windows, steps, nodes, channels = sequence.shape
    flattened = sequence.transpose(0, 2, 1, 3).reshape(windows * nodes, steps * channels)
    current = split["current_level"].reshape(windows * nodes, -1)
    return np.concatenate([flattened, current], axis=1).astype(np.float64)


def ridge_fit_predict_scaled(
    train_split: dict[str, np.ndarray],
    target_split: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit one shared closed-form Ridge head for every direct horizon."""
    x_train = _ridge_design(train_split)
    x_target = _ridge_design(target_split)
    train_windows, output_steps, nodes, _ = train_split["y"].shape
    y_train = train_split["y"].transpose(0, 2, 1, 3).reshape(train_windows * nodes, output_steps)
    mask_train = train_split["y_mask"].transpose(0, 2, 1, 3).reshape(train_windows * nodes, output_steps)
    x_train_aug = np.concatenate([np.ones((len(x_train), 1)), x_train], axis=1)
    x_target_aug = np.concatenate([np.ones((len(x_target), 1)), x_target], axis=1)
    regularizer = np.eye(x_train_aug.shape[1], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    predictions = np.zeros((len(x_target), output_steps), dtype=np.float64)
    for horizon in range(output_steps):
        valid = mask_train[:, horizon].astype(bool) & np.isfinite(y_train[:, horizon])
        design = x_train_aug[valid]
        target = y_train[valid, horizon]
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
        predictions[:, horizon] = x_target_aug @ coefficients
    target_windows = len(target_split["y"])
    return predictions.reshape(target_windows, nodes, output_steps, 1).transpose(0, 2, 1, 3).astype(np.float32)


def ridge_arrays(
    train_split: dict[str, np.ndarray],
    target_split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Restore Ridge delta outputs to the absolute target scale."""
    prediction_scaled = ridge_fit_predict_scaled(train_split, target_split, alpha=alpha)
    prediction_delta = scalers.inverse_transform_target(prediction_scaled)
    prediction = delta.restore_absolute_from_delta(prediction_delta, target_split["last_target"])
    truth = target_split["y_abs"]
    mask = target_split["y_mask"].astype(bool) & np.isfinite(prediction) & np.isfinite(truth)
    return {"pred": prediction, "true": truth, "mask": mask}
