#!/usr/bin/env python3
"""Retrain the V2 24h-to-36h mainline and selected baselines in memory."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.baselines import literature_baseline_models as literature
from scripts.common import v2_experiment_protocol as protocol
from scripts.common import wq_modeling_common as common
from scripts.common.wq_gru_data import FEATURE_COLUMNS
from scripts.gru import history_branch_ablation as branch
from scripts.gru import rolling_dual_branch_delta_gru as rolling
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_v2_multistep_history_branch_ablation as mainline
from scripts.gru import run_wentu_diff_delta_gru as delta


INPUT_STEPS = 6
OUTPUT_STEPS = 9
INPUT_HOURS = INPUT_STEPS * 4
OUTPUT_HOURS = OUTPUT_STEPS * 4
TARGET_FEATURES = protocol.TARGET_FEATURE_COLUMNS
MODEL_ORDER = ("lstm", "gru", "full_D", "rolling_D", "ridge", "mlp")
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "plain GRU",
    "full_D": "mainline Full D",
    "rolling_D": "rolling D",
    "ridge": "Ridge regression",
    "mlp": "MLP",
}


@dataclass(frozen=True)
class BaselineData:
    """Common raw-history data used by LSTM, GRU, Ridge and MLP."""

    raw_splits: dict[str, dict[str, np.ndarray]]
    scaled_splits: dict[str, dict[str, np.ndarray]]
    samples: dict[str, dict[str, np.ndarray]]
    scalers: base.GraphForecastScalers


def parse_model_names(value: str) -> tuple[str, ...]:
    """Parse a comma-separated subset while preserving the formal display order."""
    aliases = {
        "mainline": "full_D",
        "ours": "full_D",
        "recursive_D": "rolling_D",
        "ridge_lr": "ridge",
    }
    requested = {aliases.get(item.strip(), item.strip()) for item in value.split(",") if item.strip()}
    unknown = requested - set(MODEL_ORDER)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown models: {', '.join(sorted(unknown))}")
    if not requested:
        raise argparse.ArgumentTypeError("at least one model is required")
    return tuple(model for model in MODEL_ORDER if model in requested)


def _absolute_graph_splits(
    delta_splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Use raw histories and absolute targets while retaining the delta task mask."""
    return {
        split_name: {
            "x": split["self_x"],
            "y": split["y_abs"],
            "x_mask": split["self_mask"],
            "y_mask": split["y_mask"],
            "target_start": split["target_start"],
            "target_end": split["target_end"],
        }
        for split_name, split in delta_splits.items()
    }


def prepare_baseline_data(data: pd.DataFrame, stations: tuple[str, ...]) -> BaselineData:
    """Build one leakage-safe 24h-to-36h dataset for every conventional baseline."""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=FEATURE_COLUMNS,
        target_columns=TARGET_FEATURES,
        freq=protocol.RESAMPLE_RULE,
    )
    delta_splits = base.split_graph_by_time(dataset, protocol.TRAIN_END, protocol.VAL_END)
    raw_splits = _absolute_graph_splits(delta_splits)
    scaled_splits, scalers = base.scale_graph_splits(raw_splits, append_mask_features=True)
    samples = {
        split: common.station_samples(scaled_splits[split], len(stations))
        for split in ("train", "val", "test")
    }
    return BaselineData(raw_splits, scaled_splits, samples, scalers)


def make_baseline_model(
    torch,
    name: str,
    input_dim: int,
    station_count: int,
    target_dim: int,
):
    """Build the same ordinary architectures used by the Stage 2 baselines."""
    hidden_size = literature.HIDDEN_SIZE
    embedding_dim = literature.STATION_EMBED_DIM
    dropout = 0.10

    class MlpBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, embedding_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(INPUT_STEPS * input_dim + embedding_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            state = torch.cat([x.reshape(x.shape[0], -1), self.station_embedding(station_id)], dim=-1)
            return self.net(state).reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    class RecurrentBaseline(torch.nn.Module):
        def __init__(self, recurrent_type: str) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, embedding_dim)
            recurrent = torch.nn.GRU if recurrent_type == "gru" else torch.nn.LSTM
            self.encoder = recurrent(input_dim, hidden_size, num_layers=1, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size + embedding_dim, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            output, _ = self.encoder(x)
            state = torch.cat([output[:, -1], self.station_embedding(station_id)], dim=-1)
            return self.head(state).reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    if name == "mlp":
        return MlpBaseline()
    if name in {"gru", "lstm"}:
        return RecurrentBaseline(name)
    raise ValueError(f"unknown neural baseline: {name}")


def _make_baseline_loader(torch, samples: dict[str, np.ndarray], *, shuffle: bool, train: bool):
    return common.make_station_loader(
        torch,
        samples,
        literature.BATCH_SIZE,
        shuffle=shuffle,
        train_only_valid=train,
    )


def _collect_baseline_predictions(torch, model, loader, device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, station_id, _, _ in loader:
            predictions.append(model(x.to(device), station_id.to(device)).cpu().numpy())
    if predictions:
        return np.concatenate(predictions)
    return np.empty((0, OUTPUT_STEPS, len(TARGET_FEATURES)), dtype=np.float32)


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_neural_baseline(
    torch,
    name: str,
    baseline_data: BaselineData,
    station_count: int,
    seed: int,
    max_epochs: int,
    device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Retrain one direct multi-step baseline and keep its best state in memory."""
    _set_seed(torch, seed)
    samples = baseline_data.samples
    model = make_baseline_model(
        torch,
        name,
        input_dim=samples["train"]["x"].shape[-1],
        station_count=station_count,
        target_dim=len(TARGET_FEATURES),
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=literature.LEARNING_RATE)
    train_loader = _make_baseline_loader(torch, samples["train"], shuffle=True, train=True)
    eval_loaders = {
        split: _make_baseline_loader(torch, split_samples, shuffle=False, train=False)
        for split, split_samples in samples.items()
    }
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x, station_id, y, mask in train_loader:
            x = x.to(device)
            station_id = station_id.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = common.masked_l1_loss(torch, model(x, station_id), y, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prediction = _collect_baseline_predictions(torch, model, eval_loaders["val"], device)
        val_rmse = common.scaled_flat_rmse(val_prediction, samples["val"])
        improved = val_rmse < best_val_rmse - literature.MIN_DELTA
        console.print(
            f"{name} seed={seed} epoch={epoch:03d} "
            f"train_loss={float(np.mean(losses)):.6f} val_scaled_rmse={val_rmse:.6f}",
            flush=True,
        )
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= literature.EARLY_STOPPING_PATIENCE:
                console.info("early stop", model=name, seed=seed, epoch=epoch, best_epoch=best_epoch)
                break

    if best_state is None:
        raise RuntimeError(f"no valid checkpoint for {name}/seed={seed}")
    model.load_state_dict(best_state)
    predictions = {
        split: _collect_baseline_predictions(torch, model, loader, device)
        for split, loader in eval_loaders.items()
    }
    del model
    return {
        "best_epoch": best_epoch,
        "best_val_scaled_rmse": best_val_rmse,
        "parameter_count": parameter_count,
    }, predictions


def ridge_fit_predict(
    train_samples: dict[str, np.ndarray],
    eval_samples: dict[str, np.ndarray],
    alpha: float = literature.RIDGE_ALPHA,
) -> np.ndarray:
    """Fit independent closed-form Ridge heads for all 45 horizon-target cells."""
    x_train = common.tabular_features(train_samples)
    x_eval = common.tabular_features(eval_samples)
    y_train = train_samples["y"].astype(np.float64)
    mask_train = train_samples["mask"]
    output_steps, target_dim = y_train.shape[1:]
    y_flat = y_train.reshape(len(y_train), output_steps * target_dim)
    mask_flat = mask_train.reshape(len(mask_train), output_steps * target_dim)
    prediction = np.zeros((len(x_eval), output_steps * target_dim), dtype=np.float32)
    regularizer = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0

    for output_index in range(output_steps * target_dim):
        valid = mask_flat[:, output_index] & np.isfinite(y_flat[:, output_index])
        design = x_train[valid]
        target = y_flat[valid, output_index]
        if len(target) == 0:
            prediction[:, output_index] = np.nan
            continue
        coefficients = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ target,
        )
        prediction[:, output_index] = (x_eval @ coefficients).astype(np.float32)
    return prediction.reshape(len(x_eval), output_steps, target_dim)


def prediction_arrays_by_target(
    predictions: dict[str, np.ndarray],
    baseline_data: BaselineData,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Restore scaled baseline predictions and split them into per-target arrays."""
    output = {split: {} for split in ("train", "val", "test")}
    for split, prediction_scaled in predictions.items():
        samples = baseline_data.samples[split]
        windows = int(samples["windows"])
        stations = int(samples["station_count"])
        prediction_graph_scaled = prediction_scaled.reshape(
            windows, stations, OUTPUT_STEPS, len(TARGET_FEATURES)
        ).transpose(0, 2, 1, 3)
        prediction = baseline_data.scalers.inverse_transform_target(prediction_graph_scaled)
        truth = baseline_data.raw_splits[split]["y"]
        mask = (
            baseline_data.raw_splits[split]["y_mask"].astype(bool)
            & np.isfinite(prediction)
            & np.isfinite(truth)
        )
        for target_index, target in enumerate(TARGET_FEATURES):
            output[split][target] = {
                "pred": prediction[..., target_index : target_index + 1],
                "true": truth[..., target_index : target_index + 1],
                "mask": mask[..., target_index : target_index + 1],
            }
    return output


def run_baseline_model(
    torch,
    model: str,
    baseline_data: BaselineData,
    stations: tuple[str, ...],
    seed: int,
    max_epochs: int,
    device,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Retrain one conventional model and return standard multi-step tables."""
    if model == "ridge":
        predictions = {
            split: ridge_fit_predict(baseline_data.samples["train"], baseline_data.samples[split])
            for split in ("train", "val", "test")
        }
        artifact = {"best_epoch": None, "parameter_count": None, "alpha": literature.RIDGE_ALPHA}
        table_seed = -1
    else:
        artifact, predictions = fit_neural_baseline(
            torch,
            model,
            baseline_data,
            len(stations),
            seed,
            max_epochs,
            device,
        )
        table_seed = seed
    arrays = prediction_arrays_by_target(predictions, baseline_data)
    tables = mainline.aggregate_metric_tables(model, table_seed, arrays, stations)
    return tables, artifact


def run_mainline_model(
    torch,
    prepared: dict[str, mainline.PreparedTarget],
    stations: tuple[str, ...],
    seed: int,
    device,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Retrain the Full D mainline without consulting or writing cached artifacts."""
    variant = next(item for item in branch.NEURAL_VARIANTS if item.key == "full_D")
    _set_seed(torch, seed)
    paper.SEED = seed
    tables, targets, _, _ = mainline.run_neural_variant(
        torch,
        variant,
        seed,
        prepared,
        stations,
        device,
    )
    return tables, {
        "best_epoch": {item["target"]: item["best_epoch"] for item in targets},
        "parameter_count": {item["target"]: item["parameter_count"] for item in targets},
    }


def run_rolling_model(
    torch,
    data: pd.DataFrame,
    prepared: dict[str, mainline.PreparedTarget],
    stations: tuple[str, ...],
    seed: int,
    device,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Retrain one-step D models and roll their predictions through all nine horizons."""
    _set_seed(torch, seed)
    paper.SEED = seed
    arrays = {split: {} for split in ("train", "val", "test")}
    targets = []
    for target, item in prepared.items():
        result, target_arrays = rolling.fit_target_model(
            torch,
            data,
            stations,
            target,
            INPUT_STEPS,
            item.input_columns,
            item.raw_splits,
            OUTPUT_STEPS,
            seed,
            device,
        )
        targets.append(result)
        for split in arrays:
            arrays[split][target] = target_arrays[split]
    tables = mainline.aggregate_metric_tables("rolling_D", seed, arrays, stations)
    return tables, {
        "best_epoch": {item["target"]: item["best_epoch"] for item in targets},
        "parameter_count": {item["target"]: item["parameter_count"] for item in targets},
    }


def validate_model_counts(reference: pd.DataFrame | None, candidate: pd.DataFrame, model: str) -> pd.DataFrame:
    """Require every model and seed to use identical valid test points per horizon."""
    test = candidate[candidate["split"].eq("test")]
    inconsistent = test.groupby("horizon_hours")["valid_points"].nunique()
    if (inconsistent != 1).any():
        raise ValueError(f"{model} has seed-dependent valid-point counts")
    counts = (
        test.groupby("horizon_hours", as_index=False)["valid_points"]
        .first()
        .sort_values("horizon_hours")
        .reset_index(drop=True)
    )
    if reference is not None and not counts.equals(reference):
        comparison = reference.merge(counts, on="horizon_hours", suffixes=("_reference", f"_{model}"))
        raise ValueError(f"unfair valid-point counts for {model}:\n{comparison.to_string(index=False)}")
    return counts


def _metric_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    neural = frame[frame["seed"].ge(0)]
    source = neural if not neural.empty else frame
    grouped = source.groupby(group_columns, sort=True)
    summary = grouped.agg(
        runs=("seed", "nunique"),
        valid_points=("valid_points", "first"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        nse=("nse", "mean"),
        rmse_std=("rmse", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0),
    ).reset_index()
    return summary


def print_model_result(model: str, overall: pd.DataFrame, horizon: pd.DataFrame) -> None:
    """Print one model's overall and horizon results without mixing model columns."""
    label = MODEL_LABELS[model]
    overall_summary = _metric_summary(overall, ["split"])
    test_horizon = _metric_summary(horizon[horizon["split"].eq("test")], ["horizon_hours"])
    for frame in (overall_summary, test_horizon):
        for column in ("mae", "rmse", "nse", "rmse_std"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(6)
    console.table(
        f"{label} overall",
        overall_summary,
        columns=("split", "runs", "valid_points", "mae", "rmse", "rmse_std", "nse"),
        max_rows=3,
    )
    console.table(
        f"{label} test horizons",
        test_horizon,
        columns=("horizon_hours", "runs", "valid_points", "mae", "rmse", "rmse_std", "nse"),
        max_rows=OUTPUT_STEPS,
    )


def run_suite(
    models: tuple[str, ...] = MODEL_ORDER,
    seed: int = protocol.PILOT_SEED,
    formal: bool = False,
    max_epochs: int | None = None,
) -> int:
    """Retrain all requested long-horizon models and print model-separated results."""
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    baseline_max_epochs = literature.MAX_EPOCHS if max_epochs is None else int(max_epochs)
    mainline_max_epochs = paper.MAX_EPOCHS if max_epochs is None else int(max_epochs)
    seeds = protocol.FORMAL_SEEDS if formal else (int(seed),)
    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    torch = base.require_torch()
    device = base.choose_device(torch)

    console.phase("V2 long-horizon mainline vs baselines")
    console.info(
        "task",
        input=f"{INPUT_STEPS} steps / {INPUT_HOURS}h",
        output=f"{OUTPUT_STEPS} direct steps / 4-{OUTPUT_HOURS}h",
        stations=len(stations),
        targets=len(TARGET_FEATURES),
    )
    console.info("execution", models=models, seeds=seeds, device=device, cache="disabled", files="none")

    needs_baseline_data = any(model in {"lstm", "gru", "ridge", "mlp"} for model in models)
    baseline_data = prepare_baseline_data(data, stations) if needs_baseline_data else None
    prepared_mainline = (
        mainline.prepare_targets(data, stations)
        if any(model in {"full_D", "rolling_D"} for model in models)
        else None
    )

    original_max_epochs = paper.MAX_EPOCHS
    paper.MAX_EPOCHS = mainline_max_epochs
    reference_counts: pd.DataFrame | None = None
    try:
        for model_index, model in enumerate(models, start=1):
            console.phase(f"retrain {MODEL_LABELS[model]}", current=model_index, total=len(models))
            overall_tables = []
            horizon_tables = []
            model_seeds = (-1,) if model == "ridge" else seeds
            for active_seed in model_seeds:
                if model in {"full_D", "rolling_D"}:
                    if prepared_mainline is None:
                        raise RuntimeError("mainline data was not prepared")
                    if model == "full_D":
                        tables, artifact = run_mainline_model(
                            torch,
                            prepared_mainline,
                            stations,
                            int(active_seed),
                            device,
                        )
                    else:
                        tables, artifact = run_rolling_model(
                            torch,
                            data,
                            prepared_mainline,
                            stations,
                            int(active_seed),
                            device,
                        )
                else:
                    if baseline_data is None:
                        raise RuntimeError("baseline data was not prepared")
                    tables, artifact = run_baseline_model(
                        torch,
                        model,
                        baseline_data,
                        stations,
                        int(seed if active_seed < 0 else active_seed),
                        baseline_max_epochs,
                        device,
                    )
                overall_tables.append(tables["overall"])
                horizon_tables.append(tables["horizon"])
                console.info(
                    "trained",
                    model=MODEL_LABELS[model],
                    seed="deterministic" if active_seed < 0 else active_seed,
                    best_epoch=artifact.get("best_epoch"),
                    parameters=artifact.get("parameter_count"),
                )

            overall = pd.concat(overall_tables, ignore_index=True)
            horizon = pd.concat(horizon_tables, ignore_index=True)
            reference_counts = validate_model_counts(reference_counts, horizon, model)
            print_model_result(model, overall, horizon)
    finally:
        paper.MAX_EPOCHS = original_max_epochs

    console.phase("completed")
    console.info("result", source="fresh training in this process", output="terminal only")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain and print the V2 24h-to-36h mainline/baseline comparison."
    )
    parser.add_argument(
        "--models",
        type=parse_model_names,
        default=MODEL_ORDER,
        help="comma-separated subset of lstm,gru,full_D,rolling_D,ridge,mlp",
    )
    parser.add_argument("--seed", type=int, default=protocol.PILOT_SEED, help="single-run seed")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="retrain neural models with all formal seeds instead of only --seed",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override each neural model's original maximum epoch setting",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_suite(
        models=args.models,
        seed=args.seed,
        formal=args.formal,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
