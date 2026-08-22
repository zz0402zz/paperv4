#!/usr/bin/env python3
"""Run a causal, single-station short-history TabPFN versus delta-GRU study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
from importlib import metadata
import random
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.tabpfn_comparison import config, data, io, models


@dataclass(frozen=True)
class ArrayScaler:
    """Training-only standardization with robust all-missing fallbacks."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        values = np.asarray(values, dtype=float)
        flattened = values.reshape(-1, values.shape[-1])
        finite = np.where(np.isfinite(flattened), flattened, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(finite, axis=0)
            scale = np.nanstd(finite, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _experiment_code_hashes() -> dict[str, str]:
    paths = tuple(
        Path("scripts/tabpfn_comparison") / name
        for name in ("config.py", "data.py", "io.py", "models.py", "run.py")
    )
    return {str(path): _file_sha256(path) for path in paths}


def task_metadata(
    *,
    evaluation_split: str,
    fit_splits: tuple[str, ...],
    model: str,
    seed: int,
    target: str,
    station: str,
) -> dict[str, object]:
    """Exact metadata guard used before a saved prediction is resumed."""
    return {
        "experiment": config.EXPERIMENT_ID,
        "model": model,
        "seed": int(seed),
        "station": str(station),
        "target": str(target),
        "evaluation_split": evaluation_split,
        "fit_splits": list(fit_splits),
        "start_date": config.START_DATE,
        "train_end": config.TRAIN_END,
        "val_end": config.VAL_END,
        "step_hours": config.STEP_HOURS,
        "input_steps": config.INPUT_STEPS,
        "output_steps": config.OUTPUT_STEPS,
        "input_features": list(config.INPUT_FEATURES),
        "target_policy": "approved_original_observations_only",
        "cross_station_features": False,
        "observed_data_path": str(config.OBSERVED_DATA_PATH),
        "observed_data_sha256": _file_sha256(config.OBSERVED_DATA_PATH),
        "quality_data_path": str(config.QUALITY_DATA_PATH),
        "quality_data_sha256": _file_sha256(config.QUALITY_DATA_PATH),
        "numpy_version": np.__version__,
        "torch_version": _package_version("torch") if model == config.DELTA_GRU_KEY else None,
        "tabpfn_version": _package_version("tabpfn") if model == config.DELTA_TABPFN_KEY else None,
        "tabpfn_fit_mode": config.TABPFN_FIT_MODE if model == config.DELTA_TABPFN_KEY else None,
        "tabpfn_prediction_batch_size": (
            config.TABPFN_PREDICTION_BATCH_SIZE if model == config.DELTA_TABPFN_KEY else None
        ),
        "model_identity": models.model_identity(model),
        "code_sha256": _experiment_code_hashes(),
    }


def _valid_label_rows(split: dict[str, np.ndarray]) -> np.ndarray:
    """Rows trainable for every requested direct forecast horizon."""
    return np.asarray(split["y_mask"], dtype=bool).all(axis=1) & np.isfinite(
        np.asarray(split["y_delta"], dtype=float)
    ).all(axis=1)


def _set_torch_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gru_inputs(
    fit_split: dict[str, np.ndarray],
    evaluation_split: dict[str, np.ndarray],
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ArrayScaler]:
    """Standardize continuous inputs from fit rows only and retain masks."""
    def continuous(split: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((split["x_raw"], split["x_diff"]), axis=-1)

    def masks(split: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((split["x_raw_mask"], split["x_diff_mask"]), axis=-1).astype(float)

    sequence_scaler = ArrayScaler.fit(continuous(fit_split)[fit_rows])
    current_scaler = ArrayScaler.fit(fit_split["current"][fit_rows])

    def sequence(split: dict[str, np.ndarray]) -> np.ndarray:
        scaled = sequence_scaler.transform(continuous(split))
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate((scaled, masks(split)), axis=-1).astype(np.float32)

    def current(split: dict[str, np.ndarray]) -> np.ndarray:
        scaled = current_scaler.transform(split["current"])
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate((scaled, split["current_mask"].astype(float)), axis=-1).astype(np.float32)

    return (
        sequence(fit_split),
        current(fit_split),
        sequence(evaluation_split),
        current(evaluation_split),
        ArrayScaler.fit(fit_split["y_delta"][fit_rows]),
    )


def predict_delta_gru(
    fit_split: dict[str, np.ndarray], evaluation_split: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    """Fit the matched local delta-GRU without using validation for early stop."""
    import torch

    fit_rows = _valid_label_rows(fit_split)
    if not fit_rows.any():
        raise ValueError("No fully approved training labels for this station-target task.")
    _set_torch_seed(torch, seed)
    train_sequence, train_current, eval_sequence, eval_current, target_scaler = _gru_inputs(
        fit_split, evaluation_split, fit_rows
    )
    target = target_scaler.transform(fit_split["y_delta"][fit_rows]).astype(np.float32)

    class ShortHistoryDeltaGru(torch.nn.Module):
        def __init__(self, sequence_dim: int) -> None:
            super().__init__()
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_dim,
                hidden_size=config.GRU_HIDDEN_SIZE,
                batch_first=True,
            )
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, config.GRU_CURRENT_HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Linear(config.GRU_CURRENT_HIDDEN_SIZE, config.GRU_HIDDEN_SIZE),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(config.GRU_HIDDEN_SIZE * 2, config.OUTPUT_STEPS)

        def forward(self, sequence_x, current_x):
            encoded, _ = self.sequence_encoder(sequence_x)
            current_state = self.current_encoder(current_x)
            return self.head(torch.cat((encoded[:, -1, :], current_state), dim=1))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    model = ShortHistoryDeltaGru(train_sequence.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GRU_LEARNING_RATE)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_sequence[fit_rows]),
        torch.as_tensor(train_current[fit_rows]),
        torch.as_tensor(target),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.GRU_BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    model.train()
    for _ in range(config.GRU_EPOCHS):
        for sequence_x, current_x, target_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence_x.to(device), current_x.to(device))
            loss = torch.nn.functional.mse_loss(prediction, target_y.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eval_sequence), config.GRU_BATCH_SIZE):
            end = start + config.GRU_BATCH_SIZE
            batch = model(
                torch.as_tensor(eval_sequence[start:end], device=device),
                torch.as_tensor(eval_current[start:end], device=device),
            )
            predictions.append(batch.cpu().numpy())
    if not predictions:
        return np.empty((0, config.OUTPUT_STEPS), dtype=float)
    return target_scaler.inverse_transform(np.concatenate(predictions, axis=0))


def predict_delta_tabpfn(
    fit_split: dict[str, np.ndarray], evaluation_split: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    """Fit TabPFN on one station's temporal origins and predict local deltas."""
    fit_rows = _valid_label_rows(fit_split)
    if not fit_rows.any():
        raise ValueError("No fully approved training labels for this station-target task.")
    train_x = data.tabpfn_features(fit_split)[fit_rows]
    evaluation_x = data.tabpfn_features(evaluation_split)
    medians = models.finite_feature_medians(train_x)
    train_x = models.apply_feature_medians(train_x, medians)
    evaluation_x = models.apply_feature_medians(evaluation_x, medians)
    train_y = np.asarray(fit_split["y_delta"], dtype=float)[fit_rows]
    prediction = np.full((len(evaluation_x), config.OUTPUT_STEPS), np.nan, dtype=float)
    # The matched GRU is run immediately before TabPFN in ``--model all``.
    # Release its allocator cache before constructing the TabPFN model.
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    for horizon in range(config.OUTPUT_STEPS):
        regressor = models.make_v2_regressor(seed)
        regressor.fit(train_x, train_y[:, horizon])
        batches = []
        for start in range(0, len(evaluation_x), config.TABPFN_PREDICTION_BATCH_SIZE):
            stop = start + config.TABPFN_PREDICTION_BATCH_SIZE
            batches.append(np.asarray(regressor.predict(evaluation_x[start:stop]), dtype=float))
        prediction[:, horizon] = np.concatenate(batches) if batches else np.asarray([], dtype=float)
    return prediction


def predict_absolute(
    model: str,
    fit_split: dict[str, np.ndarray],
    evaluation_split: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    """Convert the two delta-model outputs back to the official absolute scale."""
    current = np.repeat(np.asarray(evaluation_split["current"], dtype=float), config.OUTPUT_STEPS, axis=1)
    if model == config.PERSISTENCE_KEY:
        return current
    if model == config.DELTA_GRU_KEY:
        delta = predict_delta_gru(fit_split, evaluation_split, seed)
    elif model == config.DELTA_TABPFN_KEY:
        delta = predict_delta_tabpfn(fit_split, evaluation_split, seed)
    else:
        raise ValueError(f"Unknown model: {model}")
    return current + delta


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in _parse_csv(value))
    if not parsed:
        raise ValueError("At least one seed is required.")
    unknown = set(parsed).difference(config.FORMAL_SEEDS)
    if unknown:
        raise ValueError(f"Seeds outside the frozen protocol: {sorted(unknown)}")
    return parsed


def _selection(
    panel, stations_arg: str | None, targets_arg: str | None, all_stations: bool, all_targets: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = data.available_stations(panel)
    stations = available if all_stations else _parse_csv(stations_arg or "")
    targets = config.TARGETS if all_targets else _parse_csv(targets_arg or "")
    if not stations:
        raise ValueError("Pass --stations or explicitly request --all-stations.")
    if not targets:
        raise ValueError("Pass --targets or explicitly request --all-targets.")
    missing_stations = set(stations).difference(available)
    missing_targets = set(targets).difference(config.TARGETS)
    if missing_stations:
        raise ValueError(f"Unknown V2 station(s): {sorted(missing_stations)}")
    if missing_targets:
        raise ValueError(f"Unknown prediction target(s): {sorted(missing_targets)}")
    return tuple(stations), tuple(targets)


def save_run_summary(
    *,
    evaluation_split: str,
    fit_splits: tuple[str, ...],
    stations: tuple[str, ...],
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    selected_models: tuple[str, ...],
) -> None:
    output_dir = config.output_dir_for_split(evaluation_split)
    report_path = Path("scripts/tabpfn_comparison/report.py")
    payload = {
        "experiment": config.EXPERIMENT_ID,
        "evaluation_split": evaluation_split,
        "fit_splits": list(fit_splits),
        "stations": list(stations),
        "targets": list(targets),
        "seeds": list(seeds),
        "models": list(selected_models),
        "input_steps": config.INPUT_STEPS,
        "output_steps": config.OUTPUT_STEPS,
        "step_hours": config.STEP_HOURS,
        "input_features": list(config.INPUT_FEATURES),
        "tabpfn_fit_mode": config.TABPFN_FIT_MODE,
        "tabpfn_prediction_batch_size": config.TABPFN_PREDICTION_BATCH_SIZE,
        "observed_data_path": str(config.OBSERVED_DATA_PATH),
        "observed_data_sha256": _file_sha256(config.OBSERVED_DATA_PATH),
        "quality_data_path": str(config.QUALITY_DATA_PATH),
        "quality_data_sha256": _file_sha256(config.QUALITY_DATA_PATH),
        "cross_station_features": False,
        "code_sha256": {
            **_experiment_code_hashes(),
            str(report_path): _file_sha256(report_path),
        },
    }
    io.save_json(output_dir / "运行摘要.json", payload)
    io.save_json(output_dir / "运行清单.json", payload)


def run(
    *,
    model: str,
    evaluation_split: str,
    stations: tuple[str, ...],
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    force: bool,
) -> None:
    panel = data.load_v2_panel()
    fit_splits = ("train",) if evaluation_split == "val" else ("train", "val")
    selected_models = config.MODEL_KEYS if model == "all" else (model,)
    save_run_summary(
        evaluation_split=evaluation_split,
        fit_splits=fit_splits,
        stations=stations,
        targets=targets,
        seeds=seeds,
        selected_models=selected_models,
    )
    total = len(stations) * len(targets)
    completed = 0
    for station in stations:
        for target in targets:
            completed += 1
            console.phase(f"{station} / {target}", current=completed, total=total)
            splits = data.split_by_time(data.build_station_target_dataset(panel, station, target))
            fit_split = data.join_splits(*(splits[name] for name in fit_splits))
            evaluation = splits[evaluation_split]
            if not len(evaluation["target_start"]):
                raise ValueError(f"No {evaluation_split} windows for {station} / {target}.")
            if any(name != config.PERSISTENCE_KEY for name in selected_models) and not _valid_label_rows(fit_split).any():
                raise ValueError(f"No fit labels for {station} / {target}.")
            for model_name in selected_models:
                for seed in config.model_seeds(model_name, seeds):
                    path = io.prediction_path(evaluation_split, model_name, seed, target, station)
                    expected = task_metadata(
                        evaluation_split=evaluation_split,
                        fit_splits=fit_splits,
                        model=model_name,
                        seed=seed,
                        target=target,
                        station=station,
                    )
                    if io.should_skip(path, expected, force=force):
                        console.info("resume", model=model_name, seed=seed, status="already complete")
                        continue
                    prediction = predict_absolute(model_name, fit_split, evaluation, seed)
                    arrays = data.target_arrays(evaluation)
                    arrays["pred"] = prediction
                    io.save_prediction(path, arrays, expected)
                    console.info("saved", model=model_name, seed=seed, rows=len(prediction))
    console.done(config.output_dir_for_split(evaluation_split))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *config.MODEL_KEYS), default="all")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="Comma-separated V2 station names.")
    station_group.add_argument("--all-stations", action="store_true", help="Run every current V2 station.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--targets", help="Comma-separated target names.")
    target_group.add_argument("--all-targets", action="store_true", help="Run all five official targets.")
    parser.add_argument("--seeds", default=",".join(map(str, config.FORMAL_SEEDS)))
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--test-approved", action="store_true", help="Required before writing final-test predictions.")
    parser.add_argument("--force", action="store_true", help="Replace an existing prediction after explicit review.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_split == "test" and not args.test_approved:
        raise SystemExit("Test is locked. Review validation first, then pass --test-approved.")
    try:
        seeds = _parse_seeds(args.seeds)
        panel = data.load_v2_panel()
        stations, targets = _selection(panel, args.stations, args.targets, args.all_stations, args.all_targets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run(
        model=args.model,
        evaluation_split=args.evaluation_split,
        stations=stations,
        targets=targets,
        seeds=seeds,
        force=args.force,
    )


if __name__ == "__main__":
    main()
