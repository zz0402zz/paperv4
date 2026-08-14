#!/usr/bin/env python3
"""Run strict local TabPFN inference for the frozen 2024 comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from scripts.common.terminal_output import console
from scripts.common.wq_gru_data import add_feature_enhancements
from scripts.common import forecasting
from scripts.attention import comparison as mainline_comparison
from scripts.data import jinhua_panel
from scripts.tabpfn_comparison import config, data, io, models


def run_frozen_gru_export_in_main_environment(*, force: bool) -> None:
    """Keep checkpoint loading in the environment that created the GRU files."""
    main_python = Path(".venv") / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not main_python.is_file():
        raise SystemExit(
            "找不到论文主环境解释器 "
            f"`{main_python}`。请先按 "
            "`scripts/tabpfn_comparison/README.md` 创建并安装 `.venv`。"
        )
    command = [
        str(main_python),
        "-m",
        "scripts.tabpfn_comparison.run",
        "--model",
        "frozen_gru",
        "--internal-frozen-gru-export",
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def load_panel() -> pd.DataFrame:
    panel = jinhua_panel.load_panel(
        start_date=config.START_DATE,
        include_test=False,
    )
    return add_feature_enhancements(panel, ("diff1",))


def task_metadata(
    *,
    model: str,
    seed: int,
    target: str,
    station: str,
    features: tuple[str, ...],
) -> dict:
    if model in {config.DELTA_GRU_KEY, config.MATCHED_GRU_KEY}:
        import torch

        versions = {"torch": torch.__version__}
    elif model in config.TABPFN_KEYS:
        versions = models.require_dependencies(
            need_time_series=model in config.NATIVE_SPECS
        )
    else:
        versions = {}
    return {
        "protocol": "tabpfn_2024_strict_v2",
        "model": model,
        "seed": int(seed),
        "target": target,
        "station": station,
        "selected_features": list(features),
        "start_date": config.START_DATE,
        "train_end": config.TRAIN_END,
        "val_end": config.VAL_END,
        "input_steps": config.INPUT_STEPS,
        "output_steps": config.OUTPUT_STEPS,
        "step_hours": config.STEP_HOURS,
        "tabpfn_version": versions.get("tabpfn"),
        "tabpfn_time_series_version": versions.get("tabpfn-time-series"),
        "torch_version": versions.get("torch"),
        "model_identity": models.model_identity(model),
    }


def _save(
    *,
    model: str,
    seed: int,
    target: str,
    station: str,
    features: tuple[str, ...],
    arrays: dict[str, np.ndarray],
) -> None:
    metadata_payload = task_metadata(
        model=model,
        seed=seed,
        target=target,
        station=station,
        features=features,
    )
    io.save_prediction(
        io.prediction_path(model, seed, target, station),
        arrays,
        metadata_payload,
    )


def run_native(
    model: str,
    panel: pd.DataFrame,
    features_by_target: dict[str, tuple[str, ...]],
    *,
    force: bool,
    batch_size: int,
    checkpoint_every_batches: int,
) -> None:
    spec = config.NATIVE_SPECS[model]
    pipeline = models.make_native_pipeline(
        spec.version,
        max_context_length=spec.max_context_length,
    )
    seed = 0
    for target in config.TARGETS:
        raw, _, _ = data.validation_splits(
            panel,
            target,
            features_by_target[target],
        )
        val = raw["val"]
        origins = pd.to_datetime(val["target_start"]) - pd.Timedelta(
            hours=config.STEP_HOURS
        )
        for station_index, station in enumerate(config.STATIONS):
            path = io.prediction_path(model, seed, target, station)
            partial_path = io.partial_prediction_path(path)
            expected = task_metadata(
                model=model,
                seed=seed,
                target=target,
                station=station,
                features=features_by_target[target],
            )
            if io.should_skip(path, expected, force=force):
                console.info("resume", model=model, target=target, station=station)
                continue
            series = data.approved_target_series(panel, station, target)
            base = data.station_arrays(val, station_index)
            if force or not partial_path.exists():
                pred = np.empty((0, config.OUTPUT_STEPS), dtype=float)
            else:
                pred = io.load_prediction_prefix(partial_path, expected, base)
                console.info(
                    "resume partial",
                    model=model,
                    target=target,
                    station=station,
                    origins=f"{len(pred)}/{len(origins)}",
                )
            resumed_count = len(pred)
            for batch_index, start in enumerate(
                range(resumed_count, len(origins), batch_size),
                start=1,
            ):
                stop = min(start + batch_size, len(origins))
                context_df, future_df = data.native_origin_batch(
                    series,
                    origins[start:stop].to_numpy(),
                    max_context_length=spec.max_context_length,
                )
                result = pipeline.predict_df(
                    context_df,
                    future_df=future_df,
                    quantiles=[0.5],
                )
                pred = np.concatenate(
                    (
                        pred,
                        data.reshape_native_prediction(result, stop - start),
                    ),
                    axis=0,
                )
                if (
                    batch_index % checkpoint_every_batches == 0
                    or stop == len(origins)
                ):
                    io.save_prediction(
                        partial_path,
                        {
                            "pred": pred,
                            "true": base["true"][:stop],
                            "mask": base["mask"][:stop],
                            "current": base["current"][:stop],
                            "target_start": base["target_start"][:stop],
                        },
                        expected,
                    )
                console.info(
                    "native progress",
                    model=model,
                    target=target,
                    station=station,
                    origins=f"{stop}/{len(origins)}",
                )
            if len(pred) != len(base["target_start"]):
                raise RuntimeError("Native prediction checkpoint is incomplete")
            # The full partial file already contains the exact final payload;
            # atomic rename avoids a second large write and leaves no stale file.
            partial_path.replace(path)


def run_delta_tabpfn(
    panel: pd.DataFrame,
    features_by_target: dict[str, tuple[str, ...]],
    *,
    seeds: tuple[int, ...],
    force: bool,
) -> None:
    model_key = config.DELTA_TABPFN_KEY
    for target in config.TARGETS:
        raw, _, _ = data.validation_splits(
            panel,
            target,
            features_by_target[target],
        )
        for seed in seeds:
            for station_index, station in enumerate(config.STATIONS):
                path = io.prediction_path(model_key, seed, target, station)
                expected = task_metadata(
                    model=model_key,
                    seed=seed,
                    target=target,
                    station=station,
                    features=features_by_target[target],
                )
                if io.should_skip(path, expected, force=force):
                    console.info(
                        "resume",
                        model=model_key,
                        seed=seed,
                        target=target,
                        station=station,
                    )
                    continue
                base = data.station_arrays(raw["val"], station_index)
                pred_delta = np.full_like(base["true"], np.nan, dtype=float)
                for horizon_index in range(config.OUTPUT_STEPS):
                    train_x, train_y, val_x, _ = data.delta_tabular_xy(
                        raw["train"],
                        raw["val"],
                        station_index,
                        horizon_index,
                    )
                    medians = models.finite_feature_medians(train_x)
                    train_x = models.apply_feature_medians(train_x, medians)
                    val_x = models.apply_feature_medians(val_x, medians)
                    regressor = models.make_v2_regressor(seed)
                    regressor.fit(train_x, train_y)
                    pred_delta[:, horizon_index] = regressor.predict(val_x)
                arrays = {
                    **base,
                    "pred": base["current"] + pred_delta,
                }
                _save(
                    model=model_key,
                    seed=seed,
                    target=target,
                    station=station,
                    features=features_by_target[target],
                    arrays=arrays,
                )
                console.info(
                    "saved",
                    model=model_key,
                    seed=seed,
                    target=target,
                    station=station,
                )


def export_frozen_gru(
    panel: pd.DataFrame,
    features_by_target: dict[str, tuple[str, ...]],
    *,
    force: bool,
) -> None:
    """Load the already-frozen GRU checkpoints; never train or modify them."""
    torch = forecasting.require_torch()
    device = forecasting.choose_device(torch)
    variants = mainline_comparison.VARIANTS[:2]
    for seed in config.SEEDS:
        _, predictions = mainline_comparison.evaluate_checkpoint_seed(
            seed=seed,
            data=panel,
            stations=config.STATIONS,
            selected_by_target=features_by_target,
            torch=torch,
            device=device,
            validation_dir=config.MAINLINE_VALIDATION_DIR,
            variants=variants,
            targets=config.TARGETS,
            split_name="val",
            input_steps=config.INPUT_STEPS,
            output_steps=config.OUTPUT_STEPS,
        )
        for variant in variants:
            for target in config.TARGETS:
                combined = predictions[(seed, variant.key, target)]
                for station_index, station in enumerate(config.STATIONS):
                    path = io.prediction_path(variant.key, seed, target, station)
                    expected = task_metadata(
                        model=variant.key,
                        seed=seed,
                        target=target,
                        station=station,
                        features=features_by_target[target],
                    )
                    if io.should_skip(path, expected, force=force):
                        continue
                    arrays = {
                        "pred": combined["pred"][:, :, station_index, 0],
                        "true": combined["true"][:, :, station_index, 0],
                        "mask": combined["mask"][:, :, station_index, 0],
                        "current": combined["current"][:, station_index, :],
                        "target_start": combined["target_start"],
                    }
                    io.save_prediction(path, arrays, expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("frozen_gru", *config.TABPFN_KEYS, "all"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=config.NATIVE_BATCH_SIZE)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=config.NATIVE_CHECKPOINT_EVERY_BATCHES,
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, config.SEEDS)),
        help="Only applies to delta_tabpfn_v2.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--internal-frozen-gru-export",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.checkpoint_every_batches < 1:
        raise SystemExit("--checkpoint-every-batches must be positive")
    selected_seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    unknown = set(selected_seeds).difference(config.SEEDS)
    if unknown:
        raise SystemExit(f"Seeds outside frozen protocol: {sorted(unknown)}")
    if args.model == "frozen_gru" and not args.internal_frozen_gru_export:
        run_frozen_gru_export_in_main_environment(force=args.force)
        console.done(config.OUTPUT_DIR / "predictions")
        return
    panel = load_panel()
    features_by_target = data.selected_features()
    models_to_run = (
        ("frozen_gru", *config.TABPFN_KEYS)
        if args.model == "all"
        else (args.model,)
    )
    for model in models_to_run:
        console.phase(f"run {model}")
        if model == "frozen_gru" and not args.internal_frozen_gru_export:
            run_frozen_gru_export_in_main_environment(force=args.force)
        elif model == "frozen_gru":
            export_frozen_gru(panel, features_by_target, force=args.force)
        elif model in config.NATIVE_SPECS:
            run_native(
                model,
                panel,
                features_by_target,
                force=args.force,
                batch_size=args.batch_size,
                checkpoint_every_batches=args.checkpoint_every_batches,
            )
        else:
            run_delta_tabpfn(
                panel,
                features_by_target,
                seeds=selected_seeds,
                force=args.force,
            )
    console.done(config.OUTPUT_DIR / "predictions")


if __name__ == "__main__":
    main()
