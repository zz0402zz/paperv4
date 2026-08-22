#!/usr/bin/env python3
"""Run A/C/D/E by one-target output-mode flips on the 2024 validation block."""

from __future__ import annotations

import argparse
from functools import lru_cache
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import preprocessing_mode_interaction_config as config
from scripts.multitarget_forecasting.preprocessing_mode_interaction_model import (
    train_mode_interaction,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_choices(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected = _csv(value)
    if not selected:
        raise ValueError(f"至少需要一种{label}。")
    unknown = set(selected).difference(allowed)
    if unknown:
        raise ValueError(f"未知{label}: {sorted(unknown)}")
    return selected


@lru_cache(maxsize=1)
def frozen_identity() -> dict[str, object]:
    paths = (
        Path("scripts/multitarget_forecasting/config.py"),
        Path("scripts/multitarget_forecasting/data.py"),
        Path("scripts/multitarget_forecasting/head_ablation_config.py"),
        Path("scripts/multitarget_forecasting/head_ablation_model.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_data.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_model.py"),
        Path("scripts/multitarget_forecasting/preprocessing_component_model.py"),
        Path("scripts/multitarget_forecasting/preprocessing_mode_interaction_config.py"),
        Path("scripts/multitarget_forecasting/preprocessing_mode_interaction_model.py"),
    )
    return {
        **io.data_identity(),
        "code_sha256": {str(path): io.file_sha256(path) for path in paths},
    }


def expected_metadata(
    station: str, preprocessing: str, flip: str, seed: int
) -> dict[str, object]:
    import torch

    spec = config.PREPROCESSING_SPECS[preprocessing]
    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": "joint_five_target_preprocessing_mode_interaction_validation_prediction",
        "station": station,
        "preprocessing": preprocessing,
        "preprocessing_spec": spec,
        "flip": flip,
        "flip_label": config.flip_label(flip),
        "target_output_modes": config.flipped_modes(flip),
        "base_target_output_modes": config.BASE_MODES,
        "context": config.CONTEXT,
        "seed": int(seed),
        "targets": list(config.TARGETS),
        "horizon_hours": list(config.HORIZON_HOURS),
        "train_end": base_config.TRAIN_END,
        "validation_end": base_config.VAL_END,
        "internal_validation_start": base_config.INTERNAL_VAL_START,
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "screening_weights_saved": False,
        "huber_delta": config.HUBER_DELTA if spec["loss"] == "huber" else None,
        "torch_version": package_metadata.version("torch"),
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        **frozen_identity(),
    }


def load_exact(path: Path, expected: dict[str, object]):
    if not path.exists():
        return None
    arrays, actual = io.load_archive(path)
    if actual != expected:
        raise RuntimeError(
            f"已有交互消融结果与当前协议不一致: {path}。"
            "请审阅后显式使用 --force。"
        )
    return arrays


def run(
    *,
    stations: str | None,
    all_stations: bool,
    preprocessings: tuple[str, ...],
    flips: tuple[str, ...],
    seeds: tuple[int, ...],
    force: bool,
) -> None:
    panel = data.load_development_panel()
    selected_stations = select_stations(panel, stations, all_stations)
    total = len(selected_stations) * len(preprocessings) * len(flips) * len(seeds)
    completed = 0
    for station in selected_stations:
        dataset = data.build_station_dataset(panel, station)
        splits = data.split_by_time(dataset)
        train = splits["train"]
        validation = splits["val"]
        if not len(train["target_start"]) or not len(validation["target_start"]):
            raise ValueError(f"站点缺少训练集或验证集窗口: {station}")
        lower, upper = data.warning_thresholds(train)
        for preprocessing in preprocessings:
            for flip in flips:
                for seed in seeds:
                    completed += 1
                    expected = expected_metadata(station, preprocessing, flip, seed)
                    path = config.prediction_path(station, preprocessing, flip, seed)
                    existing = None if force else load_exact(path, expected)
                    if existing is not None:
                        console.info(
                            "resume",
                            progress=f"{completed}/{total}",
                            station=station,
                            preprocessing=config.PREPROCESSING_SPECS[preprocessing]["label"],
                            flip=config.flip_label(flip),
                        )
                        continue
                    console.info(
                        "train",
                        progress=f"{completed}/{total}",
                        station=station,
                        preprocessing=config.PREPROCESSING_SPECS[preprocessing]["label"],
                        flip=config.flip_label(flip),
                        seed=seed,
                        outputs=f"{config.OUTPUT_STEPS}x{len(config.TARGETS)}",
                    )
                    prediction, diagnostics = train_mode_interaction(
                        train,
                        validation,
                        preprocessing=preprocessing,
                        flip=flip,
                        seed=seed,
                    )
                    arrays = {
                        "pred": prediction,
                        "true": np.asarray(validation["y_abs"], dtype=float),
                        "mask": np.asarray(validation["y_mask"], dtype=bool),
                        "current": np.asarray(validation["current"], dtype=float),
                        "current_mask": np.asarray(
                            validation["current_mask"], dtype=bool
                        ),
                        "target_start": np.asarray(
                            validation["target_start"], dtype="datetime64[ns]"
                        ),
                        "warning_lower": lower,
                        "warning_upper": upper,
                        **diagnostics,
                    }
                    io.save_archive(path, arrays, expected)
                    console.info(
                        "saved",
                        station=station,
                        preprocessing=config.PREPROCESSING_SPECS[preprocessing]["label"],
                        flip=config.flip_label(flip),
                        seed=seed,
                        rows=len(prediction),
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理与逐指标原值/变化量翻转交互消融")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--preprocessings", default=",".join(config.PREPROCESSINGS))
    parser.add_argument("--flips", default=",".join(config.FLIPS))
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        stations=args.stations,
        all_stations=args.all_stations,
        preprocessings=_parse_choices(
            args.preprocessings, config.PREPROCESSINGS, "预处理链"
        ),
        flips=_parse_choices(args.flips, config.FLIPS, "指标翻转"),
        seeds=_parse_seeds(args.seeds),
        force=args.force,
    )


if __name__ == "__main__":
    main()
