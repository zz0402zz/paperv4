#!/usr/bin/env python3
"""Train B/C while preserving the completed A/D/E preprocessing experiments."""

from __future__ import annotations

import argparse
from functools import lru_cache
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import data, io
from scripts.multitarget_forecasting import preprocessing_component_config as config
from scripts.multitarget_forecasting.preprocessing_component_model import (
    train_component_ablation,
)
from scripts.multitarget_forecasting.run import _parse_seeds, select_stations


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_variants(value: str) -> tuple[str, ...]:
    variants = _csv(value)
    if not variants:
        raise ValueError("至少需要一种预处理组件变体。")
    unknown = set(variants).difference(config.TRAIN_VARIANTS)
    if unknown:
        raise ValueError(f"未知预处理组件变体: {sorted(unknown)}")
    return variants


@lru_cache(maxsize=1)
def frozen_identity() -> dict[str, object]:
    paths = (
        Path("scripts/multitarget_forecasting/config.py"),
        Path("scripts/multitarget_forecasting/data.py"),
        Path("scripts/multitarget_forecasting/head_ablation_config.py"),
        Path("scripts/multitarget_forecasting/head_ablation_model.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_data.py"),
        Path("scripts/multitarget_forecasting/preprocessing_ablation_model.py"),
        Path("scripts/multitarget_forecasting/preprocessing_component_config.py"),
        Path("scripts/multitarget_forecasting/preprocessing_component_model.py"),
    )
    return {
        **io.data_identity(),
        "code_sha256": {str(path): io.file_sha256(path) for path in paths},
    }


def expected_metadata(station: str, variant: str, seed: int) -> dict[str, object]:
    import torch

    spec = config.TRAIN_VARIANT_SPECS[variant]
    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": "joint_five_target_preprocessing_component_validation_prediction",
        "station": station,
        "variant": variant,
        "variant_label": config.VARIANT_LABELS[variant],
        "variant_spec": spec,
        "context": config.CONTEXT,
        "target_output_modes": config.TARGET_OUTPUT_MODES,
        "seed": int(seed),
        "targets": list(config.TARGETS),
        "horizon_hours": list(config.HORIZON_HOURS),
        "train_end": base_config.TRAIN_END,
        "validation_end": base_config.VAL_END,
        "internal_validation_start": base_config.INTERNAL_VAL_START,
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "input_scaler": spec["input_scaler"],
        "target_scaler": spec["target_scaler"],
        "loss": spec["loss"],
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
            f"已有组件拆分结果与当前协议不一致: {path}。"
            "请审阅后显式使用 --force。"
        )
    return arrays


def run(
    *,
    stations: str | None,
    all_stations: bool,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    force: bool,
) -> None:
    panel = data.load_development_panel()
    selected_stations = select_stations(panel, stations, all_stations)
    total = len(selected_stations) * len(variants) * len(seeds)
    completed = 0
    for station in selected_stations:
        dataset = data.build_station_dataset(panel, station)
        splits = data.split_by_time(dataset)
        train = splits["train"]
        validation = splits["val"]
        if not len(train["target_start"]) or not len(validation["target_start"]):
            raise ValueError(f"站点缺少训练集或验证集窗口: {station}")
        lower, upper = data.warning_thresholds(train)
        for variant in variants:
            for seed in seeds:
                completed += 1
                expected = expected_metadata(station, variant, seed)
                path = config.prediction_path(station, variant, seed)
                existing = None if force else load_exact(path, expected)
                saved_model = config.model_path(station, variant, seed)
                if existing is not None and saved_model.exists():
                    console.info(
                        "resume",
                        progress=f"{completed}/{total}",
                        station=station,
                        preprocessing=config.VARIANT_LABELS[variant],
                    )
                    continue
                console.info(
                    "train",
                    progress=f"{completed}/{total}",
                    station=station,
                    preprocessing=config.VARIANT_LABELS[variant],
                    seed=seed,
                    outputs=f"{config.OUTPUT_STEPS}x{len(config.TARGETS)}",
                )
                prediction, diagnostics = train_component_ablation(
                    train,
                    validation,
                    variant=variant,
                    seed=seed,
                    model_path=saved_model,
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
                    preprocessing=config.VARIANT_LABELS[variant],
                    seed=seed,
                    rows=len(prediction),
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="补齐稳健标准化与Huber损失的单组件消融"
    )
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="逗号分隔的站点")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument("--variants", default=",".join(config.TRAIN_VARIANTS))
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        stations=args.stations,
        all_stations=args.all_stations,
        variants=parse_variants(args.variants),
        seeds=_parse_seeds(args.seeds),
        force=args.force,
    )


if __name__ == "__main__":
    main()
