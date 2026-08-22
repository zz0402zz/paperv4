#!/usr/bin/env python3
"""Run paired input-context screening for one-pass five-target GRU models."""

from __future__ import annotations

import argparse
from importlib import metadata as package_metadata

import numpy as np

from scripts.common.terminal_output import console
from scripts.multitarget_forecasting import config, data, io
from scripts.multitarget_forecasting.model import train_joint_gru


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_contexts(value: str) -> tuple[str, ...]:
    contexts = _csv(value)
    if not contexts:
        raise ValueError("至少需要一种输入尺度。")
    unknown = set(contexts).difference(config.CONTEXTS)
    if unknown:
        raise ValueError(f"未知输入尺度: {sorted(unknown)}")
    return contexts


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in _csv(value))
    if not seeds:
        raise ValueError("至少需要一个随机种子。")
    unknown = set(seeds).difference(config.FORMAL_SEEDS)
    if unknown:
        raise ValueError(f"随机种子不在冻结协议中: {sorted(unknown)}")
    return seeds


def _parse_target_modes(value: str) -> tuple[str, ...]:
    modes = _csv(value)
    if not modes:
        raise ValueError("至少需要一种输出表示。")
    unknown = set(modes).difference(config.TARGET_MODES)
    if unknown:
        raise ValueError(f"未知输出表示: {sorted(unknown)}")
    return modes


def select_stations(
    panel, stations: str | None, all_stations: bool
) -> tuple[str, ...]:
    available = data.available_stations(panel)
    selected = available if all_stations else _csv(stations or "")
    if not selected:
        raise ValueError("至少选择一个站点。")
    unknown = set(selected).difference(available)
    if unknown:
        raise ValueError(f"数据中不存在这些站点: {sorted(unknown)}")
    return selected


def metadata(
    station: str, context: str, target_mode: str, seed: int
) -> dict[str, object]:
    import torch

    return {
        "experiment": config.EXPERIMENT_ID,
        "kind": "joint_five_target_validation_prediction",
        "station": station,
        "context": context,
        "context_label": config.CONTEXT_LABELS[context],
        "target_mode": target_mode,
        "target_mode_label": config.TARGET_MODE_LABELS[target_mode],
        "seed": int(seed),
        "input_steps": config.CONTEXT_STEPS[context],
        "input_features": list(config.INPUT_FEATURES),
        "targets": list(config.TARGETS),
        "horizon_hours": list(config.HORIZON_HOURS),
        "output_shape_per_sample": [config.OUTPUT_STEPS, len(config.TARGETS)],
        "joint_single_forward_pass": True,
        "loss": "per_target_horizon_standardized_equal_weight_masked_mse",
        "validation_labels_used_for_fit": False,
        "test_labels_used": False,
        "warning_threshold_source": "training_split_quantiles_not_regulatory_limits",
        "warning_quantile": config.WARNING_QUANTILE,
        "gru_hidden_size": config.GRU_HIDDEN_SIZE,
        "context_hidden_size": config.CONTEXT_HIDDEN_SIZE,
        "fusion_hidden_size": config.FUSION_HIDDEN_SIZE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epoch_selection": {
            "internal_validation_start": config.INTERNAL_VAL_START,
            "maximum_epochs": config.MAX_EPOCHS,
            "minimum_epochs": config.MIN_EPOCHS,
            "evaluation_every": config.EVALUATION_EVERY,
            "patience_evaluations": config.EARLY_STOPPING_PATIENCE,
            "minimum_improvement": config.EARLY_STOPPING_MIN_DELTA,
            "selection_metric": "balanced_standardized_masked_mse",
            "refit_on_full_2022_2023": True,
        },
        "internal_validation_labels_used_for_epoch_selection": True,
        "year_2024_labels_used_for_epoch_selection": False,
        "torch_version": package_metadata.version("torch"),
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        **io.data_identity(),
        "code_sha256": io.code_identity(),
    }


def run(
    *,
    stations: str | None,
    all_stations: bool,
    contexts: tuple[str, ...],
    target_modes: tuple[str, ...],
    seeds: tuple[int, ...],
    force: bool,
) -> None:
    # Development runs cannot even construct a window containing 2025 labels.
    panel = data.load_development_panel()
    selected_stations = select_stations(panel, stations, all_stations)
    total = (
        len(selected_stations) * len(contexts) * len(target_modes) * len(seeds)
    )
    completed = 0
    for station in selected_stations:
        dataset = data.build_station_dataset(panel, station)
        splits = data.split_by_time(dataset)
        train = splits["train"]
        validation = splits["val"]
        if not len(train["target_start"]) or not len(validation["target_start"]):
            raise ValueError(f"站点缺少训练集或验证集窗口: {station}")
        lower, upper = data.warning_thresholds(train)
        for context in contexts:
            for target_mode in target_modes:
                for seed in seeds:
                    completed += 1
                    expected = metadata(station, context, target_mode, seed)
                    path = config.prediction_path(station, context, target_mode, seed)
                    existing = None if force else io.load_exact(path, expected)
                    saved_model = config.model_path(
                        station, context, target_mode, seed
                    )
                    if existing is not None and saved_model.exists():
                        expected_shape = (
                            len(validation["target_start"]),
                            config.OUTPUT_STEPS,
                            len(config.TARGETS),
                        )
                        if existing.get("pred", np.empty(0)).shape != expected_shape:
                            raise RuntimeError(f"已有预测形状错误: {path}")
                        console.info(
                            "resume",
                            progress=f"{completed}/{total}",
                            station=station,
                            context=context,
                            target_mode=target_mode,
                        )
                        continue
                    console.info(
                        "train",
                        progress=f"{completed}/{total}",
                        station=station,
                        context=config.CONTEXT_LABELS[context],
                        target_mode=config.TARGET_MODE_LABELS[target_mode],
                        seed=seed,
                        outputs=f"{config.OUTPUT_STEPS}x{len(config.TARGETS)}",
                    )
                    prediction, diagnostics = train_joint_gru(
                        train,
                        validation,
                        context=context,
                        target_mode=target_mode,
                        seed=seed,
                        model_path=config.model_path(
                            station, context, target_mode, seed
                        ),
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
                        context=context,
                        target_mode=target_mode,
                        seed=seed,
                        rows=len(prediction),
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单模型一次输出五指标和未来4至72小时的输入尺度消融"
    )
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--stations", help="逗号分隔的站点")
    station_group.add_argument("--all-stations", action="store_true")
    parser.add_argument(
        "--contexts",
        default=",".join(config.CONTEXTS),
        help="24h,72h,7d,multiscale中的一个或多个",
    )
    parser.add_argument(
        "--target-modes",
        default=",".join(config.TARGET_MODES),
        help="absolute,delta中的一个或两个",
    )
    parser.add_argument("--seeds", default=str(config.SCREENING_SEED))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        stations=args.stations,
        all_stations=args.all_stations,
        contexts=_parse_contexts(args.contexts),
        target_modes=_parse_target_modes(args.target_modes),
        seeds=_parse_seeds(args.seeds),
        force=args.force,
    )


if __name__ == "__main__":
    main()
