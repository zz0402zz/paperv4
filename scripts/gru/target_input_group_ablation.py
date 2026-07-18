"""Per-target input representation and feature-group ablation for the D-GRU."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.terminal_output import console
from scripts.common.wq_gru_data import FEATURE_COLUMNS
from scripts.data.build_hourly_representation_ablation import HOURLY_FEATURE_COLUMNS, STATISTICS, prefixed_feature
from scripts.gru import run_v2_hourly_representation_ablation as hourly
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_window_level_ablation as window


INPUT_STEPS = 6
OUTPUT_STEPS = 1
TARGETS = protocol.TARGET_FEATURE_COLUMNS
HOURLY_TARGETS = tuple(feature for feature in TARGETS if feature in HOURLY_FEATURE_COLUMNS)
NATIVE_4H_TARGETS = tuple(feature for feature in TARGETS if feature not in HOURLY_FEATURE_COLUMNS)
NON_TARGET_FEATURES = tuple(feature for feature in FEATURE_COLUMNS if feature not in TARGETS)
HOURLY_AUXILIARIES = tuple(feature for feature in NON_TARGET_FEATURES if feature in HOURLY_FEATURE_COLUMNS)
NATIVE_4H_AUXILIARIES = tuple(feature for feature in NON_TARGET_FEATURES if feature not in HOURLY_FEATURE_COLUMNS)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    active_columns: tuple[str, ...]
    control: bool = False


def ordered_union(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(column for group in groups for column in group))


def endpoint_diff(feature: str) -> str:
    return hourly.diff_column("endpoint", feature)


def mean_diff(feature: str) -> str:
    return hourly.diff_column("mean", feature)


def aligned_stats(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        prefixed_feature(f"window_{statistic}", feature)
        for feature in features
        for statistic in STATISTICS
    )


def shifted_stats(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        f"shift{hourly.CONTROL_SHIFT_STEPS}__{prefixed_feature(f'window_{statistic}', feature)}"
        for feature in features
        for statistic in STATISTICS
    )


def stage1_variants(target: str) -> tuple[Variant, ...]:
    endpoint = (endpoint_diff(target),)
    if target not in HOURLY_TARGETS:
        return (Variant("self_endpoint", "self 4h observation", endpoint),)
    return (
        Variant("self_mean", "self 4h mean", (mean_diff(target),)),
        Variant("self_endpoint", "self 4h endpoint", endpoint),
        Variant("self_endpoint_stats", "self endpoint + own stats", (*endpoint, *aligned_stats((target,)))),
        Variant(
            "self_endpoint_shifted_stats",
            "self endpoint + shifted own stats",
            (*endpoint, *shifted_stats((target,))),
            control=True,
        ),
    )


def stage2_variants(target: str, core_columns: tuple[str, ...]) -> tuple[Variant, ...]:
    other_targets = tuple(endpoint_diff(feature) for feature in TARGETS if feature != target)
    target5 = ordered_union(core_columns, other_targets)
    auxiliary_endpoint = tuple(endpoint_diff(feature) for feature in NON_TARGET_FEATURES)
    auxiliary_mean = (
        *(mean_diff(feature) for feature in HOURLY_AUXILIARIES),
        *(endpoint_diff(feature) for feature in NATIVE_4H_AUXILIARIES),
    )
    all9_endpoint = ordered_union(target5, auxiliary_endpoint)
    return (
        Variant("core_self", "selected self history", core_columns),
        Variant("target5", "+ other four targets", target5),
        Variant("target5_aux_endpoint", "+ auxiliary endpoint/raw", ordered_union(target5, auxiliary_endpoint)),
        Variant("target5_aux_mean", "+ auxiliary 4h mean/raw", ordered_union(target5, auxiliary_mean)),
        Variant(
            "target5_aux_endpoint_stats",
            "+ auxiliary endpoint + stats",
            ordered_union(all9_endpoint, aligned_stats(HOURLY_AUXILIARIES)),
        ),
        Variant(
            "target5_aux_endpoint_shifted_stats",
            "+ auxiliary endpoint + shifted stats",
            ordered_union(all9_endpoint, shifted_stats(HOURLY_AUXILIARIES)),
            control=True,
        ),
    )


def universal_columns(variants: Iterable[Variant]) -> tuple[str, ...]:
    return ordered_union(*(variant.active_columns for variant in variants))


def mask_scaled_splits(
    scaled_splits: dict[str, dict[str, np.ndarray]],
    universe: tuple[str, ...],
    active_columns: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Keep a common tensor shape while removing inactive value and mask channels."""
    active = set(active_columns)
    inactive = [index for index, column in enumerate(universe) if column not in active]
    raw_dim = len(universe)
    masked = {}
    for split_name, split in scaled_splits.items():
        self_x = split["self_x"].copy()
        if self_x.shape[-1] != raw_dim * 2:
            raise ValueError(f"Expected value+mask dimension {raw_dim * 2}, got {self_x.shape[-1]}")
        self_x[..., inactive] = 0.0
        self_x[..., [raw_dim + index for index in inactive]] = 0.0
        masked[split_name] = {**split, "self_x": self_x}
    return masked


def macro_station_rmse(metrics: dict) -> float | None:
    values = [
        item.get("rmse")
        for item in metrics.get("station_metrics", {}).values()
        if item.get("rmse") is not None and np.isfinite(item.get("rmse"))
    ]
    return float(np.mean(values)) if values else None


def change_metrics(
    arrays: dict[str, np.ndarray],
    last_target: np.ndarray,
    tail_threshold: float,
) -> dict[str, float | int | None]:
    last = last_target[:, None, :, :]
    pred_delta = arrays["pred"] - last
    true_delta = arrays["true"] - last
    valid = arrays["mask"].astype(bool) & np.isfinite(pred_delta) & np.isfinite(true_delta)
    nonzero = valid & (np.abs(true_delta) > 1e-12)
    tail = valid & (np.abs(true_delta) >= tail_threshold)
    return {
        "delta_sign_accuracy": float((np.sign(pred_delta[nonzero]) == np.sign(true_delta[nonzero])).mean()) if nonzero.any() else None,
        "tail_delta_rmse": float(np.sqrt(np.mean((pred_delta[tail] - true_delta[tail]) ** 2))) if tail.any() else None,
        "tail_points": int(tail.sum()),
        "train_p90_abs_delta": float(tail_threshold),
    }


def metric_row(
    stage: str,
    seed: int,
    target: str,
    variant: Variant,
    split: str,
    metrics: dict,
    parameters: int,
    change: dict[str, float | int | None] | None = None,
) -> dict[str, object]:
    return {
        "stage": stage,
        "seed": seed,
        "target": target,
        "variant": variant.key,
        "label": variant.label,
        "control": variant.control,
        "split": split,
        "active_channels": len(variant.active_columns),
        "model_parameters": parameters,
        "valid_points": metrics.get("valid_points"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "nse": metrics.get("nse"),
        "macro_station_rmse": macro_station_rmse(metrics),
        **(change or {}),
    }


def station_rows(stage: str, seed: int, target: str, variant: Variant, split: str, metrics: dict) -> list[dict[str, object]]:
    return [
        {
            "stage": stage,
            "seed": seed,
            "target": target,
            "variant": variant.key,
            "split": split,
            "station": station,
            "valid_points": item.get("valid_points"),
            "mae": item.get("mae"),
            "rmse": item.get("rmse"),
            "nse": item.get("nse"),
        }
        for station, item in metrics.get("station_metrics", {}).items()
    ]


def select_validation_winners(metrics: pd.DataFrame, stage: str) -> dict[str, dict[str, object]]:
    validation = metrics[
        metrics["stage"].eq(stage) & metrics["split"].eq("val") & ~metrics["control"].astype(bool)
    ]
    winners = {}
    for target, group in validation.groupby("target", sort=False):
        winner = group.sort_values(["macro_station_rmse", "active_channels", "variant"]).iloc[0]
        winners[str(target)] = {
            "variant": str(winner["variant"]),
            "label": str(winner["label"]),
            "macro_station_rmse": float(winner["macro_station_rmse"]),
            "selection_split": "val",
            "test_used": False,
        }
    return winners


def persistence_metrics(raw_split: dict[str, np.ndarray], target: str, stations: tuple[str, ...]) -> dict:
    pred = np.repeat(raw_split["last_target"][:, None, :, :], raw_split["y_abs"].shape[1], axis=1)
    mask = raw_split["y_mask"] & np.isfinite(pred) & np.isfinite(raw_split["y_abs"])
    return base.masked_error_metrics(
        pred - raw_split["y_abs"],
        mask,
        (target,),
        stations,
        truth=raw_split["y_abs"],
    )


def run_target_variants(
    *,
    stage: str,
    seed: int,
    target: str,
    variants: tuple[Variant, ...],
    data: pd.DataFrame,
    stations: tuple[str, ...],
    output_dir: Path,
    evaluation_splits: tuple[str, ...] = ("train", "val"),
    keep_checkpoints: bool = False,
    universe_columns: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    # Keep variant comparisons paired within a seed while making formal seeds real.
    random.seed(seed)
    np.random.seed(seed)
    paper.SEED = int(seed)
    universe = universe_columns or universal_columns(variants)
    unknown_active = set(ordered_union(*(variant.active_columns for variant in variants))) - set(universe)
    if unknown_active:
        raise ValueError(f"Active columns missing from universe: {sorted(unknown_active)}")
    raw_splits, scaled_splits, scalers = window.build_target_splits(
        data,
        stations,
        universe,
        target,
        INPUT_STEPS,
        include_current_level=False,
    )
    scaled_splits, current_scaler = dual.attach_scaled_current_level(raw_splits, scaled_splits)
    torch = base.require_torch()
    device = base.choose_device(torch)
    model = dual.make_dual_branch_model(
        torch,
        sequence_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        current_input_dim=scaled_splits["train"]["current_level"].shape[-1],
        target_dim=1,
        output_steps=OUTPUT_STEPS,
    )
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    metric_rows = []
    per_station_rows = []
    history_rows = []

    train_delta = raw_splits["train"]["y"]
    train_delta_mask = raw_splits["train"]["y_mask"] & np.isfinite(train_delta)
    tail_threshold = float(np.quantile(np.abs(train_delta[train_delta_mask]), 0.90))

    persistence = Variant("persistence", "Persistence", (), control=True)
    for split in evaluation_splits:
        split_metrics = persistence_metrics(raw_splits[split], target, stations)
        persistence_arrays = {
            "pred": np.repeat(raw_splits[split]["last_target"][:, None, :, :], raw_splits[split]["y_abs"].shape[1], axis=1),
            "true": raw_splits[split]["y_abs"],
            "mask": raw_splits[split]["y_mask"],
        }
        metric_rows.append(
            metric_row(
                stage,
                seed,
                target,
                persistence,
                split,
                split_metrics,
                0,
                change_metrics(persistence_arrays, raw_splits[split]["last_target"], tail_threshold),
            )
        )
        per_station_rows.extend(station_rows(stage, seed, target, persistence, split, split_metrics))

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dual.OUTPUT_DIR = checkpoint_dir
    for variant_index, variant in enumerate(variants, start=1):
        console.phase(f"{stage} | {target} | {variant.label}", current=variant_index, total=len(variants))
        active_splits = mask_scaled_splits(scaled_splits, universe, variant.active_columns)
        name = f"{stage}_{variant.key}_seed{seed}"
        result, arrays_by_split = dual.fit_dual_target_delta_gru(
            torch,
            name,
            target,
            universe,
            active_splits,
            scalers,
            current_scaler,
            stations,
            device,
            evaluation_splits=evaluation_splits,
        )
        for split, split_metrics in result["best_checkpoint"].items():
            metric_rows.append(
                metric_row(
                    stage,
                    seed,
                    target,
                    variant,
                    split,
                    split_metrics,
                    parameters,
                    change_metrics(arrays_by_split[split], raw_splits[split]["last_target"], tail_threshold),
                )
            )
            per_station_rows.extend(station_rows(stage, seed, target, variant, split, split_metrics))
        history_rows.extend(
            {
                "stage": stage,
                "seed": seed,
                "target": target,
                "variant": variant.key,
                **row,
            }
            for row in result["history"]
        )
        checkpoint = Path(result["best_model_path"])
        if not keep_checkpoints and checkpoint.exists():
            checkpoint.unlink()
    return metric_rows, per_station_rows, history_rows


def write_pilot_report(output_dir: Path, stage1: pd.DataFrame, stage2: pd.DataFrame, selections: dict) -> None:
    display_columns = ["target", "variant", "active_channels", "macro_station_rmse", "rmse", "nse"]
    lines = [
        "# Per-target input group ablation: pilot validation",
        "",
        "- Input: past 24h; output: next 4h endpoint/native observation.",
        "- Model: per-target D-GRU with L1 loss and current-target MLP branch.",
        "- Selection uses 2024 validation only; 2025 test metrics are not computed.",
        "- Every variant within a target uses the same tensor dimension and target origins.",
        "",
        "## Stage 1 validation",
        "```text",
        stage1[display_columns].sort_values(["target", "macro_station_rmse"]).to_string(index=False),
        "```",
        "",
        "## Stage 2 validation",
        "```text",
        stage2[display_columns].sort_values(["target", "macro_station_rmse"]).to_string(index=False),
        "```",
        "",
        "## Validation selections",
        "```json",
        json.dumps(selections, ensure_ascii=False, indent=2),
        "```",
    ]
    (output_dir / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pilot(output_dir: Path, seed: int = protocol.PILOT_SEED) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    paper.SEED = int(seed)
    window.OUTPUT_STEPS = OUTPUT_STEPS
    dual.INPUT_STEPS = INPUT_STEPS
    dual.OUTPUT_STEPS = OUTPUT_STEPS

    data = hourly.load_ablation_data()
    stations = tuple(sorted(data["station"].astype(str).unique()))
    console.phase("per-target input ablation pilot")
    console.info("protocol", seed=seed, stations=len(stations), input="24h", output="4h", test="sealed")

    stage1_metrics = []
    stage1_stations = []
    histories = []
    for target in TARGETS:
        rows, station_items, history = run_target_variants(
            stage="stage1_self_representation",
            seed=seed,
            target=target,
            variants=stage1_variants(target),
            data=data,
            stations=stations,
            output_dir=output_dir,
        )
        stage1_metrics.extend(rows)
        stage1_stations.extend(station_items)
        histories.extend(history)

    stage1_frame = pd.DataFrame(stage1_metrics)
    stage1_winners = select_validation_winners(stage1_frame, "stage1_self_representation")
    stage2_metrics = []
    stage2_stations = []
    for target in TARGETS:
        variants = stage1_variants(target)
        selected_key = stage1_winners[target]["variant"]
        core = next(variant.active_columns for variant in variants if variant.key == selected_key)
        rows, station_items, history = run_target_variants(
            stage="stage2_feature_groups",
            seed=seed,
            target=target,
            variants=stage2_variants(target, core),
            data=data,
            stations=stations,
            output_dir=output_dir,
        )
        stage2_metrics.extend(rows)
        stage2_stations.extend(station_items)
        histories.extend(history)

    stage2_frame = pd.DataFrame(stage2_metrics)
    stage2_winners = select_validation_winners(stage2_frame, "stage2_feature_groups")
    all_metrics = pd.concat([stage1_frame, stage2_frame], ignore_index=True)
    all_stations = pd.DataFrame([*stage1_stations, *stage2_stations])
    all_metrics.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    all_stations.to_csv(output_dir / "validation_station_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    selections = {"stage1": stage1_winners, "stage2": stage2_winners, "test_used": False}
    (output_dir / "validation_selection.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = protocol.build_run_manifest(
        experiment="stage3e_per_target_input_group_pilot",
        output_dir=output_dir,
        seed=seed,
        observed_path=hourly.VALUES_PATH,
        quality_path=hourly.QUALITY_PATH,
        code_paths=(
            Path("scripts/gru/target_input_group_ablation.py"),
            Path("scripts/gru/run_v2_target_input_group_ablation.py"),
        ),
    )
    manifest.update(
        {
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "hourly_targets": list(HOURLY_TARGETS),
            "native_4h_targets": list(NATIVE_4H_TARGETS),
            "non_target_features": list(NON_TARGET_FEATURES),
            "evaluation_splits": ["train", "val"],
            "test_metrics_computed": False,
        }
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_pilot_report(
        output_dir,
        stage1_frame[stage1_frame["split"].eq("val")],
        stage2_frame[stage2_frame["split"].eq("val")],
        selections,
    )
    console.done(output_dir, report="pilot_report.md", test="sealed")
    return 0
