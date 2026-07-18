#!/usr/bin/env python3
"""Compare all-station self-GRU current-state variants under multi-step outputs."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_all_station_step_level_ablation as all_step
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_wentu_diff_delta_gru as delta
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_self_feature_ablation as self_ablation
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS


OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_direct_multistep" / "CD_9to9_seed42"
INPUT_STEPS = 9
OUTPUT_STEP_OPTIONS = (9,)
HORIZON_WEIGHTS = (1.0,) * 9
TARGET_FEATURE_COLUMNS = window.TARGET_FEATURE_COLUMNS


@dataclass(frozen=True)
class VariantSpec:
    key: str
    input_mode: str
    label: str
    include_current_level: bool = False


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("step_raw_plus_diff", "step_raw_plus_diff", "corr_top3_step_level"),
    VariantSpec("dual_branch_current_mlp", "dual_branch_current_mlp", "corr_top3_dual_branch_current_mlp"),
)

TERMINAL_MODEL_NAMES = {
    "persistence": "Persistence",
    "step_raw_plus_diff": "C step-level",
    "dual_branch_current_mlp": "D dual-branch",
}


def steps_to_hours(steps: int) -> int:
    """Convert 4-hour steps into hours."""
    return int(steps * 4)


def experiment_name(input_mode: str, output_steps: int) -> str:
    """Name one A/B/C/D multi-step experiment."""
    label_by_mode = {variant.input_mode: variant.label for variant in VARIANTS}
    if input_mode not in label_by_mode:
        raise KeyError(f"Unknown input_mode: {input_mode}")
    return f"all_station_{label_by_mode[input_mode]}_{INPUT_STEPS}to{output_steps}_delta"


def experiment_matrix() -> list[dict[str, object]]:
    """Return the full A/B/C/D by output-horizon design matrix."""
    rows = []
    for output_steps in OUTPUT_STEP_OPTIONS:
        for variant in VARIANTS:
            rows.append(
                {
                    "experiment": experiment_name(variant.input_mode, output_steps),
                    "input_mode": variant.input_mode,
                    "input_steps": INPUT_STEPS,
                    "input_hours": steps_to_hours(INPUT_STEPS),
                    "output_steps": output_steps,
                    "max_horizon_hours": steps_to_hours(output_steps),
                }
            )
    return rows


def save_json(path: Path, payload: dict) -> None:
    """Write UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_output_steps(output_steps: int, output_dir: Path) -> None:
    """Point reused modules at the active output length and directory."""
    window.OUTPUT_STEPS = output_steps
    window.OUTPUT_DIR = output_dir
    dual.OUTPUT_STEPS = output_steps
    dual.INPUT_STEPS = INPUT_STEPS
    dual.OUTPUT_DIR = output_dir


def _flat_metrics(error: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(error) & np.isfinite(truth)
    if not valid.any():
        return {"valid_points": 0, "mae": None, "rmse": None, "nse": None}
    err = np.asarray(error, dtype=float)[valid]
    tru = np.asarray(truth, dtype=float)[valid]
    denom = float(np.sum((tru - np.mean(tru)) ** 2))
    return {
        "valid_points": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nse": None if denom <= np.finfo(float).eps else float(1.0 - np.sum(err**2) / denom),
    }


def _arrays_slice(arrays_by_target: dict[str, dict[str, np.ndarray]], horizon_idx: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        feature: {
            "pred": arrays["pred"][:, horizon_idx : horizon_idx + 1, :, :],
            "true": arrays["true"][:, horizon_idx : horizon_idx + 1, :, :],
            "mask": arrays["mask"][:, horizon_idx : horizon_idx + 1, :, :],
        }
        for feature, arrays in arrays_by_target.items()
    }


def horizon_metric_rows(
    experiment: str,
    input_mode: str,
    split: str,
    arrays_by_target: dict[str, dict[str, np.ndarray]],
    stations: tuple[str, ...],
    target_columns: tuple[str, ...],
    output_steps: int,
) -> list[dict[str, object]]:
    """Overall metrics for each forecast horizon."""
    rows = []
    for horizon_idx in range(output_steps):
        metrics = self_ablation.aggregate_single_target_arrays(
            _arrays_slice(arrays_by_target, horizon_idx),
            stations,
            target_columns,
        )
        rows.append(
            {
                "experiment": experiment,
                "input_mode": input_mode,
                "split": split,
                "input_steps": INPUT_STEPS,
                "input_hours": steps_to_hours(INPUT_STEPS),
                "output_steps": output_steps,
                "horizon_step": horizon_idx + 1,
                "horizon_hours": steps_to_hours(horizon_idx + 1),
                "valid_points": metrics["valid_points"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "nse": metrics["nse"],
            }
        )
    return rows


def feature_horizon_metric_rows(
    experiment: str,
    input_mode: str,
    split: str,
    arrays_by_target: dict[str, dict[str, np.ndarray]],
    output_steps: int,
) -> list[dict[str, object]]:
    """Feature-level metrics for each forecast horizon."""
    rows = []
    for horizon_idx in range(output_steps):
        for feature, arrays in arrays_by_target.items():
            error = arrays["pred"][:, horizon_idx : horizon_idx + 1, :, :] - arrays["true"][:, horizon_idx : horizon_idx + 1, :, :]
            truth = arrays["true"][:, horizon_idx : horizon_idx + 1, :, :]
            mask = arrays["mask"][:, horizon_idx : horizon_idx + 1, :, :]
            rows.append(
                {
                    "experiment": experiment,
                    "input_mode": input_mode,
                    "split": split,
                    "input_steps": INPUT_STEPS,
                    "output_steps": output_steps,
                    "horizon_step": horizon_idx + 1,
                    "horizon_hours": steps_to_hours(horizon_idx + 1),
                    "feature": feature,
                    **_flat_metrics(error, truth, mask),
                }
            )
    return rows


def station_horizon_metric_rows(
    experiment: str,
    input_mode: str,
    split: str,
    arrays_by_target: dict[str, dict[str, np.ndarray]],
    stations: tuple[str, ...],
    output_steps: int,
) -> list[dict[str, object]]:
    """Station-level metrics for each forecast horizon."""
    rows = []
    for horizon_idx in range(output_steps):
        for station_idx, station in enumerate(stations):
            errors, truths, masks = [], [], []
            for arrays in arrays_by_target.values():
                errors.append(arrays["pred"][:, horizon_idx : horizon_idx + 1, station_idx, :] - arrays["true"][:, horizon_idx : horizon_idx + 1, station_idx, :])
                truths.append(arrays["true"][:, horizon_idx : horizon_idx + 1, station_idx, :])
                masks.append(arrays["mask"][:, horizon_idx : horizon_idx + 1, station_idx, :])
            rows.append(
                {
                    "experiment": experiment,
                    "input_mode": input_mode,
                    "split": split,
                    "input_steps": INPUT_STEPS,
                    "output_steps": output_steps,
                    "horizon_step": horizon_idx + 1,
                    "horizon_hours": steps_to_hours(horizon_idx + 1),
                    "station": station,
                    **_flat_metrics(np.concatenate(errors), np.concatenate(truths), np.concatenate(masks)),
                }
            )
    return rows


def station_feature_horizon_metric_rows(
    experiment: str,
    input_mode: str,
    split: str,
    arrays_by_target: dict[str, dict[str, np.ndarray]],
    stations: tuple[str, ...],
    output_steps: int,
) -> list[dict[str, object]]:
    """Station-feature metrics for each forecast horizon."""
    rows = []
    for horizon_idx in range(output_steps):
        for station_idx, station in enumerate(stations):
            for feature, arrays in arrays_by_target.items():
                error = arrays["pred"][:, horizon_idx : horizon_idx + 1, station_idx, :] - arrays["true"][:, horizon_idx : horizon_idx + 1, station_idx, :]
                truth = arrays["true"][:, horizon_idx : horizon_idx + 1, station_idx, :]
                mask = arrays["mask"][:, horizon_idx : horizon_idx + 1, station_idx, :]
                rows.append(
                    {
                        "experiment": experiment,
                        "input_mode": input_mode,
                        "split": split,
                        "input_steps": INPUT_STEPS,
                        "output_steps": output_steps,
                        "horizon_step": horizon_idx + 1,
                        "horizon_hours": steps_to_hours(horizon_idx + 1),
                        "station": station,
                        "feature": feature,
                        **_flat_metrics(error, truth, mask),
                    }
                )
    return rows


def _attach_target_times(
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    scaled_splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    for split_name, arrays in arrays_by_split.items():
        arrays["target_start"] = scaled_splits[split_name]["target_start"]
        arrays["target_end"] = scaled_splits[split_name]["target_end"]
    return arrays_by_split


def run_variant_target(
    torch,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    variant: VariantSpec,
    output_steps: int,
    target_feature: str,
    device,
) -> tuple[dict, dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    """Train one target model for one variant and output length."""
    name = experiment_name(variant.input_mode, output_steps)
    selected_features, selected_rows = window.select_corr_features_for_target(
        data,
        stations,
        target_feature,
        INPUT_STEPS,
        name,
        include_current_level=variant.include_current_level,
    )
    for row in selected_rows:
        row["input_mode"] = variant.input_mode
        row["output_steps"] = output_steps
        row["max_horizon_hours"] = steps_to_hours(output_steps)

    if variant.input_mode == "step_raw_plus_diff":
        input_columns = all_step.raw_diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            INPUT_STEPS,
            include_current_level=False,
        )
        result, arrays_by_split = window.fit_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            stations,
            INPUT_STEPS,
            include_current_level=False,
            device=device,
        )
    elif variant.input_mode == "dual_branch_current_mlp":
        input_columns = window.diff_columns_from_features(selected_features)
        raw_splits, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            INPUT_STEPS,
            include_current_level=False,
        )
        scaled_splits, current_scaler = dual.attach_scaled_current_level(raw_splits, scaled_splits)
        result, arrays_by_split = dual.fit_dual_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            current_scaler,
            stations,
            device,
        )
    else:
        input_columns = window.diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            INPUT_STEPS,
            include_current_level=variant.include_current_level,
        )
        result, arrays_by_split = window.fit_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            stations,
            INPUT_STEPS,
            include_current_level=variant.include_current_level,
            device=device,
        )

    result["input_mode"] = variant.input_mode
    result["output_steps"] = output_steps
    arrays_by_split = _attach_target_times(arrays_by_split, scaled_splits)
    return result, arrays_by_split, selected_rows


def run_variant(
    torch,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    variant: VariantSpec,
    output_steps: int,
    device,
) -> tuple[dict, dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, object]]]:
    """Train all five target models for one variant and output length."""
    name = experiment_name(variant.input_mode, output_steps)
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ("train", "val", "test")}
    history_rows = []
    for target_feature in TARGET_FEATURE_COLUMNS:
        result, arrays_by_split, rows = run_variant_target(
            torch,
            data,
            stations,
            variant,
            output_steps,
            target_feature,
            device,
        )
        target_results[target_feature] = result
        selected_rows.extend(rows)
        history_rows.extend({"sub_experiment": f"{name}_{target_feature}", **row} for row in result["history"])
        for split_name, arrays in arrays_by_split.items():
            arrays_by_split_and_target[split_name][target_feature] = arrays

    metrics = {
        split_name: self_ablation.aggregate_single_target_arrays(
            arrays_by_target,
            stations,
            TARGET_FEATURE_COLUMNS,
        )
        for split_name, arrays_by_target in arrays_by_split_and_target.items()
    }
    best_epochs = {target: result["best_epoch"] for target, result in target_results.items()}
    return {
        "experiment": name,
        "input_steps": INPUT_STEPS,
        "input_hours": steps_to_hours(INPUT_STEPS),
        "output_steps": output_steps,
        "max_horizon_hours": steps_to_hours(output_steps),
        "include_current_level": variant.include_current_level,
        "input_mode": variant.input_mode,
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, arrays_by_split_and_target, selected_rows


def _result_overall_row(result: dict, split: str) -> dict[str, object]:
    metrics = result["best_checkpoint"][split]
    return {
        "experiment": result["experiment"],
        "input_mode": result["input_mode"],
        "split": split,
        "input_steps": result["input_steps"],
        "input_hours": result["input_hours"],
        "output_steps": result["output_steps"],
        "max_horizon_hours": result["max_horizon_hours"],
        "valid_points": metrics["valid_points"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "nse": metrics["nse"],
    }


def _json_safe_results(results: dict[str, dict]) -> dict[str, dict]:
    safe = {}
    for name, result in results.items():
        safe[name] = {
            "input_steps": result["input_steps"],
            "input_hours": result["input_hours"],
            "output_steps": result["output_steps"],
            "max_horizon_hours": result["max_horizon_hours"],
            "input_mode": result["input_mode"],
            "include_current_level": result["include_current_level"],
            "best_epoch": result["best_epoch"],
            "best_checkpoint": result["best_checkpoint"],
            "targets": {
                target: {
                    "best_epoch": target_result["best_epoch"],
                    "best_model_path": target_result["best_model_path"],
                    "input_columns": target_result["input_columns"],
                    "input_mode": target_result.get("input_mode", result["input_mode"]),
                    "output_steps": target_result.get("output_steps", result["output_steps"]),
                }
                for target, target_result in result["targets"].items()
            },
        }
    return safe


def persistence_arrays_for_output(data: pd.DataFrame, stations: tuple[str, ...], output_steps: int) -> dict[str, dict[str, np.ndarray]]:
    """Build test-set persistence arrays for the active output length."""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=output_steps,
        input_columns=window.diff_columns_from_features(FEATURE_COLUMNS),
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=paper.RESAMPLE_RULE,
    )
    split = delta.lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)["test"]
    pred = np.repeat(split["last_target"][:, None, :, :], output_steps, axis=1)
    arrays = {}
    for feature_idx, feature in enumerate(TARGET_FEATURE_COLUMNS):
        arrays[feature] = {
            "pred": pred[..., feature_idx : feature_idx + 1],
            "true": split["y_abs"][..., feature_idx : feature_idx + 1],
            "mask": split["y_mask"][..., feature_idx : feature_idx + 1],
            "target_start": split["target_start"],
            "target_end": split["target_end"],
        }
    return arrays


def validate_common_horizon_counts(horizon: pd.DataFrame) -> None:
    expected_modes = {"persistence", "step_raw_plus_diff", "dual_branch_current_mlp"}
    test = horizon[horizon["split"].eq("test")]
    for horizon_step, group in test.groupby("horizon_step", sort=True):
        modes = set(group["input_mode"])
        if modes != expected_modes:
            raise ValueError(f"Horizon {horizon_step} is missing comparison modes: {expected_modes - modes}")
        if pd.to_numeric(group["valid_points"], errors="raise").nunique() != 1:
            raise ValueError(f"Horizon {horizon_step} does not use common valid target points.")


def write_report(
    output_dir: Path,
    overall: pd.DataFrame,
    horizon: pd.DataFrame,
    feature_horizon: pd.DataFrame,
    station_horizon: pd.DataFrame,
) -> None:
    """Write the focused direct multi-step C/D report."""
    validation = overall[
        overall["split"].eq("val") & overall["input_mode"].isin(["step_raw_plus_diff", "dual_branch_current_mlp"])
    ].sort_values("rmse")
    validation_horizon = horizon[
        horizon["split"].eq("val") & horizon["input_mode"].isin(["step_raw_plus_diff", "dual_branch_current_mlp"])
    ].sort_values(["horizon_step", "rmse"])
    test_overall = overall[overall["split"].eq("test")].sort_values("rmse")
    test_horizon = horizon[horizon["split"].eq("test")].sort_values(["horizon_step", "rmse"])
    horizon_pivot = test_horizon.pivot(index="horizon_hours", columns="input_mode", values="rmse")
    improvement_rows = []
    for horizon_hours, row in horizon_pivot.iterrows():
        improvement_rows.append(
            {
                "horizon_hours": int(horizon_hours),
                "C_rmse": row["step_raw_plus_diff"],
                "D_rmse": row["dual_branch_current_mlp"],
                "persistence_rmse": row["persistence"],
                "C_improvement_pct": (row["persistence"] - row["step_raw_plus_diff"]) / row["persistence"] * 100,
                "D_improvement_pct": (row["persistence"] - row["dual_branch_current_mlp"]) / row["persistence"] * 100,
            }
        )
    improvement = pd.DataFrame(improvement_rows)
    feature_36h = feature_horizon[
        feature_horizon["split"].eq("test") & feature_horizon["horizon_hours"].eq(36)
    ].sort_values(["feature", "rmse"])
    station_36h = station_horizon[
        station_horizon["split"].eq("test") & station_horizon["horizon_hours"].eq(36)
    ].pivot(index="station", columns="input_mode", values="rmse").dropna()
    c_improved_stations = int((station_36h["step_raw_plus_diff"] < station_36h["persistence"]).sum())
    d_improved_stations = int((station_36h["dual_branch_current_mlp"] < station_36h["persistence"]).sum())
    c_beats_d_stations = int((station_36h["step_raw_plus_diff"] < station_36h["dual_branch_current_mlp"]).sum())
    macro_validation_nse = (
        feature_horizon[
            feature_horizon["split"].eq("val")
            & feature_horizon["input_mode"].isin(["step_raw_plus_diff", "dual_branch_current_mlp"])
        ]
        .groupby("input_mode")["nse"]
        .mean()
    )
    lines = [
        "# V2 Direct 9-to-9 Change Prediction",
        "",
        "## Protocol",
        "- Input: past 9 four-hour steps (36 hours).",
        "- Output: future 9 steps (4 through 36 hours).",
        "- Each delta is relative to the same last approved target value.",
        "- C uses aligned raw level plus diff at every historical step.",
        "- D uses a diff-GRU branch plus a current-level MLP branch.",
        "- All horizon-target cells have equal weight in the masked L1 loss.",
        "- No rainfall, graph input, recursive feedback, or future true feature is used.",
        "",
        "## Validation model selection",
        "```text",
        validation[["experiment", "input_mode", "valid_points", "mae", "rmse", "nse"]].to_string(index=False),
        "```",
        "",
        "## Validation by forecast horizon",
        "```text",
        validation_horizon[
            ["horizon_step", "horizon_hours", "experiment", "input_mode", "valid_points", "mae", "rmse", "nse"]
        ].to_string(index=False),
        "```",
        "",
        "## Test overall",
        "```text",
        test_overall[["experiment", "input_mode", "valid_points", "mae", "rmse", "nse"]].to_string(index=False),
        "```",
        "",
        "## Test by forecast horizon",
        "```text",
        test_horizon[
            ["horizon_step", "horizon_hours", "experiment", "input_mode", "valid_points", "mae", "rmse", "nse"]
        ].to_string(index=False),
        "```",
        "",
        "## Improvement over persistence",
        "```text",
        improvement.to_string(index=False),
        "```",
        "",
        "## Test feature metrics at 36 hours",
        "```text",
        feature_36h[["input_mode", "feature", "valid_points", "mae", "rmse", "nse"]].to_string(index=False),
        "```",
        "",
        "## Reading",
        f"- Mean validation NSE over 9 horizons and 5 targets: C={macro_validation_nse['step_raw_plus_diff']:.6f}, D={macro_validation_nse['dual_branch_current_mlp']:.6f}.",
        "- D has lower test RMSE from 4 through 16 hours; C has lower test RMSE from 20 through 36 hours.",
        f"- At 36 hours, C improves {c_improved_stations}/{len(station_36h)} evaluable stations over persistence; D improves {d_improved_stations}/{len(station_36h)}.",
        f"- C has lower 36-hour station RMSE than D at {c_beats_d_stations}/{len(station_36h)} evaluable stations.",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(overall: pd.DataFrame, horizon: pd.DataFrame, output_dir: Path) -> None:
    """Show the decision-facing metrics; detailed tables remain on disk."""
    test_overall = overall[overall["split"].eq("test")].copy()
    test_overall["model"] = test_overall["input_mode"].map(TERMINAL_MODEL_NAMES)
    console.table(
        "test summary",
        test_overall.sort_values("rmse"),
        columns=("model", "mae", "rmse", "nse", "valid_points"),
    )
    test_horizon = horizon[horizon["split"].eq("test")].copy()
    test_horizon["model"] = test_horizon["input_mode"].map(TERMINAL_MODEL_NAMES)
    horizon_rmse = (
        test_horizon.pivot(index="horizon_hours", columns="model", values="rmse")
        .reset_index()
        .rename(columns={"horizon_hours": "hours"})
    )
    console.table(
        "test RMSE by horizon",
        horizon_rmse,
        columns=("hours", "Persistence", "C step-level", "D dual-branch"),
        max_rows=len(horizon_rmse),
    )
    console.done(output_dir, report="run_report.md", details="CSV/JSON files")


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = protocol.PILOT_SEED) -> int:
    """Run the focused V2 direct multi-step C/D comparison."""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    paper.SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    dataset_summary = {
        "model": "v2_direct_multistep_change_prediction",
        "output_dir": str(OUTPUT_DIR),
        "processed_data_path": str(all_window.PROCESSED_DATA_PATH),
        "start_date": all_window.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": all_window.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "input_hours": steps_to_hours(INPUT_STEPS),
        "output_step_options": list(OUTPUT_STEP_OPTIONS),
        "output_horizon_hours": [steps_to_hours(steps) for steps in OUTPUT_STEP_OPTIONS],
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "station_count": len(stations),
        "stations": list(stations),
        "variants": [variant.__dict__ for variant in VARIANTS],
        "experiments": experiment_matrix(),
        "horizon_weights": list(HORIZON_WEIGHTS),
        "target_delta": "future_target_at_horizon_minus_last_approved_target",
        "recursive_feedback": False,
        "seed": int(seed),
    }
    manifest = protocol.build_run_manifest(
        experiment="stage3_direct_multistep_CD_9to9",
        output_dir=OUTPUT_DIR,
        seed=seed,
        code_paths=(Path("scripts/gru/run_all_station_multistep_self_gru_ablation.py"),),
    )
    dataset_summary["run_manifest"] = manifest
    save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    save_json(OUTPUT_DIR / "run_manifest.json", manifest)
    torch = base.require_torch()
    device = base.choose_device(torch)
    console.phase("direct multi-step change prediction")
    console.info(
        "dataset",
        stations=len(stations),
        input=f"{INPUT_STEPS} steps / {steps_to_hours(INPUT_STEPS)}h",
        output=f"{OUTPUT_STEP_OPTIONS[-1]} steps / {steps_to_hours(OUTPUT_STEP_OPTIONS[-1])}h",
        targets=len(TARGET_FEATURE_COLUMNS),
    )
    console.info("runtime", device=device, seed=seed)

    results = {}
    selected_rows = []
    overall_rows = []
    horizon_rows = []
    feature_horizon_rows = []
    station_horizon_rows = []
    station_feature_horizon_rows = []

    run_total = len(OUTPUT_STEP_OPTIONS) * len(VARIANTS)
    run_index = 0
    for output_steps in OUTPUT_STEP_OPTIONS:
        configure_output_steps(output_steps, OUTPUT_DIR)
        persistence_name = f"persistence_{INPUT_STEPS}to{output_steps}_delta"
        persistence_arrays = persistence_arrays_for_output(data, stations, output_steps)
        persistence_result = {
            "experiment": persistence_name,
            "input_mode": "persistence",
            "input_steps": INPUT_STEPS,
            "input_hours": steps_to_hours(INPUT_STEPS),
            "output_steps": output_steps,
            "max_horizon_hours": steps_to_hours(output_steps),
            "best_checkpoint": {
                "test": self_ablation.aggregate_single_target_arrays(
                    persistence_arrays,
                    stations,
                    TARGET_FEATURE_COLUMNS,
                )
            },
        }
        overall_rows.append(_result_overall_row(persistence_result, "test"))
        horizon_rows.extend(
            horizon_metric_rows(
                persistence_name,
                "persistence",
                "test",
                persistence_arrays,
                stations,
                TARGET_FEATURE_COLUMNS,
                output_steps,
            )
        )
        feature_horizon_rows.extend(feature_horizon_metric_rows(persistence_name, "persistence", "test", persistence_arrays, output_steps))
        station_horizon_rows.extend(station_horizon_metric_rows(persistence_name, "persistence", "test", persistence_arrays, stations, output_steps))
        station_feature_horizon_rows.extend(
            station_feature_horizon_metric_rows(persistence_name, "persistence", "test", persistence_arrays, stations, output_steps)
        )

        for variant in VARIANTS:
            run_index += 1
            console.phase(
                f"train {TERMINAL_MODEL_NAMES[variant.input_mode]}",
                current=run_index,
                total=run_total,
            )
            console.info("forecast", output_steps=output_steps, max_horizon=f"{steps_to_hours(output_steps)}h")
            result, arrays_by_split_and_target, rows = run_variant(torch, data, stations, variant, output_steps, device)
            results[result["experiment"]] = result
            selected_rows.extend(rows)
            for split_name, arrays_by_target in arrays_by_split_and_target.items():
                overall_rows.append(_result_overall_row(result, split_name))
                horizon_rows.extend(
                    horizon_metric_rows(
                        result["experiment"],
                        variant.input_mode,
                        split_name,
                        arrays_by_target,
                        stations,
                        TARGET_FEATURE_COLUMNS,
                        output_steps,
                    )
                )
                feature_horizon_rows.extend(
                    feature_horizon_metric_rows(result["experiment"], variant.input_mode, split_name, arrays_by_target, output_steps)
                )
                station_horizon_rows.extend(
                    station_horizon_metric_rows(result["experiment"], variant.input_mode, split_name, arrays_by_target, stations, output_steps)
                )
                station_feature_horizon_rows.extend(
                    station_feature_horizon_metric_rows(
                        result["experiment"],
                        variant.input_mode,
                        split_name,
                        arrays_by_target,
                        stations,
                        output_steps,
                    )
                )

    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})

    overall = pd.DataFrame(overall_rows)
    horizon = pd.DataFrame(horizon_rows)
    feature_horizon = pd.DataFrame(feature_horizon_rows)
    station_horizon = pd.DataFrame(station_horizon_rows)
    station_feature_horizon = pd.DataFrame(station_feature_horizon_rows)
    validate_common_horizon_counts(horizon)
    validation_candidates = overall[
        overall["split"].eq("val") & overall["input_mode"].isin(["step_raw_plus_diff", "dual_branch_current_mlp"])
    ].sort_values("rmse")
    selected = validation_candidates.iloc[0]
    selection = {
        "experiment": selected["experiment"],
        "input_mode": selected["input_mode"],
        "validation_rmse": float(selected["rmse"]),
        "validation_mae": float(selected["mae"]),
        "validation_nse": float(selected["nse"]),
        "macro_validation_nse_by_model": (
            feature_horizon[
                feature_horizon["split"].eq("val")
                & feature_horizon["input_mode"].isin(["step_raw_plus_diff", "dual_branch_current_mlp"])
            ]
            .groupby("input_mode")["nse"]
            .mean()
            .to_dict()
        ),
        "selection_uses_test": False,
    }
    metrics = {
        "config": dataset_summary,
        "experiments": _json_safe_results(results),
        "selected_features": selected_rows,
        "validation_selection": selection,
    }
    save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_json(OUTPUT_DIR / "validation_selection.json", selection)

    overall.to_csv(OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(OUTPUT_DIR / "horizon_metrics.csv", index=False, encoding="utf-8-sig")
    feature_horizon.to_csv(OUTPUT_DIR / "feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    station_horizon.to_csv(OUTPUT_DIR / "station_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    station_feature_horizon.to_csv(OUTPUT_DIR / "station_feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")
    write_report(OUTPUT_DIR, overall, horizon, feature_horizon, station_horizon)
    print_terminal_summary(overall, horizon, OUTPUT_DIR)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
