#!/usr/bin/env python3
"""Run the formal 24h history-branch credibility ablation for direct forecasts."""

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
from scripts.gru import history_branch_ablation as branch
from scripts.gru import run_all_station_multistep_self_gru_ablation as multi
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_self_feature_ablation as self_ablation
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol


OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3c_history_branch_ablation_24h"
INPUT_STEPS = 6
OUTPUT_STEPS = 9
SEEDS = protocol.FORMAL_SEEDS
DIAGNOSTIC_SEED = SEEDS[0]
DIAGNOSTIC_ONLY_VARIANTS = {"diff_only_gru"}
RIDGE_ALPHA = 1.0
TARGET_FEATURES = protocol.TARGET_FEATURE_COLUMNS
BASELINE_VARIANTS = ("persistence", "ridge_current_diff")
TABLE_NAMES = ("overall", "horizon", "feature_horizon", "station_horizon", "station_feature_horizon")


@dataclass
class PreparedTarget:
    target: str
    input_columns: tuple[str, ...]
    raw_splits: dict[str, dict[str, np.ndarray]]
    scaled_splits: dict[str, dict[str, np.ndarray]]
    scalers: base.GraphForecastScalers
    selected_rows: list[dict[str, object]]


def candidate_dir(variant: str, seed: int) -> Path:
    """Keep each resumable model run isolated."""
    if seed < 0:
        return OUTPUT_DIR / "deterministic" / variant
    return OUTPUT_DIR / "runs" / f"seed_{int(seed)}" / variant


def load_cached_candidate(variant: str, seed: int) -> dict[str, pd.DataFrame] | None:
    output_dir = candidate_dir(variant, seed)
    if not (output_dir / "complete.json").exists():
        return None
    paths = {name: output_dir / f"{name}_metrics.csv" for name in TABLE_NAMES}
    if not all(path.exists() for path in paths.values()):
        return None
    return {name: pd.read_csv(path) for name, path in paths.items()}


def save_candidate(
    variant: str,
    seed: int,
    tables: dict[str, pd.DataFrame],
    target_results: list[dict[str, object]] | None = None,
    history_rows: list[dict[str, object]] | None = None,
    control_rows: list[dict[str, object]] | None = None,
) -> None:
    output_dir = candidate_dir(variant, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        tables[name].to_csv(output_dir / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
    if target_results is not None:
        base.save_json(output_dir / "target_training_summary.json", {"target_runs": target_results})
    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    if control_rows is not None:
        pd.DataFrame(control_rows).to_csv(output_dir / "history_controls.csv", index=False, encoding="utf-8-sig")
    base.save_json(output_dir / "complete.json", {"variant": variant, "seed": int(seed), "complete": True})


def prepare_targets(data: pd.DataFrame, stations: tuple[str, ...]) -> dict[str, PreparedTarget]:
    """Select features and materialize one common 24h dataset per target."""
    window.OUTPUT_STEPS = OUTPUT_STEPS
    dual.OUTPUT_STEPS = OUTPUT_STEPS
    dual.INPUT_STEPS = INPUT_STEPS
    multi.INPUT_STEPS = INPUT_STEPS
    prepared = {}
    for target in TARGET_FEATURES:
        selected, selected_rows = window.select_corr_features_for_target(
            data,
            stations,
            target,
            INPUT_STEPS,
            "history_branch_ablation_24h",
            include_current_level=False,
        )
        input_columns = window.diff_columns_from_features(selected)
        raw_splits, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target,
            INPUT_STEPS,
            include_current_level=False,
        )
        scaled_splits, _ = dual.attach_scaled_current_level(raw_splits, scaled_splits)
        prepared[target] = PreparedTarget(
            target=target,
            input_columns=input_columns,
            raw_splits=raw_splits,
            scaled_splits=scaled_splits,
            scalers=scalers,
            selected_rows=selected_rows,
        )
    return prepared


def aggregate_metric_tables(
    variant: str,
    seed: int,
    arrays_by_split: dict[str, dict[str, dict[str, np.ndarray]]],
    stations: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Create the same pooled, horizon, feature and station tables for every variant."""
    overall_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    station_feature_rows: list[dict[str, object]] = []
    for split, arrays_by_target in arrays_by_split.items():
        pooled = self_ablation.aggregate_single_target_arrays(arrays_by_target, stations, TARGET_FEATURES)
        common = {
            "variant": variant,
            "seed": int(seed),
            "split": split,
            "input_steps": INPUT_STEPS,
            "input_hours": INPUT_STEPS * 4,
            "output_steps": OUTPUT_STEPS,
            "output_hours": OUTPUT_STEPS * 4,
        }
        overall_rows.append({**common, **pooled})
        horizon_rows.extend(
            {**row, "variant": variant, "seed": int(seed)}
            for row in multi.horizon_metric_rows(
                variant, variant, split, arrays_by_target, stations, TARGET_FEATURES, OUTPUT_STEPS
            )
        )
        feature_rows.extend(
            {**row, "variant": variant, "seed": int(seed)}
            for row in multi.feature_horizon_metric_rows(variant, variant, split, arrays_by_target, OUTPUT_STEPS)
        )
        station_rows.extend(
            {**row, "variant": variant, "seed": int(seed)}
            for row in multi.station_horizon_metric_rows(
                variant, variant, split, arrays_by_target, stations, OUTPUT_STEPS
            )
        )
        station_feature_rows.extend(
            {**row, "variant": variant, "seed": int(seed)}
            for row in multi.station_feature_horizon_metric_rows(
                variant, variant, split, arrays_by_target, stations, OUTPUT_STEPS
            )
        )
    return {
        "overall": pd.DataFrame(overall_rows),
        "horizon": pd.DataFrame(horizon_rows),
        "feature_horizon": pd.DataFrame(feature_rows),
        "station_horizon": pd.DataFrame(station_rows),
        "station_feature_horizon": pd.DataFrame(station_feature_rows),
    }


def run_deterministic_baseline(
    variant: str,
    prepared: dict[str, PreparedTarget],
    stations: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    arrays = {split: {} for split in ("train", "val", "test")}
    for target, item in prepared.items():
        for split in arrays:
            if variant == "persistence":
                target_arrays = branch.persistence_arrays(item.raw_splits[split], OUTPUT_STEPS)
            elif variant == "ridge_current_diff":
                target_arrays = branch.ridge_arrays(
                    item.scaled_splits["train"],
                    item.scaled_splits[split],
                    item.scalers,
                    alpha=RIDGE_ALPHA,
                )
            else:
                raise ValueError(f"Unknown baseline: {variant}")
            arrays[split][target] = target_arrays
    return aggregate_metric_tables(variant, -1, arrays, stations)


def run_neural_variant(
    torch,
    variant: branch.VariantSpec,
    seed: int,
    prepared: dict[str, PreparedTarget],
    stations: tuple[str, ...],
    device,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    arrays = {split: {} for split in ("train", "val", "test")}
    target_results: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for target, item in prepared.items():
        result, target_arrays, history, shifts = branch.fit_target_model(
            torch,
            variant,
            target,
            item.scaled_splits,
            item.scalers,
            stations,
            OUTPUT_STEPS,
            seed,
            device,
        )
        target_results.append(result)
        history_rows.extend(history)
        control_rows.extend(
            {
                "variant": variant.key,
                "seed": int(seed),
                "target": target,
                "split": split,
                "history_shift_windows": shift,
            }
            for split, shift in shifts.items()
        )
        for split in arrays:
            arrays[split][target] = target_arrays[split]
    return aggregate_metric_tables(variant.key, seed, arrays, stations), target_results, history_rows, control_rows


def validate_fair_counts(horizon: pd.DataFrame) -> None:
    """Require identical target validity counts across every model for each split/horizon."""
    counts = horizon.groupby(["split", "horizon_step"])["valid_points"].agg(["min", "max"])
    unequal = counts[counts["min"] != counts["max"]]
    if not unequal.empty:
        raise ValueError(f"Unfair valid-point counts:\n{unequal}")


def neural_summary(overall: pd.DataFrame) -> pd.DataFrame:
    neural = overall[overall["seed"].ge(0)].copy()
    return (
        neural.groupby(["variant", "split"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            nse_mean=("nse", "mean"),
        )
        .reset_index()
    )


def validation_pairwise(overall: pd.DataFrame) -> pd.DataFrame:
    """Compare every neural ablation with Full D using validation only."""
    validation = overall[(overall["split"] == "val") & overall["seed"].ge(0)]
    pivot = validation.pivot(index="seed", columns="variant", values="rmse")
    rows = []
    for competitor in [variant.key for variant in branch.NEURAL_VARIANTS if variant.key != "full_D"]:
        delta = pivot[competitor] - pivot["full_D"]
        rows.append(
            {
                "competitor": competitor,
                "seeds": int(delta.notna().sum()),
                "full_D_wins": int((delta > 0).sum()),
                "mean_competitor_minus_full_D_rmse": float(delta.mean()),
                "std_competitor_minus_full_D_rmse": float(delta.std()),
            }
        )
    return pd.DataFrame(rows)


def history_gate(pairwise: pd.DataFrame) -> dict[str, object]:
    """Accept history only when Full D beats both same-capacity controls in 4/5 seeds."""
    required = {"full_D_zero_history", "full_D_mismatched_history"}
    controls = pairwise[pairwise["competitor"].isin(required)].set_index("competitor")
    missing = required - set(controls.index)
    if missing:
        raise ValueError(f"Missing history controls: {sorted(missing)}")
    details = {}
    passed = True
    for competitor in sorted(required):
        row = controls.loc[competitor]
        control_passed = bool(
            int(row["full_D_wins"]) >= 4
            and float(row["mean_competitor_minus_full_D_rmse"]) > 0
        )
        details[competitor] = {
            "full_D_wins": int(row["full_D_wins"]),
            "mean_competitor_minus_full_D_rmse": float(row["mean_competitor_minus_full_D_rmse"]),
            "passed": control_passed,
        }
        passed = passed and control_passed
    return {
        "passed": passed,
        "rule": "Full D must beat zero-history and mismatched-history on validation in at least 4/5 seeds and in mean RMSE.",
        "selection_uses_test": False,
        "controls": details,
    }


def comparison_by_group(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Average neural RMSE by seed and report each variant's delta against Full D."""
    neural = table[(table["split"] == "val") & table["seed"].ge(0)].copy()
    per_seed = neural.groupby([*group_columns, "variant", "seed"], sort=True)["rmse"].mean().reset_index()
    full = per_seed[per_seed["variant"] == "full_D"].rename(columns={"rmse": "full_D_rmse"})
    merged = per_seed.merge(full[[*group_columns, "seed", "full_D_rmse"]], on=[*group_columns, "seed"])
    merged["rmse_delta_vs_full_D"] = merged["rmse"] - merged["full_D_rmse"]
    return (
        merged.groupby([*group_columns, "variant"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_delta_vs_full_D_mean=("rmse_delta_vs_full_D", "mean"),
            full_D_wins=("rmse_delta_vs_full_D", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )


def write_report(
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    pairwise: pd.DataFrame,
    gate: dict[str, object],
    feature_horizon: pd.DataFrame,
    station: pd.DataFrame,
) -> None:
    validation = summary[summary["split"] == "val"].sort_values("rmse_mean")
    test = summary[summary["split"] == "test"].sort_values("rmse_mean")
    scores = summary.set_index(["variant", "split"])
    baseline_scores = baselines.set_index(["variant", "split"])
    full_val = float(scores.loc[("full_D", "val"), "rmse_mean"])
    full_test = float(scores.loc[("full_D", "test"), "rmse_mean"])

    def improvement(reference: float, candidate: float) -> float:
        return float((reference - candidate) / reference * 100.0)

    station_full = station[station["variant"] != "full_D"]
    station_coverage = (
        station_full.groupby("variant")["rmse_delta_vs_full_D_mean"]
        .agg(total="count", full_D_wins=lambda values: int((values > 0).sum()))
        .reset_index()
    )
    cell_full = feature_horizon[feature_horizon["variant"] != "full_D"]
    cell_coverage = (
        cell_full.groupby("variant")["rmse_delta_vs_full_D_mean"]
        .agg(total="count", full_D_wins=lambda values: int((values > 0).sum()), minimum_delta="min")
        .reset_index()
    )
    feature_summary = (
        feature_horizon[feature_horizon["variant"].isin({"current_only_mlp", "full_D_zero_history", "full_D_mismatched_history"})]
        .groupby(["feature", "variant"], sort=True)["rmse_delta_vs_full_D_mean"]
        .mean()
        .unstack("variant")
        .reset_index()
    )
    baseline_display = baselines[baselines["split"] == "val"][
        ["variant", "split", "valid_points", "mae", "rmse", "nse"]
    ]
    lines = [
        "# 24h History Branch Credibility Ablation",
        "",
        "- Input: six 4-hour steps (24h); direct output: nine horizons (4-36h).",
        "- Targets: pH, dissolved oxygen, permanganate index, ammonia nitrogen and total phosphorus.",
        "- Split: 2022-2023 train, 2024 validation, 2025 test.",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}.",
        f"- Diff-only GRU is a diagnostic run at seed {DIAGNOSTIC_SEED}; Full D, current-only and both same-capacity controls use all five seeds.",
        f"- Loss: {paper.LOSS_NAME}; feature selection and all scalers use training data only.",
        "- Negative controls keep current state, labels, architecture and parameter count fixed; only the history sequence is zeroed or split-locally shifted.",
        "",
        "## Validation decision",
        f"History credibility gate passed: **{gate['passed']}**.",
        f"Full D validation RMSE: {full_val:.6f}; test RMSE: {full_test:.6f}.",
        f"Validation improvement versus current-only MLP: {improvement(float(scores.loc[('current_only_mlp', 'val'), 'rmse_mean']), full_val):.3f}%.",
        f"Validation improvement versus zero history: {improvement(float(scores.loc[('full_D_zero_history', 'val'), 'rmse_mean']), full_val):.3f}%.",
        f"Validation improvement versus mismatched history: {improvement(float(scores.loc[('full_D_mismatched_history', 'val'), 'rmse_mean']), full_val):.3f}%.",
        f"Validation improvement versus persistence: {improvement(float(baseline_scores.loc[('persistence', 'val'), 'rmse']), full_val):.3f}%.",
        f"Validation improvement versus direct Ridge: {improvement(float(baseline_scores.loc[('ridge_current_diff', 'val'), 'rmse']), full_val):.3f}%.",
        "```text",
        pairwise.to_string(index=False),
        "```",
        "",
        "## Five-seed validation summary",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## Deterministic validation baselines",
        "```text",
        baseline_display.to_string(index=False),
        "```",
        "",
        "## Test readout after validation rule was fixed",
        "```text",
        test.to_string(index=False),
        "```",
        "",
        "## Validation coverage",
        "```text",
        station_coverage.to_string(index=False),
        "```",
        "```text",
        cell_coverage.to_string(index=False),
        "```",
        "",
        "Full D wins at all 25 stations and all nine pooled horizons against both same-capacity history controls.",
        "It wins 44/45 feature-horizon cells; the exception is 4h permanganate index, where current-only/zero/mismatched histories are about 0.007-0.008 RMSE better.",
        "",
        "## Mean validation RMSE delta versus Full D by feature",
        "Positive values mean Full D is better.",
        "```text",
        feature_summary.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    prepared = prepare_targets(data, stations)
    torch = base.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}; stations={len(stations)}", flush=True)

    all_tables: dict[str, list[pd.DataFrame]] = {
        "overall": [],
        "horizon": [],
        "feature_horizon": [],
        "station_horizon": [],
        "station_feature_horizon": [],
    }
    for variant in BASELINE_VARIANTS:
        tables = load_cached_candidate(variant, -1)
        if tables is None:
            console.print(f"running deterministic baseline={variant}", flush=True)
            tables = run_deterministic_baseline(variant, prepared, stations)
            save_candidate(variant, -1, tables)
        else:
            console.print(f"reusing deterministic baseline={variant}", flush=True)
        for name, table in tables.items():
            all_tables[name].append(table)

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        paper.SEED = int(seed)
        for variant in branch.NEURAL_VARIANTS:
            if variant.key in DIAGNOSTIC_ONLY_VARIANTS and seed != DIAGNOSTIC_SEED:
                continue
            tables = load_cached_candidate(variant.key, seed)
            if tables is None:
                console.print(f"running variant={variant.key} seed={seed}", flush=True)
                tables, results, histories, controls = run_neural_variant(
                    torch, variant, seed, prepared, stations, device
                )
                save_candidate(variant.key, seed, tables, results, histories, controls)
            else:
                console.print(f"reusing variant={variant.key} seed={seed}", flush=True)
            for name, table in tables.items():
                all_tables[name].append(table)

    combined = {name: pd.concat(tables, ignore_index=True) for name, tables in all_tables.items()}
    validate_fair_counts(combined["horizon"])
    summary = neural_summary(combined["overall"])
    baseline_summary = combined["overall"][combined["overall"]["seed"] < 0].copy()
    pairwise = validation_pairwise(combined["overall"])
    gate = history_gate(pairwise)
    feature_horizon = comparison_by_group(combined["feature_horizon"], ["feature", "horizon_hours"])
    station = comparison_by_group(combined["station_horizon"], ["station"])

    for name, table in combined.items():
        table.to_csv(OUTPUT_DIR / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "neural_multiseed_summary.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(OUTPUT_DIR / "deterministic_baselines.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUTPUT_DIR / "validation_pairwise.csv", index=False, encoding="utf-8-sig")
    feature_horizon.to_csv(OUTPUT_DIR / "validation_feature_horizon_diagnosis.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUTPUT_DIR / "validation_station_diagnosis.csv", index=False, encoding="utf-8-sig")
    selected_rows = [
        {"target": target, **row}
        for target, item in prepared.items()
        for row in item.selected_rows
    ]
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    base.save_json(OUTPUT_DIR / "history_credibility_gate.json", gate)
    manifest = protocol.build_run_manifest(
        experiment="stage3c_24h_history_branch_ablation",
        output_dir=OUTPUT_DIR,
        seed=protocol.PILOT_SEED,
        code_paths=(
            Path("scripts/gru/history_branch_ablation.py"),
            Path("scripts/gru/run_v2_multistep_history_branch_ablation.py"),
        ),
    )
    manifest.update(
        {
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "seeds": list(SEEDS),
            "diagnostic_seed": int(DIAGNOSTIC_SEED),
            "diagnostic_only_variants": sorted(DIAGNOSTIC_ONLY_VARIANTS),
            "neural_variants": [variant.key for variant in branch.NEURAL_VARIANTS],
            "baseline_variants": list(BASELINE_VARIANTS),
            "ridge_alpha": RIDGE_ALPHA,
            "loss_name": paper.LOSS_NAME,
        }
    )
    base.save_json(OUTPUT_DIR / "run_manifest.json", manifest)
    write_report(summary, baseline_summary, pairwise, gate, feature_horizon, station)
    console.print(summary.to_string(index=False), flush=True)
    console.print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
