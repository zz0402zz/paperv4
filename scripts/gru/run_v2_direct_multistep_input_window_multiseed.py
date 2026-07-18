#!/usr/bin/env python3
"""Check the D direct multi-step input-window choice across random seeds."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_all_station_multistep_self_gru_ablation as multi
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_v2_direct_multistep_input_window_ablation as single_seed
from scripts.common import v2_experiment_protocol as protocol

OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_direct_multistep" / "input_window_multiseed_D_9out"
CANDIDATE_INPUT_STEPS = (9, 12, 18, 24)
SEEDS = (42, 52, 62, 72, 82)
OUTPUT_STEPS = 9
INPUT_MODE = "dual_branch_current_mlp"
LOSS_NAME = paper.LOSS_NAME
REFERENCE_INPUT_STEPS = 9
TABLE_FILES = single_seed.CANDIDATE_TABLE_FILES


def candidate_dir(seed: int, input_steps: int) -> Path:
    """Return the isolated output folder for one seed/window run."""
    return OUTPUT_DIR / f"seed_{int(seed)}" / f"input_{int(input_steps)}step_{int(input_steps) * 4}h"


def _d_variant() -> multi.VariantSpec:
    return next(variant for variant in multi.VARIANTS if variant.input_mode == INPUT_MODE)


def _with_candidate_columns(
    tables: dict[str, pd.DataFrame],
    seed: int,
    input_steps: int,
    source: str,
) -> dict[str, pd.DataFrame]:
    enriched = {}
    for name, table in tables.items():
        frame = table.copy()
        frame["seed"] = int(seed)
        frame["input_steps"] = int(input_steps)
        frame["input_hours"] = int(input_steps) * 4
        frame["output_steps"] = OUTPUT_STEPS
        frame["candidate_source"] = source
        enriched[name] = frame
    return enriched


def load_multiseed_cached_candidate(seed: int, input_steps: int) -> dict[str, pd.DataFrame] | None:
    """Load a candidate produced by this multiseed runner."""
    output_dir = candidate_dir(seed, input_steps)
    required = [output_dir / filename for filename in TABLE_FILES.values()]
    required.append(output_dir / "run_manifest.json")
    if not all(path.exists() for path in required):
        return None
    tables = {name: pd.read_csv(output_dir / filename) for name, filename in TABLE_FILES.items()}
    return _with_candidate_columns(tables, seed, input_steps, "multiseed_cache")


def load_seed42_prior_candidate(seed: int, input_steps: int) -> dict[str, pd.DataFrame] | None:
    """Reuse the already completed seed-42 pilot tables when available."""
    if int(seed) != protocol.PILOT_SEED:
        return None
    if int(input_steps) == REFERENCE_INPUT_STEPS:
        tables = single_seed.load_existing_9step()
    else:
        tables = single_seed.load_cached_candidate(input_steps)
        if tables is None:
            return None
    return _with_candidate_columns(tables, seed, input_steps, "reused_seed42_prior")


def load_cached_candidate(seed: int, input_steps: int) -> dict[str, pd.DataFrame] | None:
    """Load this runner's cache, falling back to compatible seed-42 prior runs."""
    cached = load_multiseed_cached_candidate(seed, input_steps)
    if cached is not None:
        return cached
    return load_seed42_prior_candidate(seed, input_steps)


def run_candidate(torch, data: pd.DataFrame, stations: tuple[str, ...], seed: int, input_steps: int, device) -> dict[str, pd.DataFrame]:
    """Train one D candidate and return all metric tables."""
    output_dir = candidate_dir(seed, input_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    paper.SEED = int(seed)
    multi.INPUT_STEPS = int(input_steps)
    multi.configure_output_steps(OUTPUT_STEPS, output_dir)

    result, arrays_by_split, selected_rows = multi.run_variant(
        torch,
        data,
        stations,
        _d_variant(),
        OUTPUT_STEPS,
        device,
    )

    overall_rows = []
    horizon_rows = []
    feature_rows = []
    station_rows = []
    station_feature_rows = []
    for split, arrays in arrays_by_split.items():
        overall_rows.append(multi._result_overall_row(result, split))
        horizon_rows.extend(
            multi.horizon_metric_rows(
                result["experiment"],
                INPUT_MODE,
                split,
                arrays,
                stations,
                multi.TARGET_FEATURE_COLUMNS,
                OUTPUT_STEPS,
            )
        )
        feature_rows.extend(multi.feature_horizon_metric_rows(result["experiment"], INPUT_MODE, split, arrays, OUTPUT_STEPS))
        station_rows.extend(
            multi.station_horizon_metric_rows(result["experiment"], INPUT_MODE, split, arrays, stations, OUTPUT_STEPS)
        )
        station_feature_rows.extend(
            multi.station_feature_horizon_metric_rows(
                result["experiment"],
                INPUT_MODE,
                split,
                arrays,
                stations,
                OUTPUT_STEPS,
            )
        )

    tables = _with_candidate_columns(
        {
            "overall": pd.DataFrame(overall_rows),
            "horizon": pd.DataFrame(horizon_rows),
            "feature_horizon": pd.DataFrame(feature_rows),
            "station_horizon": pd.DataFrame(station_rows),
            "station_feature_horizon": pd.DataFrame(station_feature_rows),
        },
        seed,
        input_steps,
        "trained_multiseed",
    )
    for name, table in tables.items():
        table.to_csv(output_dir / TABLE_FILES[name], index=False, encoding="utf-8-sig")

    selected = pd.DataFrame(selected_rows)
    selected["seed"] = int(seed)
    selected["input_steps"] = int(input_steps)
    selected.to_csv(output_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
    history = pd.DataFrame(result["history"])
    history["seed"] = int(seed)
    history["input_steps"] = int(input_steps)
    history.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")

    manifest = protocol.build_run_manifest(
        experiment=f"D_direct_{input_steps}to{OUTPUT_STEPS}_input_window_multiseed_seed{seed}",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/gru/run_v2_direct_multistep_input_window_multiseed.py"),
            Path("scripts/gru/run_all_station_multistep_self_gru_ablation.py"),
        ),
    )
    manifest.update(
        {
            "input_steps": int(input_steps),
            "input_hours": int(input_steps) * 4,
            "output_steps": OUTPUT_STEPS,
            "output_hours": OUTPUT_STEPS * 4,
            "input_mode": INPUT_MODE,
            "loss_name": LOSS_NAME,
        }
    )
    base.save_json(output_dir / "run_manifest.json", manifest)
    base.save_json(
        output_dir / "metrics.json",
        {
            "manifest": manifest,
            "experiment": multi._json_safe_results({result["experiment"]: result}),
            "selected_features": selected_rows,
        },
    )
    return tables


def validate_complete_seed_window_grid(
    frame: pd.DataFrame,
    seeds: tuple[int, ...] = SEEDS,
    input_steps: tuple[int, ...] = CANDIDATE_INPUT_STEPS,
) -> None:
    """Require every seed/window pair before final cross-seed selection."""
    required = {(int(seed), int(steps)) for seed in seeds for steps in input_steps}
    observed = {
        (int(row.seed), int(row.input_steps))
        for row in frame[["seed", "input_steps"]].drop_duplicates().itertuples(index=False)
    }
    missing = sorted(required - observed)
    extra = sorted(observed - required)
    if missing or extra:
        raise ValueError(f"Incomplete seed/window grid. missing={missing}; extra={extra}")


def per_seed_validation_ranking(overall: pd.DataFrame, feature_horizon: pd.DataFrame) -> pd.DataFrame:
    """Compute one validation selection score per seed/window."""
    macro = (
        feature_horizon[feature_horizon["split"].eq("val")]
        .groupby(["seed", "input_steps"], sort=True)["nse"]
        .mean()
        .rename("macro_validation_nse")
        .reset_index()
    )
    pooled = (
        overall[overall["split"].eq("val")]
        .set_index(["seed", "input_steps"])[["rmse", "mae", "nse", "valid_points"]]
        .rename(
            columns={
                "rmse": "pooled_validation_rmse",
                "mae": "pooled_validation_mae",
                "nse": "pooled_validation_nse",
                "valid_points": "validation_valid_points",
            }
        )
    )
    ranking = macro.join(pooled, on=["seed", "input_steps"])
    ranking["input_hours"] = ranking["input_steps"].astype(int) * 4
    return ranking.sort_values(["seed", "macro_validation_nse", "pooled_validation_rmse"], ascending=[True, False, True])


def multiseed_validation_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation scores across seeds for each input length."""
    frame = per_seed.copy()
    if "input_hours" not in frame.columns:
        frame["input_hours"] = frame["input_steps"].astype(int) * 4
    if "pooled_validation_mae" not in frame.columns:
        frame["pooled_validation_mae"] = np.nan
    grouped = frame.groupby("input_steps", sort=True)
    summary = grouped.agg(
        input_hours=("input_hours", "first"),
        seeds_completed=("seed", "nunique"),
        macro_validation_nse_mean=("macro_validation_nse", "mean"),
        macro_validation_nse_std=("macro_validation_nse", "std"),
        pooled_validation_rmse_mean=("pooled_validation_rmse", "mean"),
        pooled_validation_rmse_std=("pooled_validation_rmse", "std"),
        pooled_validation_mae_mean=("pooled_validation_mae", "mean"),
    ).reset_index()
    return summary.sort_values(
        ["macro_validation_nse_mean", "pooled_validation_rmse_mean"],
        ascending=[False, True],
    )


def select_multiseed_window(summary: pd.DataFrame) -> dict[str, object]:
    """Select the cross-seed validation winner."""
    if summary.empty:
        raise ValueError("Cannot select from an empty validation summary.")
    row = summary.sort_values(
        ["macro_validation_nse_mean", "pooled_validation_rmse_mean"],
        ascending=[False, True],
    ).iloc[0]
    return {key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value for key, value in row.items()}


def validation_station_win_rates(station_horizon: pd.DataFrame, reference_steps: int = REFERENCE_INPUT_STEPS) -> pd.DataFrame:
    """Count station/seed pairs where a window improves validation RMSE versus the reference."""
    val = station_horizon[station_horizon["split"].eq("val")].copy()
    station_scores = (
        val.groupby(["seed", "input_steps", "station"], sort=True)["rmse"]
        .mean()
        .rename("validation_station_rmse")
        .reset_index()
    )
    reference = station_scores[station_scores["input_steps"].eq(reference_steps)].rename(
        columns={"validation_station_rmse": "reference_validation_station_rmse"}
    )
    merged = station_scores.merge(
        reference[["seed", "station", "reference_validation_station_rmse"]],
        on=["seed", "station"],
        how="inner",
    )
    merged = merged[~merged["input_steps"].eq(reference_steps)].copy()
    merged["improved"] = merged["validation_station_rmse"] < merged["reference_validation_station_rmse"]
    rows = []
    for input_steps, group in merged.groupby("input_steps", sort=True):
        improved = int(group["improved"].sum())
        total = int(len(group))
        rows.append(
            {
                "input_steps": int(input_steps),
                "input_hours": int(input_steps) * 4,
                "station_seed_pairs": total,
                "improved_pairs": improved,
                "improved_pair_rate": float(improved / total) if total else None,
                "mean_rmse_delta_vs_36h": float(
                    (group["validation_station_rmse"] - group["reference_validation_station_rmse"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def split_summary(overall: pd.DataFrame, feature_horizon: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation/test pooled and macro feature-horizon metrics by seed/window."""
    macro = (
        feature_horizon.groupby(["split", "seed", "input_steps"], sort=True)["nse"]
        .mean()
        .rename("macro_feature_horizon_nse")
        .reset_index()
    )
    pooled = overall.rename(columns={"rmse": "pooled_rmse", "mae": "pooled_mae", "nse": "pooled_nse"})
    return macro.merge(
        pooled[["split", "seed", "input_steps", "input_hours", "valid_points", "pooled_mae", "pooled_rmse", "pooled_nse"]],
        on=["split", "seed", "input_steps"],
        how="left",
    )


def write_report(
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    station_wins: pd.DataFrame,
    selected: dict[str, object],
    split_scores: pd.DataFrame,
) -> None:
    """Write a compact markdown report for the multiseed check."""
    selected_steps = int(selected["input_steps"])
    reference = summary.set_index("input_steps").loc[REFERENCE_INPUT_STEPS]
    selected_row = summary.set_index("input_steps").loc[selected_steps]
    nse_gain = float(selected_row["macro_validation_nse_mean"] - reference["macro_validation_nse_mean"])
    lines = [
        "# V2 Direct Multi-Step Input Window Multiseed Check",
        "",
        "- Model: D change-GRU, dual branch current-level MLP.",
        f"- Loss: `{LOSS_NAME}`; unchanged from the current D model.",
        "- Output: future 9 steps, 4 through 36 hours.",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}.",
        "- Selection: validation macro NSE averaged across seeds; pooled validation RMSE breaks ties.",
        "",
        "## Cross-seed validation summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        f"Selected input: {selected_steps} steps ({selected_steps * 4} hours).",
        f"Mean macro validation NSE change versus 36h input: {nse_gain:+.6f}.",
        "",
        "## Per-seed validation ranking",
        "```text",
        per_seed.to_string(index=False),
        "```",
        "",
        "## Station validation win rates versus 36h",
        "```text",
        station_wins.to_string(index=False),
        "```",
        "",
        "## Split-level scores",
        "```text",
        split_scores.sort_values(["split", "input_steps", "seed"]).to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    torch = base.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)

    tables_by_pair: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    for seed in SEEDS:
        for input_steps in CANDIDATE_INPUT_STEPS:
            cached = load_cached_candidate(seed, input_steps)
            if cached is not None:
                console.print(f"reusing seed={seed} input_steps={input_steps} ({input_steps * 4}h)", flush=True)
                tables_by_pair[(seed, input_steps)] = cached
                continue
            console.print(f"running seed={seed} input_steps={input_steps} ({input_steps * 4}h)", flush=True)
            tables_by_pair[(seed, input_steps)] = run_candidate(torch, data, stations, seed, input_steps, device)

    def combine(name: str) -> pd.DataFrame:
        return pd.concat(
            [tables[name] for _, tables in sorted(tables_by_pair.items())],
            ignore_index=True,
        )

    overall = combine("overall")
    horizon = combine("horizon")
    feature_horizon = combine("feature_horizon")
    station_horizon = combine("station_horizon")
    station_feature_horizon = combine("station_feature_horizon")
    validate_complete_seed_window_grid(overall)
    single_seed.validate_common_horizon_counts(horizon, CANDIDATE_INPUT_STEPS)

    per_seed = per_seed_validation_ranking(overall, feature_horizon)
    summary = multiseed_validation_summary(per_seed)
    selected = select_multiseed_window(summary)
    station_wins = validation_station_win_rates(station_horizon)
    split_scores = split_summary(overall, feature_horizon)

    outputs = {
        "overall_comparison": overall,
        "horizon_comparison": horizon,
        "feature_horizon_comparison": feature_horizon,
        "station_horizon_comparison": station_horizon,
        "station_feature_horizon_comparison": station_feature_horizon,
        "per_seed_validation_ranking": per_seed,
        "multiseed_validation_summary": summary,
        "station_validation_win_rates": station_wins,
        "split_summary": split_scores,
    }
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    selection = {
        **selected,
        "output_steps": OUTPUT_STEPS,
        "output_hours": OUTPUT_STEPS * 4,
        "seeds": list(SEEDS),
        "candidate_input_steps": list(CANDIDATE_INPUT_STEPS),
        "loss_name": LOSS_NAME,
        "input_mode": INPUT_MODE,
        "selection_uses_test": False,
    }
    base.save_json(OUTPUT_DIR / "selected_input_window_multiseed.json", selection)
    base.save_json(
        OUTPUT_DIR / "run_manifest.json",
        protocol.build_run_manifest(
            experiment="D_direct_multistep_input_window_multiseed",
            output_dir=OUTPUT_DIR,
            seed=protocol.PILOT_SEED,
            code_paths=(Path("scripts/gru/run_v2_direct_multistep_input_window_multiseed.py"),),
        )
        | {
            "seeds": list(SEEDS),
            "candidate_input_steps": list(CANDIDATE_INPUT_STEPS),
            "output_steps": OUTPUT_STEPS,
            "loss_name": LOSS_NAME,
        },
    )
    write_report(per_seed, summary, station_wins, selection, split_scores)
    console.print(summary.to_string(index=False), flush=True)
    console.print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
