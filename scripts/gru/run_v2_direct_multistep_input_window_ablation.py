#!/usr/bin/env python3
"""Select the D direct multi-step model's historical input length on V2."""

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
from scripts.common import v2_experiment_protocol as protocol

OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_direct_multistep" / "input_window_ablation_D_9out_seed42"
EXISTING_9STEP_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_direct_multistep" / "CD_9to9_seed42"
CANDIDATE_INPUT_STEPS = (3, 6, 9, 12)
EXTENSION_INPUT_STEPS = (18, 24, 30)
OUTPUT_STEPS = 9
INPUT_MODE = "dual_branch_current_mlp"
SEED = protocol.PILOT_SEED
CANDIDATE_TABLE_FILES = {
    "overall": "overall.csv",
    "horizon": "horizon.csv",
    "feature_horizon": "feature_horizon.csv",
    "station_horizon": "station_horizon.csv",
    "station_feature_horizon": "station_feature_horizon.csv",
}


def select_validation_window(frame: pd.DataFrame) -> dict[str, object]:
    required = {"input_steps", "macro_validation_nse", "pooled_validation_rmse"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing validation selection columns: {sorted(missing)}")
    ranked = frame.sort_values(
        ["macro_validation_nse", "pooled_validation_rmse"],
        ascending=[False, True],
    )
    row = ranked.iloc[0].to_dict()
    return {
        key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }


def next_extension_step(selected: dict[str, object], tested_steps: tuple[int, ...]) -> int | None:
    current = int(selected["input_steps"])
    if current != max(tested_steps):
        return None
    return next((step for step in EXTENSION_INPUT_STEPS if step > current and step not in tested_steps), None)


def validate_common_horizon_counts(frame: pd.DataFrame, expected_steps: tuple[int, ...]) -> None:
    expected = set(expected_steps)
    for (split, horizon_step), group in frame.groupby(["split", "horizon_step"], sort=True):
        if split not in {"val", "test"}:
            continue
        found = set(pd.to_numeric(group["input_steps"], errors="raise").astype(int))
        if found != expected:
            raise ValueError(f"{split} horizon {horizon_step} is missing input windows: {expected - found}")
        if pd.to_numeric(group["valid_points"], errors="raise").nunique() != 1:
            raise ValueError(f"{split} horizon {horizon_step} has window-specific target counts.")


def _d_variant() -> multi.VariantSpec:
    return next(variant for variant in multi.VARIANTS if variant.input_mode == INPUT_MODE)


def _candidate_dir(input_steps: int) -> Path:
    return OUTPUT_DIR / f"input_{input_steps}step_{input_steps * 4}h"


def load_cached_candidate(input_steps: int) -> dict[str, pd.DataFrame] | None:
    output_dir = _candidate_dir(input_steps)
    required = [output_dir / filename for filename in CANDIDATE_TABLE_FILES.values()]
    required.append(output_dir / "run_manifest.json")
    if not all(path.exists() for path in required):
        return None
    return {name: pd.read_csv(output_dir / filename) for name, filename in CANDIDATE_TABLE_FILES.items()}


def run_candidate(torch, data: pd.DataFrame, stations: tuple[str, ...], input_steps: int, device) -> dict[str, pd.DataFrame]:
    output_dir = _candidate_dir(input_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                result["experiment"], INPUT_MODE, split, arrays, stations, multi.TARGET_FEATURE_COLUMNS, OUTPUT_STEPS
            )
        )
        feature_rows.extend(multi.feature_horizon_metric_rows(result["experiment"], INPUT_MODE, split, arrays, OUTPUT_STEPS))
        station_rows.extend(
            multi.station_horizon_metric_rows(result["experiment"], INPUT_MODE, split, arrays, stations, OUTPUT_STEPS)
        )
        station_feature_rows.extend(
            multi.station_feature_horizon_metric_rows(
                result["experiment"], INPUT_MODE, split, arrays, stations, OUTPUT_STEPS
            )
        )

    tables = {
        "overall": pd.DataFrame(overall_rows),
        "horizon": pd.DataFrame(horizon_rows),
        "feature_horizon": pd.DataFrame(feature_rows),
        "station_horizon": pd.DataFrame(station_rows),
        "station_feature_horizon": pd.DataFrame(station_feature_rows),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["history"]).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")

    manifest = protocol.build_run_manifest(
        experiment=f"D_direct_{input_steps}to{OUTPUT_STEPS}_input_window_ablation",
        output_dir=output_dir,
        seed=SEED,
        code_paths=(Path("scripts/gru/run_v2_direct_multistep_input_window_ablation.py"),),
    )
    manifest["input_steps"] = int(input_steps)
    manifest["input_hours"] = int(input_steps * 4)
    manifest["output_steps"] = OUTPUT_STEPS
    manifest["output_hours"] = OUTPUT_STEPS * 4
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


def load_existing_9step() -> dict[str, pd.DataFrame]:
    if not (EXISTING_9STEP_DIR / "run_manifest.json").exists():
        raise FileNotFoundError("The matching V2 9-to-9 result is missing; rerun the direct multi-step experiment first.")

    def filtered(filename: str) -> pd.DataFrame:
        frame = pd.read_csv(EXISTING_9STEP_DIR / filename)
        return frame[frame["input_mode"].eq(INPUT_MODE)].copy()

    return {
        "overall": filtered("overall_summary.csv"),
        "horizon": filtered("horizon_metrics.csv"),
        "feature_horizon": filtered("feature_horizon_metrics.csv"),
        "station_horizon": filtered("station_horizon_metrics.csv"),
        "station_feature_horizon": filtered("station_feature_horizon_metrics.csv"),
    }


def validation_ranking(overall: pd.DataFrame, feature_horizon: pd.DataFrame) -> pd.DataFrame:
    pooled = overall[overall["split"].eq("val")].set_index("input_steps")
    macro = (
        feature_horizon[feature_horizon["split"].eq("val")]
        .groupby("input_steps", sort=True)["nse"]
        .mean()
    )
    rows = []
    for input_steps in sorted(macro.index.astype(int)):
        item = pooled.loc[input_steps]
        rows.append(
            {
                "input_steps": input_steps,
                "input_hours": input_steps * 4,
                "macro_validation_nse": float(macro.loc[input_steps]),
                "pooled_validation_rmse": float(item["rmse"]),
                "pooled_validation_mae": float(item["mae"]),
                "pooled_validation_nse": float(item["nse"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_validation_nse", "pooled_validation_rmse"],
        ascending=[False, True],
    )


def write_report(ranking: pd.DataFrame, horizon: pd.DataFrame, selected: dict[str, object]) -> None:
    validation_horizon = horizon[horizon["split"].eq("val")].sort_values(["horizon_hours", "input_steps"])
    test_horizon = horizon[horizon["split"].eq("test")].sort_values(["horizon_hours", "input_steps"])
    selected_steps = int(selected["input_steps"])
    selected_row = ranking.set_index("input_steps").loc[selected_steps]
    reference_steps = 9
    reference_row = ranking.set_index("input_steps").loc[reference_steps]
    nse_gain = float(selected_row["macro_validation_nse"] - reference_row["macro_validation_nse"])
    rmse_change = float(
        (selected_row["pooled_validation_rmse"] / reference_row["pooled_validation_rmse"] - 1.0) * 100.0
    )
    lines = [
        "# V2 Direct Multi-Step Input Window Ablation",
        "",
        "- Model: D direct multi-step change prediction.",
        "- Output: future 9 steps (4 through 36 hours).",
        "- Selection: highest macro validation NSE; pooled validation RMSE is the tie-breaker.",
        f"- Tested input windows: {', '.join(str(value) + 'h' for value in sorted(ranking['input_hours'].astype(int)))}.",
        "- Boundary search extends to 96 and 120 hours only while the longest tested window remains best.",
        "- Test metrics are reported only after validation selection.",
        "",
        "## Validation ranking",
        "```text",
        ranking.to_string(index=False),
        "```",
        "",
        f"Selected input: {int(selected['input_steps'])} steps ({int(selected['input_hours'])} hours).",
        f"Compared with the 36-hour input, macro validation NSE changes by {nse_gain:+.6f} and pooled validation RMSE by {rmse_change:+.3f}%.",
        "The selected duration is the numerical validation winner; the small margin should be described as a modest, not large, gain.",
        "",
        "## Validation by horizon",
        "```text",
        validation_horizon[["input_steps", "input_hours", "horizon_hours", "valid_points", "mae", "rmse", "nse"]].to_string(index=False),
        "```",
        "",
        "## Test by horizon",
        "```text",
        test_horizon[["input_steps", "input_hours", "horizon_hours", "valid_points", "mae", "rmse", "nse"]].to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    paper.SEED = SEED
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    torch = base.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)

    tables_by_step: dict[int, dict[str, pd.DataFrame]] = {9: load_existing_9step()}
    for input_steps in (3, 6, 12):
        cached = load_cached_candidate(input_steps)
        if cached is not None:
            console.print(f"reusing D direct input_steps={input_steps} ({input_steps * 4}h)", flush=True)
            tables_by_step[input_steps] = cached
        else:
            console.print(f"running D direct input_steps={input_steps} ({input_steps * 4}h)", flush=True)
            tables_by_step[input_steps] = run_candidate(torch, data, stations, input_steps, device)

    def combine(name: str) -> pd.DataFrame:
        return pd.concat([tables[name] for _, tables in sorted(tables_by_step.items())], ignore_index=True)

    while True:
        overall = combine("overall")
        horizon = combine("horizon")
        feature_horizon = combine("feature_horizon")
        ranking = validation_ranking(overall, feature_horizon)
        selected = select_validation_window(ranking)
        extension = next_extension_step(selected, tuple(sorted(tables_by_step)))
        if extension is None:
            break
        cached = load_cached_candidate(extension)
        if cached is not None:
            console.print(f"reusing boundary input_steps={extension} ({extension * 4}h)", flush=True)
            tables_by_step[extension] = cached
        else:
            console.print(f"boundary winner found; running input_steps={extension} ({extension * 4}h)", flush=True)
            tables_by_step[extension] = run_candidate(torch, data, stations, extension, device)

    expected_steps = tuple(sorted(tables_by_step))
    validate_common_horizon_counts(horizon, expected_steps)
    aggregate_tables = {
        "overall_comparison": overall,
        "horizon_comparison": horizon,
        "feature_horizon_comparison": feature_horizon,
        "station_horizon_comparison": combine("station_horizon"),
        "station_feature_horizon_comparison": combine("station_feature_horizon"),
        "validation_ranking": ranking,
    }
    for name, table in aggregate_tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    selection = {
        **selected,
        "output_steps": OUTPUT_STEPS,
        "output_hours": OUTPUT_STEPS * 4,
        "seed": SEED,
        "selection_uses_test": False,
        "tested_input_steps": list(expected_steps),
        "boundary_search_limit_steps": max(EXTENSION_INPUT_STEPS),
    }
    base.save_json(OUTPUT_DIR / "selected_input_window.json", selection)
    base.save_json(
        OUTPUT_DIR / "run_manifest.json",
        protocol.build_run_manifest(
            experiment="D_direct_multistep_input_window_ablation",
            output_dir=OUTPUT_DIR,
            seed=SEED,
            code_paths=(Path("scripts/gru/run_v2_direct_multistep_input_window_ablation.py"),),
        ),
    )
    write_report(ranking, horizon, selection)
    console.print(ranking.to_string(index=False), flush=True)
    console.print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
