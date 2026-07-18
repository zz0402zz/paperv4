#!/usr/bin/env python3
"""Build the consolidated report for the V2 Stage 1-3 single-step rerun."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
from pathlib import Path

import pandas as pd

from scripts.common import v2_experiment_protocol as protocol
from scripts.data.wq_preprocessing_v2 import write_json


PREFLIGHT_PATH = protocol.PROTOCOL_OUTPUT_ROOT / "stage1_protocol" / "preflight.json"
BASELINE_DIR = protocol.BASELINE_OUTPUT_ROOT / "stage2_baselines_9to1" / "seed_42"
A_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "A_diff_only_9to1_seed42"
C_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "C_step_raw_diff_9to1_seed42"
D_DIR = protocol.GRU_OUTPUT_ROOT / "stage3_change_ablation" / "D_dual_branch_9to1_seed42"
REPORT_DIR = protocol.REPORT_OUTPUT_ROOT


def select_validation_winner(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame[pd.to_numeric(frame["val_rmse"], errors="coerce").notna()].copy()
    if valid.empty:
        raise ValueError("No validation RMSE is available for model selection.")
    valid["val_rmse"] = pd.to_numeric(valid["val_rmse"], errors="raise")
    row = valid.sort_values("val_rmse").iloc[0].to_dict()
    return {
        key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }


def validate_manifest_hashes(manifests: list[dict[str, object]]) -> tuple[str, str]:
    observed = {item["inputs"]["observed"]["sha256"] for item in manifests}
    quality = {item["inputs"]["quality"]["sha256"] for item in manifests}
    if len(observed) != 1 or len(quality) != 1:
        raise ValueError("Stage 1-3 artifacts do not share identical observed and quality hashes.")
    return next(iter(observed)), next(iter(quality))


def validate_common_valid_points(frame: pd.DataFrame) -> int:
    points = pd.to_numeric(frame["valid_points"], errors="raise").astype(int)
    if points.nunique() != 1:
        raise ValueError("A/C/D variants do not share identical valid target points.")
    return int(points.iloc[0])


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validation_feature_rows() -> pd.DataFrame:
    rows = []
    specs = (
        ("A", "diff_only", A_DIR),
        ("C", "step_raw_plus_diff", C_DIR),
        ("D", "dual_branch_current_mlp", D_DIR),
    )
    for label, input_mode, directory in specs:
        metrics = _read_json(directory / "metrics.json")
        experiment, result = next(iter(metrics["experiments"].items()))
        validation = result["best_checkpoint"]["val"]
        for feature in protocol.TARGET_FEATURE_COLUMNS:
            rows.append(
                {
                    "variant": label,
                    "experiment": experiment,
                    "input_mode": input_mode,
                    "feature": feature,
                    "val_rmse": validation["feature_rmse"].get(feature),
                    "val_nse": validation["feature_nse"].get(feature),
                }
            )
    return pd.DataFrame(rows)


def build_report(output_dir: Path = REPORT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _read_json(PREFLIGHT_PATH)
    manifests = [
        _read_json(path / "run_manifest.json")
        for path in (BASELINE_DIR, A_DIR, C_DIR, D_DIR)
    ]
    observed_hash, quality_hash = validate_manifest_hashes(manifests)

    baseline = pd.read_csv(BASELINE_DIR / "summary.csv").sort_values("val_rmse", na_position="last")
    acd = pd.read_csv(D_DIR / "comparison_overall_acd.csv").sort_values("val_rmse")
    acd_features = pd.read_csv(D_DIR / "comparison_feature_acd.csv")
    acd_stations = pd.read_csv(D_DIR / "comparison_station_acd.csv")
    validation_features = validation_feature_rows()
    validate_common_valid_points(acd)
    winner = select_validation_winner(acd)

    baseline.to_csv(output_dir / "baseline_summary_9to1_seed42.csv", index=False, encoding="utf-8-sig")
    acd.to_csv(output_dir / "acd_summary_9to1_seed42.csv", index=False, encoding="utf-8-sig")
    acd_features.to_csv(output_dir / "acd_feature_metrics_9to1_seed42.csv", index=False, encoding="utf-8-sig")
    acd_stations.to_csv(output_dir / "acd_station_metrics_9to1_seed42.csv", index=False, encoding="utf-8-sig")
    validation_features.to_csv(
        output_dir / "acd_validation_feature_metrics_9to1_seed42.csv",
        index=False,
        encoding="utf-8-sig",
    )
    macro_val_nse = validation_features.groupby("variant")["val_nse"].mean().to_dict()

    new_absolute_points = int(baseline.loc[baseline["model"].ne("persistence"), "valid_points"].max())
    new_delta_points = int(pd.to_numeric(acd["valid_points"], errors="coerce").min())

    summary = {
        "task": "past_36h_to_next_4h",
        "input_steps": 9,
        "output_steps": 1,
        "pilot_seed": protocol.PILOT_SEED,
        "observed_sha256": observed_hash,
        "quality_sha256": quality_hash,
        "preflight_checks": preflight["checks"],
        "new_absolute_valid_points": new_absolute_points,
        "new_delta_valid_points": new_delta_points,
        "acd_macro_validation_nse": macro_val_nse,
        "validation_selected_change_backbone": winner,
    }
    write_json(output_dir / "stage1_stage3_summary.json", summary)

    baseline_view = baseline[
        ["model", "best_epoch", "val_rmse", "test_mae", "test_rmse", "test_nse", "valid_points"]
    ]
    acd_view = acd[
        ["experiment", "input_mode", "val_rmse", "test_mae", "test_rmse", "test_nse", "valid_points"]
    ]
    lines = [
        "# V2 Stage 1-3 Single-Step Rerun",
        "",
        "## Protocol",
        "",
        "- Input: past 9 four-hour steps (36 hours).",
        "- Output: next 1 four-hour step.",
        "- Train: 2022-2023; validation: 2024; test: 2025 through May 15.",
        "- Targets use approved original observations only; review interpolation is excluded.",
        "- This is a seed-42 pilot. Multi-step 36-hour output is deferred.",
        f"- V2 preflight checks passed: {sum(value is True or value == 0 for value in preflight['checks'].values())}/{len(preflight['checks'])}.",
        "",
        "## Baselines",
        "",
        "```text",
        baseline_view.to_string(index=False),
        "```",
        "",
        "## Change-Prediction A/C/D",
        "",
        "```text",
        acd_view.to_string(index=False),
        "```",
        "",
        "## A/C/D validation by feature",
        "",
        "```text",
        validation_features[["variant", "feature", "val_rmse", "val_nse"]].to_string(index=False),
        "```",
        "",
        "## Selection",
        "",
        f"- Validation winner: `{winner['input_mode']}` with validation RMSE {float(winner['val_rmse']):.6f}.",
        f"- Its test RMSE is {float(winner['test_rmse']):.6f}; test performance was not used for selection.",
        f"- Absolute baselines use {new_absolute_points} approved test cells; delta models use {new_delta_points} cells because an approved anchor is also required.",
        f"- Mean validation NSE across the five targets: A={macro_val_nse['A']:.6f}, C={macro_val_nse['C']:.6f}, D={macro_val_nse['D']:.6f}.",
        "- D is the provisional seed-42 backbone: it leads pooled validation RMSE and mean validation NSE, mainly through permanganate-index improvement.",
        "- C remains the required graph-stage control because it is better on validation pH and dissolved oxygen and on three of five test features.",
    ]
    (output_dir / "stage1_stage3_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    summary = build_report()
    console.print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
