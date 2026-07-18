"""Five-seed validation gates and sealed-test evaluation for per-target inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_paper_style as paper
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.terminal_output import console
from scripts.gru import run_v2_hourly_representation_ablation as hourly
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.gru import target_input_group_ablation as ablation


SEEDS = protocol.FORMAL_SEEDS
PILOT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3e_target_input_group_ablation" / "pilot_seed42"
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3e_target_input_group_ablation" / "formal_multiseed"
FORCE = False
MIN_RELATIVE_IMPROVEMENT_PCT = 0.5
MIN_SEED_WINS = 4
MIN_STATION_WINS = 15

FULL_STAGE1_KEYS = (
    "self_mean",
    "self_endpoint",
    "self_endpoint_stats",
    "self_endpoint_shifted_stats",
)
FORMAL_STAGE2_KEYS = {
    "pH(无量纲)": (
        "core_self",
        "target5",
        "target5_aux_endpoint",
        "target5_aux_endpoint_stats",
        "target5_aux_endpoint_shifted_stats",
    ),
    "溶解氧(mg/L)": (
        "core_self",
        "target5",
        "target5_aux_endpoint",
        "target5_aux_endpoint_stats",
        "target5_aux_endpoint_shifted_stats",
    ),
    "高锰酸盐指数(mg/L)": (
        "core_self",
        "target5",
        "target5_aux_endpoint",
        "target5_aux_endpoint_stats",
        "target5_aux_endpoint_shifted_stats",
    ),
    "氨氮(mg/L)": (
        "core_self",
        "target5_aux_endpoint",
        "target5_aux_endpoint_stats",
        "target5_aux_endpoint_shifted_stats",
    ),
    "总磷(mg/L)": ("core_self", "target5_aux_endpoint"),
}


def variants_by_key(variants: tuple[ablation.Variant, ...]) -> dict[str, ablation.Variant]:
    return {variant.key: variant for variant in variants}


def validation_gate(
    metrics: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    target: str,
    candidate: str,
    baseline: str,
    control: str | None = None,
) -> dict[str, object]:
    values = metrics[(metrics["split"].eq("val")) & (metrics["target"].eq(target))]
    seed_pivot = values.pivot(index="seed", columns="variant", values="macro_station_rmse")
    common = seed_pivot[[baseline, candidate]].dropna()
    baseline_mean = float(common[baseline].mean())
    candidate_mean = float(common[candidate].mean())
    improvement = 100.0 * (baseline_mean - candidate_mean) / baseline_mean
    seed_wins = int((common[candidate] < common[baseline]).sum())

    station_values = stations[(stations["split"].eq("val")) & (stations["target"].eq(target))]
    station_mean = station_values.groupby(["station", "variant"], as_index=False)["rmse"].mean()
    station_pivot = station_mean.pivot(index="station", columns="variant", values="rmse")
    station_common = station_pivot[[baseline, candidate]].dropna()
    station_wins = int((station_common[candidate] < station_common[baseline]).sum())

    control_improvement = None
    control_seed_wins = None
    control_passed = True
    if control is not None:
        control_common = seed_pivot[[control, candidate]].dropna()
        control_mean = float(control_common[control].mean())
        aligned_mean = float(control_common[candidate].mean())
        control_improvement = 100.0 * (control_mean - aligned_mean) / control_mean
        control_seed_wins = int((control_common[candidate] < control_common[control]).sum())
        control_passed = control_improvement > 0 and control_seed_wins >= MIN_SEED_WINS

    passed = (
        improvement >= MIN_RELATIVE_IMPROVEMENT_PCT
        and seed_wins >= MIN_SEED_WINS
        and station_wins >= MIN_STATION_WINS
        and control_passed
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "relative_improvement_pct": improvement,
        "seed_wins": seed_wins,
        "seed_count": int(len(common)),
        "station_wins": station_wins,
        "station_count": int(len(station_common)),
        "control": control,
        "improvement_vs_control_pct": control_improvement,
        "control_seed_wins": control_seed_wins,
        "passed": bool(passed),
    }


def choose_stage1(
    metrics: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    decisions = {}
    for target in ablation.TARGETS:
        if target not in ablation.HOURLY_TARGETS:
            decisions[target] = {
                "selected_variant": "self_endpoint",
                "reason": "native 4h target; no hourly self-statistic ablation",
                "passed": True,
            }
            continue
        values = metrics[(metrics["split"].eq("val")) & metrics["target"].eq(target)]
        ranking = (
            values[~values["control"].astype(bool)]
            .groupby("variant", as_index=False)["macro_station_rmse"]
            .mean()
            .sort_values("macro_station_rmse")
        )
        selected = "self_endpoint"
        gates = []
        for candidate in ranking["variant"]:
            if candidate == "self_endpoint":
                continue
            control = "self_endpoint_shifted_stats" if candidate == "self_endpoint_stats" else None
            gate = validation_gate(
                metrics,
                stations,
                target=target,
                candidate=str(candidate),
                baseline="self_endpoint",
                control=control,
            )
            gates.append(gate)
            if gate["passed"]:
                selected = str(candidate)
                break
        decisions[target] = {
            "selected_variant": selected,
            "passed": selected != "self_endpoint",
            "gates": gates,
            "selection_split": "val",
            "test_used": False,
        }
    return decisions


def choose_stage2(
    metrics: pd.DataFrame,
    stations: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    decisions = {}
    for target in ablation.TARGETS:
        values = metrics[(metrics["split"].eq("val")) & metrics["target"].eq(target)]
        ranking = (
            values[~values["control"].astype(bool)]
            .groupby("variant", as_index=False)["macro_station_rmse"]
            .mean()
            .sort_values("macro_station_rmse")
        )
        selected = "core_self"
        gates = []
        for candidate in ranking["variant"]:
            if candidate == "core_self":
                continue
            control = (
                "target5_aux_endpoint_shifted_stats"
                if candidate == "target5_aux_endpoint_stats"
                else None
            )
            gate = validation_gate(
                metrics,
                stations,
                target=target,
                candidate=str(candidate),
                baseline="core_self",
                control=control,
            )
            gates.append(gate)
            if gate["passed"]:
                selected = str(candidate)
                break
        decisions[target] = {
            "selected_variant": selected,
            "passed": selected != "core_self",
            "gates": gates,
            "selection_split": "val",
            "test_used": False,
        }
    return decisions


def load_pilot_stage(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(PILOT_DIR / "validation_metrics.csv")
    stations = pd.read_csv(PILOT_DIR / "validation_station_metrics.csv")
    return metrics[metrics["stage"].eq(stage)].copy(), stations[stations["stage"].eq(stage)].copy()


def run_stage1_seed(seed: int, data: pd.DataFrame, stations: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = OUTPUT_DIR / "validation_runs" / f"seed_{seed}" / "stage1"
    metrics_path = seed_dir / "metrics.csv"
    station_path = seed_dir / "station_metrics.csv"
    if metrics_path.exists() and station_path.exists() and not FORCE:
        console.info("reuse stage1", seed=seed)
        return pd.read_csv(metrics_path), pd.read_csv(station_path)
    metric_rows = []
    station_rows = []
    for target in ablation.HOURLY_TARGETS:
        all_variants = ablation.stage1_variants(target)
        subset = tuple(variants_by_key(all_variants)[key] for key in FULL_STAGE1_KEYS)
        rows, station_items, _ = ablation.run_target_variants(
            stage="stage1_self_representation",
            seed=seed,
            target=target,
            variants=subset,
            data=data,
            stations=stations,
            output_dir=seed_dir,
            universe_columns=ablation.universal_columns(all_variants),
        )
        metric_rows.extend(rows)
        station_rows.extend(station_items)
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    station_metrics = pd.DataFrame(station_rows)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    station_metrics.to_csv(station_path, index=False, encoding="utf-8-sig")
    return metrics, station_metrics


def run_stage2_seed(
    seed: int,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    stage1_decisions: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = OUTPUT_DIR / "validation_runs" / f"seed_{seed}" / "stage2"
    metrics_path = seed_dir / "metrics.csv"
    station_path = seed_dir / "station_metrics.csv"
    if metrics_path.exists() and station_path.exists() and not FORCE:
        console.info("reuse stage2", seed=seed)
        return pd.read_csv(metrics_path), pd.read_csv(station_path)
    metric_rows = []
    station_rows = []
    for target in ablation.TARGETS:
        stage1_by_key = variants_by_key(ablation.stage1_variants(target))
        core = stage1_by_key[str(stage1_decisions[target]["selected_variant"])].active_columns
        all_variants = ablation.stage2_variants(target, core)
        by_key = variants_by_key(all_variants)
        subset = tuple(by_key[key] for key in FORMAL_STAGE2_KEYS[target])
        rows, station_items, _ = ablation.run_target_variants(
            stage="stage2_feature_groups",
            seed=seed,
            target=target,
            variants=subset,
            data=data,
            stations=stations,
            output_dir=seed_dir,
            universe_columns=ablation.universal_columns(all_variants),
        )
        metric_rows.extend(rows)
        station_rows.extend(station_items)
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    station_metrics = pd.DataFrame(station_rows)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    station_metrics.to_csv(station_path, index=False, encoding="utf-8-sig")
    return metrics, station_metrics


def run_selected_test_seed(
    seed: int,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    stage1_decisions: dict[str, dict[str, object]],
    stage2_decisions: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = OUTPUT_DIR / "sealed_test_runs" / f"seed_{seed}"
    metrics_path = seed_dir / "metrics.csv"
    station_path = seed_dir / "station_metrics.csv"
    if metrics_path.exists() and station_path.exists() and not FORCE:
        console.info("reuse sealed test", seed=seed)
        return pd.read_csv(metrics_path), pd.read_csv(station_path)
    metric_rows = []
    station_rows = []
    for target in ablation.TARGETS:
        stage1_by_key = variants_by_key(ablation.stage1_variants(target))
        core = stage1_by_key[str(stage1_decisions[target]["selected_variant"])].active_columns
        all_variants = ablation.stage2_variants(target, core)
        selected = variants_by_key(all_variants)[str(stage2_decisions[target]["selected_variant"])]
        rows, station_items, _ = ablation.run_target_variants(
            stage="sealed_test",
            seed=seed,
            target=target,
            variants=(selected,),
            data=data,
            stations=stations,
            output_dir=seed_dir,
            evaluation_splits=("train", "val", "test"),
            universe_columns=ablation.universal_columns(all_variants),
        )
        metric_rows.extend(rows)
        station_rows.extend(station_items)
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    station_metrics = pd.DataFrame(station_rows)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    station_metrics.to_csv(station_path, index=False, encoding="utf-8-sig")
    return metrics, station_metrics


def audit_validation_reproduction(
    stage2_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    stage2_decisions: dict[str, dict[str, object]],
) -> dict[str, object]:
    comparisons = []
    for target, decision in stage2_decisions.items():
        variant = str(decision["selected_variant"])
        locked = stage2_metrics[
            stage2_metrics["split"].eq("val")
            & stage2_metrics["target"].eq(target)
            & stage2_metrics["variant"].eq(variant)
        ][["seed", "macro_station_rmse", "rmse"]]
        retrained = test_metrics[
            test_metrics["split"].eq("val")
            & test_metrics["target"].eq(target)
            & test_metrics["variant"].eq(variant)
        ][["seed", "macro_station_rmse", "rmse"]]
        merged = locked.merge(retrained, on="seed", suffixes=("_locked", "_retrained"))
        if len(merged) != len(SEEDS):
            raise RuntimeError(f"Incomplete validation reproduction for {target}: {len(merged)}/{len(SEEDS)}")
        for row in merged.itertuples(index=False):
            comparisons.append(
                {
                    "target": target,
                    "variant": variant,
                    "seed": int(row.seed),
                    "macro_station_rmse_abs_diff": abs(
                        float(row.macro_station_rmse_locked) - float(row.macro_station_rmse_retrained)
                    ),
                    "rmse_abs_diff": abs(float(row.rmse_locked) - float(row.rmse_retrained)),
                }
            )
    maximum = max(
        max(item["macro_station_rmse_abs_diff"], item["rmse_abs_diff"])
        for item in comparisons
    )
    if maximum > 1e-10:
        raise RuntimeError(f"Locked validation result changed during test retraining: {maximum}")
    return {
        "passed": True,
        "tolerance": 1e-10,
        "max_absolute_difference": maximum,
        "comparisons": comparisons,
    }


def write_report(
    stage1_metrics: pd.DataFrame,
    stage2_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    stage1_decisions: dict,
    stage2_decisions: dict,
) -> None:
    def summary(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        return (
            frame[frame["split"].eq(split)]
            .groupby(["target", "variant"], as_index=False)
            .agg(
                macro_rmse_mean=("macro_station_rmse", "mean"),
                macro_rmse_std=("macro_station_rmse", "std"),
                rmse_mean=("rmse", "mean"),
                nse_mean=("nse", "mean"),
                sign_accuracy_mean=("delta_sign_accuracy", "mean"),
                tail_delta_rmse_mean=("tail_delta_rmse", "mean"),
            )
            .sort_values(["target", "macro_rmse_mean"])
        )

    stage1_summary = summary(stage1_metrics, "val")
    stage2_summary = summary(stage2_metrics, "val")
    test_summary = summary(test_metrics, "test")
    lines = [
        "# Per-target input group ablation: five-seed formal result",
        "",
        f"- Seeds: {list(SEEDS)}.",
        "- Input 24h, output next 4h; D-GRU and L1 loss fixed.",
        "- Stage 1 and Stage 2 selection use validation only.",
        "- Test is evaluated only for the locked per-target Stage 2 variant.",
        "- Promotion gate: >=0.5% mean improvement, >=4/5 seed wins, >=15/25 station wins; aligned statistics must also beat shifted statistics in >=4/5 seeds.",
        "",
        "## Stage 1 validation",
        "```text",
        stage1_summary.to_string(index=False),
        "```",
        "",
        "## Stage 1 decisions",
        "```json",
        json.dumps(stage1_decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Stage 2 validation",
        "```text",
        stage2_summary.to_string(index=False),
        "```",
        "",
        "## Stage 2 decisions",
        "```json",
        json.dumps(stage2_decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Locked test",
        "```text",
        test_summary.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "formal_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_formal() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper.SEED = protocol.PILOT_SEED
    window.OUTPUT_STEPS = ablation.OUTPUT_STEPS
    dual.INPUT_STEPS = ablation.INPUT_STEPS
    dual.OUTPUT_STEPS = ablation.OUTPUT_STEPS
    data = hourly.load_ablation_data()
    stations = tuple(sorted(data["station"].astype(str).unique()))
    console.phase("per-target input ablation | five-seed formal")
    console.info("protocol", seeds=list(SEEDS), stations=len(stations), test="sealed until validation decision")

    pilot_stage1_metrics, pilot_stage1_stations = load_pilot_stage("stage1_self_representation")
    stage1_parts = [pilot_stage1_metrics[pilot_stage1_metrics["target"].isin(ablation.HOURLY_TARGETS)]]
    stage1_station_parts = [pilot_stage1_stations[pilot_stage1_stations["target"].isin(ablation.HOURLY_TARGETS)]]
    for seed in SEEDS:
        if seed == protocol.PILOT_SEED:
            continue
        metrics, station_metrics = run_stage1_seed(seed, data, stations)
        stage1_parts.append(metrics)
        stage1_station_parts.append(station_metrics)
    stage1_metrics = pd.concat(stage1_parts, ignore_index=True)
    stage1_station_metrics = pd.concat(stage1_station_parts, ignore_index=True)
    stage1_decisions = choose_stage1(stage1_metrics, stage1_station_metrics)
    stage1_metrics.to_csv(OUTPUT_DIR / "stage1_validation_metrics.csv", index=False, encoding="utf-8-sig")
    stage1_station_metrics.to_csv(OUTPUT_DIR / "stage1_validation_station_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "stage1_decision.json").write_text(json.dumps(stage1_decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    pilot_selection = json.loads((PILOT_DIR / "validation_selection.json").read_text(encoding="utf-8"))["stage1"]
    can_reuse_pilot_stage2 = all(
        str(stage1_decisions[target]["selected_variant"]) == str(pilot_selection[target]["variant"])
        for target in ablation.TARGETS
    )
    stage2_parts = []
    stage2_station_parts = []
    if can_reuse_pilot_stage2:
        pilot_metrics, pilot_stations = load_pilot_stage("stage2_feature_groups")
        keep = pilot_metrics.apply(lambda row: row["variant"] in FORMAL_STAGE2_KEYS[row["target"]] or row["variant"] == "persistence", axis=1)
        station_keep = pilot_stations.apply(lambda row: row["variant"] in FORMAL_STAGE2_KEYS[row["target"]] or row["variant"] == "persistence", axis=1)
        stage2_parts.append(pilot_metrics[keep])
        stage2_station_parts.append(pilot_stations[station_keep])
    for seed in SEEDS:
        if seed == protocol.PILOT_SEED and can_reuse_pilot_stage2:
            continue
        metrics, station_metrics = run_stage2_seed(seed, data, stations, stage1_decisions)
        stage2_parts.append(metrics)
        stage2_station_parts.append(station_metrics)
    stage2_metrics = pd.concat(stage2_parts, ignore_index=True)
    stage2_station_metrics = pd.concat(stage2_station_parts, ignore_index=True)
    stage2_decisions = choose_stage2(stage2_metrics, stage2_station_metrics)
    stage2_metrics.to_csv(OUTPUT_DIR / "stage2_validation_metrics.csv", index=False, encoding="utf-8-sig")
    stage2_station_metrics.to_csv(OUTPUT_DIR / "stage2_validation_station_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "stage2_decision.json").write_text(json.dumps(stage2_decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    console.phase("validation locked; opening selected test runs")
    test_parts = []
    test_station_parts = []
    for seed in SEEDS:
        metrics, station_metrics = run_selected_test_seed(seed, data, stations, stage1_decisions, stage2_decisions)
        test_parts.append(metrics)
        test_station_parts.append(station_metrics)
    test_metrics = pd.concat(test_parts, ignore_index=True)
    test_station_metrics = pd.concat(test_station_parts, ignore_index=True)
    test_metrics.to_csv(OUTPUT_DIR / "locked_test_metrics.csv", index=False, encoding="utf-8-sig")
    test_station_metrics.to_csv(OUTPUT_DIR / "locked_test_station_metrics.csv", index=False, encoding="utf-8-sig")
    reproduction_audit = audit_validation_reproduction(stage2_metrics, test_metrics, stage2_decisions)
    (OUTPUT_DIR / "validation_reproduction_audit.json").write_text(
        json.dumps(reproduction_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(stage1_metrics, stage2_metrics, test_metrics, stage1_decisions, stage2_decisions)

    manifest = protocol.build_run_manifest(
        experiment="stage3e_per_target_input_group_formal",
        output_dir=OUTPUT_DIR,
        seed=protocol.PILOT_SEED,
        observed_path=hourly.VALUES_PATH,
        quality_path=hourly.QUALITY_PATH,
        code_paths=(
            Path("scripts/gru/target_input_group_ablation.py"),
            Path("scripts/gru/target_input_group_multiseed.py"),
            Path("scripts/gru/run_v2_target_input_group_multiseed.py"),
        ),
    )
    manifest.update(
        {
            "formal_seeds": list(SEEDS),
            "promotion_gate": {
                "min_relative_improvement_pct": MIN_RELATIVE_IMPROVEMENT_PCT,
                "min_seed_wins": MIN_SEED_WINS,
                "min_station_wins": MIN_STATION_WINS,
                "aligned_stats_must_beat_shifted": True,
            },
            "stage1_decisions": stage1_decisions,
            "stage2_decisions": stage2_decisions,
            "test_opened_after_validation_lock": True,
            "validation_reproduction_audit": {
                "passed": reproduction_audit["passed"],
                "max_absolute_difference": reproduction_audit["max_absolute_difference"],
            },
        }
    )
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    console.done(OUTPUT_DIR, report="formal_report.md")
    return 0
