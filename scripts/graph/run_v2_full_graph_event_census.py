#!/usr/bin/env python3
"""Run a causal event census over every V2 strict graph edge."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import hydromet_features
from scripts.graph import run_v2_delayed_step_graph_ablation as stage4b
from scripts.graph import v2_delayed_step_graph as delayed_graph
from scripts.common import v2_experiment_protocol as protocol
from scripts.graph import v2_graph_event_census as census
from scripts.graph import v2_physical_lag

OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage5_event_census"
FEATURES = protocol.TARGET_FEATURE_COLUMNS
OUTPUT_STEPS = 9
TRAIN_EVENT_MIN = 50
VAL_EVENT_MIN = 20


def load_strict_edges(path: Path = protocol.STRICT_EDGES_PATH) -> pd.DataFrame:
    edges = pd.read_csv(path, encoding="utf-8-sig")
    required = {"source_station", "target_station"}
    if not required.issubset(edges.columns):
        raise ValueError("Strict edge file is missing station columns")
    if edges.duplicated(["source_station", "target_station"]).any():
        raise ValueError("Strict edge file contains duplicates")
    return edges[["source_station", "target_station"]].copy()


def rank_candidates(feature_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate evidence from validation only."""
    validation = feature_summary[feature_summary["split"].astype(str).eq("val")].copy()
    rows = []
    for keys, group in validation.groupby(["source_station", "target_station"], sort=False):
        count = pd.to_numeric(group["event_count"], errors="coerce").fillna(0)
        uplift = pd.to_numeric(group["response_uplift"], errors="coerce")
        weights = count.where(uplift.notna(), 0.0)
        weighted_uplift = (
            float((uplift.fillna(0.0) * weights).sum() / weights.sum())
            if float(weights.sum()) > 0
            else float("nan")
        )
        flow_uplift = pd.to_numeric(
            group["flow_response_uplift"]
            if "flow_response_uplift" in group
            else pd.Series(np.nan, index=group.index),
            errors="coerce",
        )
        flow_count = pd.to_numeric(
            group["flow_supported_event_count"]
            if "flow_supported_event_count" in group
            else pd.Series(0, index=group.index),
            errors="coerce",
        ).fillna(0)
        flow_weights = flow_count.where(flow_uplift.notna(), 0.0)
        weighted_flow_uplift = (
            float((flow_uplift.fillna(0.0) * flow_weights).sum() / flow_weights.sum())
            if float(flow_weights.sum()) > 0
            else float("nan")
        )
        if "response_qvalue" in group:
            significant = (
                (uplift > 0.0)
                & (pd.to_numeric(group["response_qvalue"], errors="coerce") < 0.10)
                & (pd.to_numeric(group["event_evaluable_count"], errors="coerce") >= 10)
            )
        else:
            significant = uplift > 0.0
        rows.append(
            {
                "source_station": keys[0],
                "target_station": keys[1],
                "validation_event_count": int(count.sum()),
                "features_with_events": int((count > 0).sum()),
                "features_positive_uplift": int((uplift > 0).sum()),
                "features_significant_positive": int(significant.sum()),
                "validation_weighted_response_uplift": weighted_uplift,
                "validation_flow_event_count": int(flow_count.sum()),
                "validation_weighted_flow_response_uplift": weighted_flow_uplift,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_weighted_response_uplift", "validation_event_count"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_lag_audit(edges: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(v2_physical_lag.EDGE_AUDIT_PATH)
    rows = []
    for edge in edges.itertuples(index=False):
        match = source[
            source["source_station"].astype(str).eq(str(edge.source_station))
            & source["target_station"].astype(str).eq(str(edge.target_station))
        ]
        if len(match) != 1:
            raise ValueError(f"Missing audited edge distance for {edge.source_station} -> {edge.target_station}")
        row = match.iloc[0]
        rows.append(
            {
                "source_station": str(edge.source_station),
                "target_station": str(edge.target_station),
                "relation": row.get("draft_relation", ""),
                "distance_km": float(row["straight_distance_km"]),
                "distance_kind": "straight_distance_km",
            }
        )
    audit = v2_physical_lag.build_lag_audit(
        pd.DataFrame(rows), v2_physical_lag.load_velocity_observations()
    )
    v2_physical_lag.validate_lag_audit(audit)
    audit["lag_support_steps"] = audit["lag_primary_steps"].map(
        lambda value: "|".join(
            str(step)
            for step in delayed_graph.lag_support(int(value))
            if step <= OUTPUT_STEPS
        )
    )
    return audit


def build_edge_frame(data: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    source_frame = stage4b._station_frame(data, source, "source")
    target_frame = stage4b._station_frame(data, target, "target")
    merged = target_frame.merge(source_frame, on="time", how="inner", validate="one_to_one")
    for horizon in range(1, OUTPUT_STEPS + 1):
        future = target_frame[["time", *(f"{feature}_diff1_target" for feature in FEATURES)]].copy()
        future["time"] = future["time"] - pd.Timedelta(hours=4 * horizon)
        future = future.rename(
            columns={
                f"{feature}_diff1_target": f"{feature}_future_step_{horizon}"
                for feature in FEATURES
            }
        )
        merged = merged.merge(future, on="time", how="left", validate="one_to_one")
    return merged.sort_values("time").reset_index(drop=True)


def split_labels(times) -> np.ndarray:
    values = pd.DatetimeIndex(pd.to_datetime(times))
    return np.where(
        values < pd.Timestamp(protocol.TRAIN_END),
        "train",
        np.where(values < pd.Timestamp(protocol.VAL_END), "val", "test"),
    )


def fit_flow_thresholds(
    flow_previous: np.ndarray,
    flow_rise: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    train = labels == "train"
    previous = flow_previous[train & np.isfinite(flow_previous)]
    rise = flow_rise[train & np.isfinite(flow_rise)]
    if len(previous) < 20 or len(rise) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(previous, 0.90)), float(np.quantile(rise, 0.90))


def safe_rate(values: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    finite = selected[np.isfinite(selected)]
    return int(len(finite)), float(np.mean(finite)) if len(finite) else float("nan")


def add_significance(feature_summary: pd.DataFrame) -> pd.DataFrame:
    output = feature_summary.copy()
    output["response_pvalue"] = np.nan
    output["response_qvalue"] = np.nan
    output["flow_response_pvalue"] = np.nan
    output["flow_response_qvalue"] = np.nan
    validation = output["split"].eq("val")
    for idx in output.index[validation]:
        row = output.loc[idx]
        output.loc[idx, "response_pvalue"] = census.positive_uplift_pvalue(
            int(row["event_response_success_count"]),
            int(row["event_evaluable_count"]),
            int(row["control_response_success_count"]),
            int(row["control_evaluable_count"]),
        )
        output.loc[idx, "flow_response_pvalue"] = census.positive_uplift_pvalue(
            int(row["flow_event_response_success_count"]),
            int(row["flow_event_evaluable_count"]),
            int(row["flow_control_response_success_count"]),
            int(row["flow_control_evaluable_count"]),
        )
    output.loc[validation, "response_qvalue"] = census.benjamini_hochberg(
        output.loc[validation, "response_pvalue"].to_numpy(float)
    )
    output.loc[validation, "flow_response_qvalue"] = census.benjamini_hochberg(
        output.loc[validation, "flow_response_pvalue"].to_numpy(float)
    )
    output["significant_positive_uplift"] = (
        output["split"].eq("val")
        & (output["response_uplift"] > 0.0)
        & (output["response_qvalue"] < 0.10)
        & (output["event_evaluable_count"] >= 10)
    )
    output["significant_positive_flow_uplift"] = (
        output["split"].eq("val")
        & (output["flow_response_uplift"] > 0.0)
        & (output["flow_response_qvalue"] < 0.10)
        & (output["flow_event_evaluable_count"] >= 10)
    )
    return output


def analyze_edge_feature(
    frame: pd.DataFrame,
    source: str,
    target: str,
    feature: str,
    support: tuple[int, ...],
    flow_previous: np.ndarray,
    flow_rise: np.ndarray,
    flow_high_threshold: float,
    flow_rise_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_delta = pd.to_numeric(frame[f"{feature}_diff1_source"], errors="coerce").to_numpy(float)
    target_delta = pd.to_numeric(frame[f"{feature}_diff1_target"], errors="coerce").to_numpy(float)
    labels = split_labels(frame["time"])
    train_idx = np.flatnonzero(labels == "train")
    thresholds = census.fit_event_thresholds(source_delta, target_delta, train_idx)
    event = census.event_flags(source_delta, target_delta, thresholds)
    control = census.control_flags(source_delta, target_delta, thresholds)
    event = census.thin_event_origins(event, frame["time"], labels, min_gap_hours=24)
    control = census.thin_event_origins(control, frame["time"], labels, min_gap_hours=24)
    flow_event = (
        np.isfinite(flow_previous)
        & np.isfinite(flow_rise)
        & (
            (flow_previous > flow_high_threshold)
            | (flow_rise > flow_rise_threshold)
        )
    )

    future = frame[[f"{feature}_future_step_{step}" for step in range(1, OUTPUT_STEPS + 1)]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if support:
        response, first_step, max_signed = census.delayed_response(
            future, source_delta, thresholds.target_response_abs, support
        )
        evaluable = np.isfinite(future[:, np.asarray(support) - 1]).all(axis=1)
        response_numeric = response.astype(float)
        response_numeric[~evaluable] = np.nan
    else:
        response_numeric = np.full(len(frame), np.nan)
        first_step = np.full(len(frame), np.nan)
        max_signed = np.full(len(frame), np.nan)

    summaries = []
    records = []
    for split_name in ("train", "val", "test"):
        split_mask = labels == split_name
        split_response, split_first, split_max = census.blind_test_response(
            response_numeric.copy(), first_step.copy(), max_signed.copy(), split_name
        )
        event_mask = split_mask & event
        control_mask = split_mask & control
        flow_event_mask = event_mask & flow_event
        flow_control_mask = control_mask & flow_event
        event_eval_count, event_rate = safe_rate(split_response, event_mask)
        control_eval_count, control_rate = safe_rate(split_response, control_mask)
        flow_event_eval_count, flow_event_rate = safe_rate(split_response, flow_event_mask)
        flow_control_eval_count, flow_control_rate = safe_rate(split_response, flow_control_mask)
        event_success = int(np.nansum(split_response[event_mask]))
        control_success = int(np.nansum(split_response[control_mask]))
        flow_event_success = int(np.nansum(split_response[flow_event_mask]))
        flow_control_success = int(np.nansum(split_response[flow_control_mask]))
        summaries.append(
            {
                "source_station": source,
                "target_station": target,
                "feature": feature,
                "split": split_name,
                "eligible_origin_count": int((split_mask & np.isfinite(source_delta) & np.isfinite(target_delta)).sum()),
                "event_count": int(event_mask.sum()),
                "event_evaluable_count": event_eval_count,
                "event_response_rate": event_rate,
                "event_response_success_count": event_success,
                "control_count": int(control_mask.sum()),
                "control_evaluable_count": control_eval_count,
                "control_response_rate": control_rate,
                "control_response_success_count": control_success,
                "response_uplift": event_rate - control_rate if np.isfinite(event_rate) and np.isfinite(control_rate) else np.nan,
                "flow_supported_event_count": int(flow_event_mask.sum()),
                "flow_event_evaluable_count": flow_event_eval_count,
                "flow_event_response_rate": flow_event_rate,
                "flow_event_response_success_count": flow_event_success,
                "flow_control_count": int(flow_control_mask.sum()),
                "flow_control_evaluable_count": flow_control_eval_count,
                "flow_control_response_rate": flow_control_rate,
                "flow_control_response_success_count": flow_control_success,
                "flow_response_uplift": flow_event_rate - flow_control_rate if np.isfinite(flow_event_rate) and np.isfinite(flow_control_rate) else np.nan,
                "source_shock_threshold_abs": thresholds.source_shock_abs,
                "source_control_threshold_abs": thresholds.source_control_abs,
                "target_quiet_threshold_abs": thresholds.target_quiet_abs,
                "target_response_threshold_abs": thresholds.target_response_abs,
                "flow_high_threshold": flow_high_threshold,
                "flow_rise_threshold": flow_rise_threshold,
                "lag_support_steps": "|".join(map(str, support)),
            }
        )
        for idx in np.flatnonzero(event_mask):
            records.append(
                {
                    "source_station": source,
                    "target_station": target,
                    "feature": feature,
                    "origin_time": frame.loc[idx, "time"],
                    "split": split_name,
                    "source_delta": source_delta[idx],
                    "target_current_delta": target_delta[idx],
                    "flow_previous_day": flow_previous[idx],
                    "flow_log_rise": flow_rise[idx],
                    "high_or_rising_flow": bool(flow_event[idx]),
                    "physical_lag_support_steps": "|".join(map(str, support)),
                    "response_observed": split_response[idx],
                    "response_first_step": split_first[idx],
                    "response_max_signed_delta": split_max[idx],
                }
            )
    return summaries, records


def add_candidate_tiers(ranking: pd.DataFrame, feature_summary: pd.DataFrame) -> pd.DataFrame:
    train = feature_summary[feature_summary["split"].eq("train")].groupby(
        ["source_station", "target_station"], as_index=False
    )["event_count"].sum().rename(columns={"event_count": "train_event_count"})
    test = feature_summary[feature_summary["split"].eq("test")].groupby(
        ["source_station", "target_station"], as_index=False
    )[["event_count", "flow_supported_event_count"]].sum().rename(
        columns={
            "event_count": "blinded_test_event_count",
            "flow_supported_event_count": "blinded_test_flow_event_count",
        }
    )
    output = ranking.merge(train, on=["source_station", "target_station"], how="left")
    output = output.merge(test, on=["source_station", "target_station"], how="left")
    strong = (
        (output["train_event_count"] >= TRAIN_EVENT_MIN)
        & (output["validation_event_count"] >= VAL_EVENT_MIN)
        & (output["features_significant_positive"] >= 2)
        & (output["validation_weighted_response_uplift"] > 0.05)
    )
    exploratory = (
        (output["train_event_count"] >= 30)
        & (output["validation_event_count"] >= 10)
        & (output["features_significant_positive"] >= 1)
        & (output["validation_weighted_response_uplift"] > 0.0)
    )
    output["candidate_tier"] = np.where(strong, "primary", np.where(exploratory, "exploratory", "insufficient"))
    return output


def write_report(
    ranking: pd.DataFrame,
    feature_summary: pd.DataFrame,
    lag_audit: pd.DataFrame,
    output_path: Path,
) -> None:
    validation = feature_summary[feature_summary["split"].eq("val")]
    lines = [
        "# V2 Full-Graph Event Census",
        "",
        "- Event: upstream absolute one-step change > train Q95 while downstream current change <= train Q50.",
        "- Control: upstream train Q50-Q95 moderate change while downstream remains quiet.",
        "- Response: same-direction downstream one-step change >= train Q75 inside the physical lag support.",
        "- Flow state: previous complete day exceeds train Q90 or log-flow rise exceeds train Q90.",
        "- Consecutive eligible origins within 24h are collapsed into one event episode.",
        "- Positive response tests use one-sided two-proportion tests with Benjamini-Hochberg FDR control across 95 validation comparisons.",
        "- Candidate ranking uses 2024 validation only; 2025 response outcomes remain blind.",
        "",
        "## Candidate Ranking",
        "",
        "```text",
        ranking.to_string(index=False),
        "```",
        "",
        "## Validation Feature Results",
        "",
        "```text",
        validation.sort_values(["response_uplift", "event_count"], ascending=[False, False]).to_string(index=False),
        "```",
        "",
        "## Physical Lag Coverage",
        "",
        "```text",
        lag_audit[["source_station", "target_station", "lag_primary_steps", "lag_support_steps", "velocity_scope"]].to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_census(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = load_strict_edges()
    lag_audit = build_lag_audit(edges)
    data = stage4b.load_v2_data()
    daily_flow = hydromet_features.read_daily_flow()
    summaries = []
    records = []
    flow_rows = []
    for edge in edges.itertuples(index=False):
        source = str(edge.source_station)
        target = str(edge.target_station)
        lag_row = lag_audit[
            lag_audit["source_station"].eq(source) & lag_audit["target_station"].eq(target)
        ].iloc[0]
        support = tuple(
            int(value)
            for value in str(lag_row["lag_support_steps"]).split("|")
            if str(value)
        )
        frame = build_edge_frame(data, source, target)
        labels = split_labels(frame["time"])
        flow_code = str(hydromet_features.STATION_HYDROMET_MAP[target]["flow_station_code"])
        if flow_code not in daily_flow.columns:
            flow_previous = np.full(len(frame), np.nan)
            flow_rise = np.full(len(frame), np.nan)
        else:
            flow_features = census.causal_flow_features(daily_flow[flow_code], frame["time"])
            flow_previous = flow_features["flow_previous_day"]
            flow_rise = flow_features["flow_log_rise"]
        high_threshold, rise_threshold = fit_flow_thresholds(flow_previous, flow_rise, labels)
        flow_rows.append(
            {
                "source_station": source,
                "target_station": target,
                "target_flow_station_code": flow_code,
                "flow_valid_count": int((np.isfinite(flow_previous) & np.isfinite(flow_rise)).sum()),
                "flow_high_threshold": high_threshold,
                "flow_rise_threshold": rise_threshold,
            }
        )
        console.print(f"census {source}->{target} support={support}", flush=True)
        for feature in FEATURES:
            feature_summary, feature_records = analyze_edge_feature(
                frame,
                source,
                target,
                feature,
                support,
                flow_previous,
                flow_rise,
                high_threshold,
                rise_threshold,
            )
            summaries.extend(feature_summary)
            records.extend(feature_records)

    feature_summary = add_significance(pd.DataFrame(summaries))
    event_records = pd.DataFrame(records)
    ranking = add_candidate_tiers(rank_candidates(feature_summary), feature_summary)
    lag_audit.to_csv(output_dir / "physical_lag_audit.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(output_dir / "edge_feature_event_summary.csv", index=False, encoding="utf-8-sig")
    event_records.to_csv(output_dir / "event_records.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "candidate_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow_rows).to_csv(output_dir / "flow_regime_summary.csv", index=False, encoding="utf-8-sig")
    manifest = protocol.build_run_manifest(
        experiment="stage5_full_graph_event_census",
        output_dir=output_dir,
        code_paths=(
            Path("scripts/graph/v2_graph_event_census.py"),
            Path("scripts/graph/run_v2_full_graph_event_census.py"),
        ),
    )
    manifest.update(
        {
            "strict_edge_count": int(len(edges)),
            "features": list(FEATURES),
            "threshold_split": "train_2022_2023_only",
            "candidate_split": "validation_2024_only",
            "test_response_blinded": True,
            "event_definition": "source_abs_diff_gt_train_q95_and_target_abs_diff_le_train_q50",
            "response_definition": "same_direction_target_step_abs_ge_train_q75_in_physical_lag_support",
            "event_episode_min_gap_hours": 24,
            "multiple_testing": "one_sided_two_proportion_z_test_bh_fdr_0.10",
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(ranking, feature_summary, lag_audit, output_dir / "event_census_report.md")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run_census(args.output_dir)
    console.print(f"saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
