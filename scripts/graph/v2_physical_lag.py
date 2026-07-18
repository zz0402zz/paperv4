#!/usr/bin/env python3
"""Physical travel-time audit for V2 direct-pair graph experiments."""

from __future__ import annotations

from scripts.common.terminal_output import console

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.graph import v2_direct_pair_graph_config as cfg

EDGE_AUDIT_PATH = Path("outputs/maps/station_upstream_relation_audit_2026-06-24.csv")
VELOCITY_WORKBOOK_PATH = Path("data/流量数据-水利厅/实测流量表xls.xls")
VELOCITY_COLUMN_CANDIDATES = ("断面平均流速", "平均流速", "流速", "velocity_mps")
STATION_COLUMN_CANDIDATES = ("站名", "station", "测站名称")


def travel_hours(distance_km: float, velocity_mps: float) -> float:
    """Convert straight-line distance and velocity into travel time hours."""
    if distance_km <= 0 or velocity_mps <= 0:
        raise ValueError("distance and velocity must be positive")
    return float(distance_km) * 1000.0 / float(velocity_mps) / 3600.0


def travel_steps(distance_km: float, velocity_mps: float, step_hours: int = 4) -> int:
    """Round travel time to a causal 4-hour step count with minimum one step."""
    ratio = travel_hours(distance_km, velocity_mps) / float(step_hours)
    return max(1, int(np.floor(ratio + 0.5)))


def velocity_quantiles(values: pd.Series) -> dict[str, float | int | None]:
    """Return positive-velocity P25/median/P75 values."""
    numeric = pd.to_numeric(values, errors="coerce")
    positive = numeric[np.isfinite(numeric) & (numeric > 0)]
    if positive.empty:
        return {"count": 0, "p25": None, "median": None, "p75": None}
    return {
        "count": int(positive.size),
        "p25": float(positive.quantile(0.25)),
        "median": float(positive.quantile(0.50)),
        "p75": float(positive.quantile(0.75)),
    }


def _normalize_name(value: object) -> str:
    return str(value).strip().replace("（", "(").replace("）", ")")


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    return None


def load_velocity_observations(path: Path = VELOCITY_WORKBOOK_PATH) -> pd.DataFrame:
    """Read measured velocity observations from the water-resources workbook."""
    if not path.exists():
        return pd.DataFrame(columns=["velocity_station", "velocity_mps"])
    frames = []
    workbook = pd.ExcelFile(path)
    for sheet in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.empty:
            continue
        header_idx = None
        for idx, row in raw.head(20).iterrows():
            values = {_normalize_name(value) for value in row.dropna().tolist()}
            if values & set(VELOCITY_COLUMN_CANDIDATES):
                header_idx = idx
                break
        if header_idx is None:
            continue
        frame = pd.read_excel(path, sheet_name=sheet, header=header_idx)
        station_col = _first_present(frame.columns, STATION_COLUMN_CANDIDATES)
        velocity_col = _first_present(frame.columns, VELOCITY_COLUMN_CANDIDATES)
        if station_col is None or velocity_col is None:
            continue
        parsed = pd.DataFrame(
            {
                "velocity_station": frame[station_col].map(_normalize_name),
                "velocity_mps": pd.to_numeric(frame[velocity_col], errors="coerce"),
            }
        )
        parsed = parsed[parsed["velocity_station"].ne("") & parsed["velocity_mps"].gt(0)]
        frames.append(parsed)
    if not frames:
        return pd.DataFrame(columns=["velocity_station", "velocity_mps"])
    return pd.concat(frames, ignore_index=True)


def load_selected_edge_distances(path: Path = EDGE_AUDIT_PATH) -> pd.DataFrame:
    """Load the three configured direct-pair distances from the audited edge table."""
    audit = pd.read_csv(path)
    rows = []
    for source, target in cfg.PAIRS:
        match = audit[
            audit["source_station"].astype(str).eq(source)
            & audit["target_station"].astype(str).eq(target)
        ]
        if match.empty:
            raise ValueError(f"Missing audited edge distance for {source} -> {target}")
        row = match.iloc[0]
        rows.append(
            {
                "source_station": source,
                "target_station": target,
                "relation": row.get("draft_relation", row.get("relation", "")),
                "distance_km": float(row["straight_distance_km"]),
                "distance_kind": "straight_distance_km",
                "verified_confidence": row.get("verified_confidence", ""),
            }
        )
    return pd.DataFrame(rows)


def _pair_velocity(values: pd.DataFrame, source: str, target: str) -> tuple[str, str, pd.Series]:
    if values.empty:
        return "", "missing", pd.Series(dtype=float)
    if "velocity_station" not in values.columns:
        return "regional_all_positive_velocity", "regional_reference", values["velocity_mps"]
    stations = values["velocity_station"].astype(str)
    source_hits = values.loc[stations.eq(source), "velocity_mps"]
    target_hits = values.loc[stations.eq(target), "velocity_mps"]
    if len(source_hits) >= 10:
        return source, "source_station", source_hits
    if len(target_hits) >= 10:
        return target, "target_station", target_hits
    return "regional_all_positive_velocity", "regional_reference", values["velocity_mps"]


def build_lag_audit(edge_distances: pd.DataFrame, velocity_observations: pd.DataFrame) -> pd.DataFrame:
    """Build a physical-lag audit table for configured station pairs."""
    rows = []
    for row in edge_distances.itertuples(index=False):
        source = str(row.source_station)
        target = str(row.target_station)
        velocity_station, velocity_scope, velocity_values = _pair_velocity(velocity_observations, source, target)
        quantiles = velocity_quantiles(velocity_values)
        if quantiles["count"] == 0:
            raise ValueError("No positive velocity observations are available for lag estimation.")
        distance_km = float(row.distance_km)
        p25 = float(quantiles["p25"])
        median = float(quantiles["median"])
        p75 = float(quantiles["p75"])
        rows.append(
            {
                "source_station": source,
                "target_station": target,
                "relation": getattr(row, "relation", ""),
                "distance_km": distance_km,
                "distance_kind": getattr(row, "distance_kind", "straight_distance_km"),
                "velocity_station": velocity_station,
                "velocity_scope": velocity_scope,
                "velocity_observation_count": int(quantiles["count"]),
                "velocity_p25_mps": p25,
                "velocity_median_mps": median,
                "velocity_p75_mps": p75,
                "lag_fast_steps": travel_steps(distance_km, p75),
                "lag_primary_steps": travel_steps(distance_km, median),
                "lag_slow_steps": travel_steps(distance_km, p25),
                "lag_primary_hours": travel_steps(distance_km, median) * 4,
                "evidence_note": (
                    "regional positive measured velocities; not reach-specific"
                    if velocity_scope == "regional_reference"
                    else "station-specific measured velocities"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_lag_audit(audit: pd.DataFrame) -> None:
    """Fail before training if audit rows are incomplete or ambiguous."""
    required = {"source_station", "target_station", "distance_km", "velocity_scope", "lag_primary_steps"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Missing lag audit columns: {sorted(missing)}")
    if audit["distance_km"].isna().any() or (audit["distance_km"] <= 0).any():
        raise ValueError("Every selected pair must have a positive audited distance.")
    if audit["velocity_scope"].isna().any() or audit["velocity_scope"].astype(str).eq("").any():
        raise ValueError("Every selected pair must declare measured or regional velocity scope.")
    if (audit["velocity_observation_count"] < 10).any():
        raise ValueError("Every selected pair needs at least 10 velocity observations or a regional reference.")


def write_lag_audit(output_path: Path = cfg.LAG_AUDIT_PATH) -> pd.DataFrame:
    """Generate and save the physical-lag audit CSV."""
    cfg.ensure_output_dirs()
    edges = load_selected_edge_distances()
    velocities = load_velocity_observations()
    audit = build_lag_audit(edges, velocities)
    validate_lag_audit(audit)
    audit.to_csv(output_path, index=False, encoding="utf-8-sig")
    return audit


def main() -> int:
    audit = write_lag_audit()
    console.print(audit.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
