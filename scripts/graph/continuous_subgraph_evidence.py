#!/usr/bin/env python3
"""Validate whether a candidate river subgraph is safe for formal training."""

from __future__ import annotations

from scripts.common.terminal_output import console

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NODE_PATH = ROOT / "data/metadata/qujiang_continuous_subgraph_nodes.csv"
EDGE_PATH = ROOT / "data/metadata/qujiang_continuous_subgraph_edges.csv"
OUTPUT_DIR = ROOT / "outputs/quality/continuous_subgraph"

STATION_LEVEL_COORDINATES = {
    "official_section_coordinate",
    "official_station_coordinate",
    "manually_verified_station_coordinate",
}
SAME_REACH_FLOW_TYPES = {
    "same_reach_observed",
    "same_reach_gauged",
    "same_reach_mass_balance_estimate",
}
VERIFIED_EDGE_STATUSES = {
    "verified_direct",
    "verified_modeled_neighbor",
}


@dataclass(frozen=True)
class GraphReadiness:
    topology_ready: bool
    flow_ready: bool
    graph_ready: bool
    issue_codes: tuple[str, ...]
    issues: pd.DataFrame


def _append_issue(
    rows: list[dict[str, object]],
    scope: str,
    item: str,
    code: str,
    detail: str,
) -> None:
    rows.append({"scope": scope, "item": item, "issue_code": code, "detail": detail})


def evaluate_graph_readiness(nodes: pd.DataFrame, edges: pd.DataFrame) -> GraphReadiness:
    """Check node, topology, and flow evidence before any formal graph run."""
    node_required = {
        "station",
        "river",
        "longitude",
        "latitude",
        "coordinate_confidence",
        "coordinate_source_url",
        "coordinate_required",
        "data_ready",
    }
    edge_required = {
        "source_station",
        "target_station",
        "river",
        "direct_relation_status",
        "relation_source_url",
        "hidden_critical_node",
        "flow_station",
        "flow_mapping_type",
        "flow_source_path",
    }
    missing_nodes = sorted(node_required - set(nodes.columns))
    missing_edges = sorted(edge_required - set(edges.columns))
    if missing_nodes or missing_edges:
        raise ValueError(
            f"Missing evidence columns: nodes={missing_nodes}, edges={missing_edges}"
        )

    issues: list[dict[str, object]] = []
    node_names = set(nodes["station"].astype(str))
    duplicate_nodes = nodes["station"].astype(str).duplicated(keep=False)
    for station in nodes.loc[duplicate_nodes, "station"].astype(str).unique():
        _append_issue(issues, "node", station, "duplicate_node", "Station appears more than once.")

    for row in nodes.itertuples(index=False):
        station = str(row.station)
        if bool(row.coordinate_required):
            if str(row.coordinate_confidence) not in STATION_LEVEL_COORDINATES:
                _append_issue(
                    issues,
                    "node",
                    station,
                    "node_coordinate_not_station_level",
                    f"coordinate_confidence={row.coordinate_confidence}",
                )
            if pd.isna(row.longitude) or pd.isna(row.latitude) or not str(row.coordinate_source_url).strip():
                _append_issue(
                    issues,
                    "node",
                    station,
                    "node_coordinate_evidence_missing",
                    "Station coordinate or its source URL is missing.",
                )
        elif str(row.coordinate_confidence) != "official_relational_order_only" or not str(row.coordinate_source_url).strip():
            _append_issue(
                issues,
                "node",
                station,
                "node_relational_evidence_missing",
                "A relation-only node requires an official ordering source URL.",
            )
        if not bool(row.data_ready):
            _append_issue(
                issues,
                "node",
                station,
                "node_data_not_ready",
                "The station does not meet the declared split-coverage gate.",
            )

    for row in edges.itertuples(index=False):
        edge_name = f"{row.source_station}->{row.target_station}"
        if str(row.source_station) not in node_names or str(row.target_station) not in node_names:
            _append_issue(
                issues,
                "edge",
                edge_name,
                "edge_endpoint_missing",
                "Both edge endpoints must exist in the node manifest.",
            )
        if str(row.direct_relation_status) not in VERIFIED_EDGE_STATUSES or not str(row.relation_source_url).strip():
            _append_issue(
                issues,
                "edge",
                edge_name,
                "edge_not_verified_direct",
                f"direct_relation_status={row.direct_relation_status}",
            )
        if bool(row.hidden_critical_node):
            _append_issue(
                issues,
                "edge",
                edge_name,
                "edge_skips_critical_node",
                "A known critical intermediate section or confluence is skipped.",
            )
        if str(row.flow_mapping_type) not in SAME_REACH_FLOW_TYPES:
            _append_issue(
                issues,
                "edge",
                edge_name,
                "edge_flow_not_same_reach",
                f"flow_mapping_type={row.flow_mapping_type}",
            )
        if not str(row.flow_station).strip() or not str(row.flow_source_path).strip():
            _append_issue(
                issues,
                "edge",
                edge_name,
                "edge_flow_evidence_missing",
                "Flow station or source path is missing.",
            )

    issue_frame = pd.DataFrame(
        issues,
        columns=("scope", "item", "issue_code", "detail"),
    )
    codes = tuple(dict.fromkeys(issue_frame["issue_code"].tolist())) if len(issue_frame) else ()
    topology_codes = {
        "duplicate_node",
        "node_coordinate_not_station_level",
        "node_coordinate_evidence_missing",
        "node_relational_evidence_missing",
        "node_data_not_ready",
        "edge_endpoint_missing",
        "edge_not_verified_direct",
        "edge_skips_critical_node",
    }
    flow_codes = {"edge_flow_not_same_reach", "edge_flow_evidence_missing"}
    topology_ready = not any(code in topology_codes for code in codes)
    flow_ready = not any(code in flow_codes for code in codes)
    return GraphReadiness(
        topology_ready=topology_ready,
        flow_ready=flow_ready,
        graph_ready=topology_ready and flow_ready,
        issue_codes=codes,
        issues=issue_frame,
    )


def write_report(result: GraphReadiness, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.issues.to_csv(OUTPUT_DIR / "manual_review_items.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Continuous Subgraph Evidence Gate",
        "",
        f"- Nodes: {len(nodes)}",
        f"- Edges: {len(edges)}",
        f"- Topology ready: {result.topology_ready}",
        f"- Flow ready: {result.flow_ready}",
        f"- Formal graph ready: {result.graph_ready}",
        "",
        "## Issues",
        "",
    ]
    if result.issues.empty:
        report.append("No blocking evidence issues.")
    else:
        columns = list(result.issues.columns)
        report.append("| " + " | ".join(columns) + " |")
        report.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in result.issues.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
            report.append("| " + " | ".join(values) + " |")
    (OUTPUT_DIR / "evidence_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    nodes = pd.read_csv(NODE_PATH)
    edges = pd.read_csv(EDGE_PATH)
    result = evaluate_graph_readiness(nodes, edges)
    write_report(result, nodes, edges)
    console.print(
        f"topology_ready={result.topology_ready} "
        f"flow_ready={result.flow_ready} graph_ready={result.graph_ready}"
    )


if __name__ == "__main__":
    main()
