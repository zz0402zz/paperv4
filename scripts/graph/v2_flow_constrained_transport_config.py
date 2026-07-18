#!/usr/bin/env python3
"""Fixed protocol for the V2 constrained flow-transport ablation."""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol

TARGET = "将军岩"
SOURCES = ("横山", "费垅")
VARIANTS = (
    "self_D_6to9",
    "full_unweighted",
    "full_branch_normalized",
    "full_mass_balance_3d",
    "transport_unweighted",
    "transport_branch_normalized",
    "transport_mass_balance_3d",
    "transport_mass_balance_raw1d",
    "transport_mass_static",
    "transport_mass_shuffled",
)
PILOT_SEED = protocol.PILOT_SEED
FORMAL_SEEDS = protocol.FORMAL_SEEDS

OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage4d_constrained_flow_transport"
PILOT_DIR = OUTPUT_DIR / "pilot_seed42"
FORMAL_DIR = OUTPUT_DIR / "formal_multiseed"


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, PILOT_DIR, FORMAL_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
