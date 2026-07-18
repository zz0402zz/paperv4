#!/usr/bin/env python3
"""Fixed constants for the V2 direct-pair graph-message experiment."""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol

INPUT_STEPS = 6
OUTPUT_STEPS = 9
INPUT_HOURS = INPUT_STEPS * 4
OUTPUT_HOURS = OUTPUT_STEPS * 4
LOSS_NAME = "l1"
FORMAL_SEEDS = protocol.FORMAL_SEEDS
PILOT_SEED = protocol.PILOT_SEED
PAIRS = (
    ("浮石渡", "下童"),
    ("台口", "南江桥"),
    ("章店", "洪坞桥"),
)
OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage4_direct_pair_graph"
PILOT_DIR = OUTPUT_DIR / "pilot_seed42"
FORMAL_DIR = OUTPUT_DIR / "formal_multiseed"
TARGET_FEATURE_COLUMNS = protocol.TARGET_FEATURE_COLUMNS
INPUT_FEATURE_COLUMNS = protocol.INPUT_FEATURE_COLUMNS
LAG_AUDIT_PATH = OUTPUT_DIR / "physical_lag_audit.csv"
CONTROL_AUDIT_PATH = OUTPUT_DIR / "control_audit.csv"


def ensure_output_dirs() -> None:
    """Create the stage output folders without touching legacy results."""
    for path in (OUTPUT_DIR, PILOT_DIR, FORMAL_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
