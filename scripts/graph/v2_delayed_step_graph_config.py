#!/usr/bin/env python3
"""Fixed protocol for the V2 delayed step-graph pilot."""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol

SELF_INPUT_STEPS = 6
GRAPH_INPUT_STEPS = (6, 9)
OUTPUT_STEPS = 9
SELF_INPUT_HOURS = SELF_INPUT_STEPS * 4
GRAPH_INPUT_HOURS = tuple(step * 4 for step in GRAPH_INPUT_STEPS)
OUTPUT_HOURS = OUTPUT_STEPS * 4
LOSS_NAME = "l1"
PILOT_SEED = protocol.PILOT_SEED
FORMAL_SEEDS = protocol.FORMAL_SEEDS
PAIRS = (("浮石渡", "下童"), ("下童", "横山"))
REVERSE_SOURCES = {"下童": "横山", "横山": "将军岩"}

OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage4b_delayed_step_graph"
PILOT_DIR = OUTPUT_DIR / "pilot_seed42"
FORMAL_DIR = OUTPUT_DIR / "formal_multiseed"
LAG_AUDIT_PATH = OUTPUT_DIR / "physical_lag_audit.csv"

INPUT_FEATURE_COLUMNS = protocol.INPUT_FEATURE_COLUMNS
TARGET_FEATURE_COLUMNS = protocol.TARGET_FEATURE_COLUMNS


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, PILOT_DIR, FORMAL_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
