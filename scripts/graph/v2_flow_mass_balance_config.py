#!/usr/bin/env python3
"""Fixed configuration for the V2 flow mass-balance ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol

@dataclass(frozen=True)
class FlowComponent:
    kind: str
    gauge_code: str
    subtract_code: str = ""


@dataclass(frozen=True)
class Confluence:
    target: str
    sources: tuple[str, str]
    target_gauge_code: str
    source_components: tuple[FlowComponent, FlowComponent]


CONFLUENCES = (
    Confluence(
        target="浮石渡",
        sources=("富足山", "双港口"),
        target_gauge_code="70100500",
        source_components=(
            FlowComponent("gauge", "70100350"),
            FlowComponent("gauge", "70104660"),
        ),
    ),
    Confluence(
        target="将军岩",
        sources=("横山", "费垅"),
        target_gauge_code="70100900",
        source_components=(
            FlowComponent("gauge", "70100500"),
            FlowComponent("gauge", "70108400"),
        ),
    ),
    Confluence(
        target="费垅",
        sources=("东关桥", "洪坞桥"),
        target_gauge_code="70108400",
        source_components=(
            FlowComponent("residual", "70108400", "70110050"),
            FlowComponent("gauge", "70110050"),
        ),
    ),
)

SELF_INPUT_STEPS = 6
GRAPH_INPUT_STEPS = 9
OUTPUT_STEPS = 9
WEIGHT_MODES = ("unweighted", "branch_normalized", "mass_balance")
PILOT_SEED = protocol.PILOT_SEED
FORMAL_SEEDS = protocol.FORMAL_SEEDS

OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage4c_flow_mass_balance"
PILOT_DIR = OUTPUT_DIR / "pilot_seed42"
FORMAL_DIR = OUTPUT_DIR / "formal_multiseed"
LAG_AUDIT_PATH = OUTPUT_DIR / "physical_lag_audit.csv"


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, PILOT_DIR, FORMAL_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
