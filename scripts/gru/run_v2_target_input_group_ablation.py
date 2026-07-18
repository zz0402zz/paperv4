#!/usr/bin/env python3
"""Run the seed-42 validation-only per-target input group ablation."""

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol
from scripts.gru.target_input_group_ablation import run_pilot


OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3e_target_input_group_ablation" / "pilot_seed42"


def main() -> int:
    return run_pilot(Path(OUTPUT_DIR), seed=protocol.PILOT_SEED)


if __name__ == "__main__":
    raise SystemExit(main())
