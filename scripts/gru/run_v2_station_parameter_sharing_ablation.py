#!/usr/bin/env python3
"""Run the V2 shared, station-embedding, and local D-GRU ablation."""

import argparse

from scripts.gru import station_parameter_sharing as experiment
from scripts.common import v2_experiment_protocol as protocol

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=protocol.PILOT_SEED)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.formal:
        experiment.run_formal()
    else:
        experiment.run_seed(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
