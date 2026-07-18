#!/usr/bin/env python3
"""Run the formal five-seed per-target input-group experiment."""

from scripts.gru.target_input_group_multiseed import run_formal


def main() -> int:
    return run_formal()


if __name__ == "__main__":
    raise SystemExit(main())
