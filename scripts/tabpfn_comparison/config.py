"""Frozen protocol for the five-model 2024 comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol
from scripts.data import jinhua_panel


OUTPUT_DIR = Path("outputs/paper/tabpfn_2024_comparison")
MAINLINE_VALIDATION_DIR = Path("outputs/paper/model_comparison/validation_2024")
FEATURE_SELECTION_PATH = Path("data/metadata/mainline_selected_features.csv")

STATIONS = jinhua_panel.STATIONS
TARGETS = protocol.TARGET_FEATURE_COLUMNS
SEEDS = protocol.FORMAL_SEEDS

START_DATE = protocol.START_DATE
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END
STEP_HOURS = 4
INPUT_STEPS = 6
OUTPUT_STEPS = 18

# The paper reported the checkpoint identifier ``2noar4o2`` and used the last
# 4096 observations.  The maintained package only exposes ModelVersion.V2 as a
# stable public selector, so this benchmark does not claim checkpoint identity
# without a matching file hash.  TS-3 ships a 32768 default; our history is
# shorter, so it retains all available history.
TABPFN_TS_V2_CONTEXT = 4096
TABPFN_TS3_CONTEXT = 32768
NATIVE_BATCH_SIZE = 1
NATIVE_CHECKPOINT_EVERY_BATCHES = 32

DELTA_GRU_KEY = "delta_gru_diff"
MATCHED_GRU_KEY = "delta_gru_matched"
TABPFN_TS_V2_KEY = "tabpfn_ts_v2"
TABPFN_TS3_KEY = "tabpfn_ts3"
DELTA_TABPFN_KEY = "delta_tabpfn_v2"
PERSISTENCE_KEY = "persistence"

MODEL_KEYS = (
    DELTA_GRU_KEY,
    MATCHED_GRU_KEY,
    TABPFN_TS_V2_KEY,
    TABPFN_TS3_KEY,
    DELTA_TABPFN_KEY,
)
TABPFN_KEYS = (TABPFN_TS_V2_KEY, TABPFN_TS3_KEY, DELTA_TABPFN_KEY)


@dataclass(frozen=True)
class NativeModelSpec:
    key: str
    version: str
    max_context_length: int


NATIVE_SPECS = {
    TABPFN_TS_V2_KEY: NativeModelSpec(
        TABPFN_TS_V2_KEY,
        "v2",
        TABPFN_TS_V2_CONTEXT,
    ),
    TABPFN_TS3_KEY: NativeModelSpec(
        TABPFN_TS3_KEY,
        "ts3",
        TABPFN_TS3_CONTEXT,
    ),
}


def model_seed(model: str, seed: int) -> int:
    """Native zero-shot TS variants use one frozen seed; Delta-TabPFN uses five."""
    return 0 if model in NATIVE_SPECS else int(seed)


def model_seeds(model: str) -> tuple[int, ...]:
    return (0,) if model in NATIVE_SPECS else SEEDS
