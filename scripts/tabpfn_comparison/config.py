"""Frozen protocol for the single-station short-history TabPFN comparison.

This experiment intentionally uses the canonical V2 observed data and its
quality sidecar. It is separate from TabPFN's ``ModelVersion.V2``: the latter
is a model version, while this module defines the project's data protocol.
"""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS


EXPERIMENT_ID = "stage_tabpfn_single_station_short_history_v1"
OUTPUT_DIR = protocol.OUTPUT_ROOT / "单站短历史TabPFN对比"
EVALUATION_DIR_NAMES = {"val": "验证集", "test": "测试集"}

OBSERVED_DATA_PATH = protocol.OBSERVED_DATA_PATH
QUALITY_DATA_PATH = protocol.QUALITY_DATA_PATH
START_DATE = protocol.START_DATE
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END
STEP_HOURS = 4

# The primary question is whether a 24-hour local history can beat a matched
# delta-GRU. Window-length sensitivity is a separately named rerun, never a
# hidden change to the primary result.
INPUT_STEPS = 6
OUTPUT_STEPS = 1
INPUT_FEATURES = FEATURE_COLUMNS
TARGETS = protocol.TARGET_FEATURE_COLUMNS
FORMAL_SEEDS = protocol.FORMAL_SEEDS

PERSISTENCE_KEY = "persistence"
DELTA_GRU_KEY = "short_history_delta_gru"
DELTA_TABPFN_KEY = "short_history_delta_tabpfn_v2"
MODEL_KEYS = (PERSISTENCE_KEY, DELTA_GRU_KEY, DELTA_TABPFN_KEY)
MODEL_FILE_LABELS = {
    PERSISTENCE_KEY: "持续性",
    DELTA_GRU_KEY: "变化量GRU",
    DELTA_TABPFN_KEY: "变化量TabPFN-v2",
}

# Fixed, matched GRU capacity. It is deliberately not selected on the V2
# validation set, so validation remains an evaluation set for this comparison.
GRU_HIDDEN_SIZE = 32
GRU_CURRENT_HIDDEN_SIZE = 16
GRU_BATCH_SIZE = 128
GRU_EPOCHS = 80
GRU_LEARNING_RATE = 1e-3

# A 24-hour window yields roughly two thousand validation origins per station.
# The cached TabPFN mode is fast but holds a large key-value cache on the GPU;
# these settings keep the comparison runnable on an 8 GB GPU.
TABPFN_FIT_MODE = "fit_preprocessors"
TABPFN_MEMORY_SAVING_MODE = True
TABPFN_PREDICTION_BATCH_SIZE = 16


def output_dir_for_split(evaluation_split: str) -> Path:
    """Keep validation and final-test predictions physically separate."""
    if evaluation_split not in EVALUATION_DIR_NAMES:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split}")
    return Path(OUTPUT_DIR) / EVALUATION_DIR_NAMES[evaluation_split]


def model_seeds(model: str, requested: tuple[int, ...]) -> tuple[int, ...]:
    """Persistence is deterministic and is saved once rather than duplicated."""
    if model == PERSISTENCE_KEY:
        return (0,)
    return requested
