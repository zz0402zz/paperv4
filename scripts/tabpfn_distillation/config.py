"""Frozen protocol for the 4--72 hour causal distillation study."""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS


EXPERIMENT_ID = "tabpfn_causal_distillation_4_72h_v1"
OUTPUT_DIR = protocol.OUTPUT_ROOT / "TabPFN因果蒸馏长时距"
EVALUATION_DIR_NAMES = {"val": "验证集", "test": "测试集"}

OBSERVED_DATA_PATH = protocol.OBSERVED_DATA_PATH
QUALITY_DATA_PATH = protocol.QUALITY_DATA_PATH
START_DATE = protocol.START_DATE
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END

STEP_HOURS = 4
INPUT_STEPS = 6
HORIZON_STEPS = tuple(range(1, 19))
HORIZON_HOURS = tuple(step * STEP_HOURS for step in HORIZON_STEPS)
OUTPUT_STEPS = len(HORIZON_STEPS)
INPUT_FEATURES = FEATURE_COLUMNS
TARGETS = protocol.TARGET_FEATURE_COLUMNS

# Teacher predictions are generated once in delta space. Both student target
# representations consume the same teacher information, which makes the
# absolute-versus-delta ablation identifiable.
TEACHER_SEED = 42
TARGET_MODES = ("absolute", "delta")
TARGET_MODE_LABELS = {"absolute": "原值", "delta": "变化量"}

SUPERVISED_ABSOLUTE_KEY = "supervised_absolute_gru"
SUPERVISED_DELTA_KEY = "supervised_delta_gru"
DISTILLED_ABSOLUTE_KEY = "causal_distilled_absolute_gru"
DISTILLED_DELTA_KEY = "causal_distilled_delta_gru"
STUDENT_KEYS = (
    SUPERVISED_ABSOLUTE_KEY,
    SUPERVISED_DELTA_KEY,
    DISTILLED_ABSOLUTE_KEY,
    DISTILLED_DELTA_KEY,
)
STUDENT_FILE_LABELS = {
    SUPERVISED_ABSOLUTE_KEY: "原值监督GRU",
    SUPERVISED_DELTA_KEY: "变化量监督GRU",
    DISTILLED_ABSOLUTE_KEY: "原值因果蒸馏GRU",
    DISTILLED_DELTA_KEY: "变化量因果蒸馏GRU",
}
STUDENT_TARGET_MODES = {
    SUPERVISED_ABSOLUTE_KEY: "absolute",
    SUPERVISED_DELTA_KEY: "delta",
    DISTILLED_ABSOLUTE_KEY: "absolute",
    DISTILLED_DELTA_KEY: "delta",
}
DISTILLED_KEYS = (DISTILLED_ABSOLUTE_KEY, DISTILLED_DELTA_KEY)

STUDENT_SEEDS = protocol.FORMAL_SEEDS
GRU_HIDDEN_SIZE = 64
GRU_CURRENT_HIDDEN_SIZE = 32
GRU_BATCH_SIZE = 128
GRU_EPOCHS = 100
GRU_LEARNING_RATE = 1e-3
DISTILLATION_WEIGHT = 0.5

TABPFN_FIT_MODE = "fit_preprocessors"
TABPFN_MEMORY_SAVING_MODE = True
TABPFN_PREDICTION_BATCH_SIZE = 16
MIN_TEACHER_TRAIN_ROWS = 256

# Each prediction block is strictly later than every label used to fit its
# teacher. The first six months are a warm-up and intentionally have no OOF
# teacher labels.
OOF_FOLDS = (
    ("2022年下半年", "2022-07-01", "2023-01-01"),
    ("2023年上半年", "2023-01-01", "2023-07-01"),
    ("2023年下半年", "2023-07-01", TRAIN_END),
)


def output_dir_for_split(evaluation_split: str) -> Path:
    if evaluation_split not in EVALUATION_DIR_NAMES:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split}")
    return OUTPUT_DIR / EVALUATION_DIR_NAMES[evaluation_split]


def student_target_mode(variant: str) -> str:
    try:
        return STUDENT_TARGET_MODES[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown student variant: {variant}") from exc


def is_distilled(variant: str) -> bool:
    if variant not in STUDENT_KEYS:
        raise ValueError(f"Unknown student variant: {variant}")
    return variant in DISTILLED_KEYS
