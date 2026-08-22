"""Frozen protocol for joint five-target context-length screening."""

from __future__ import annotations

from pathlib import Path

from scripts.common import v2_experiment_protocol as protocol


EXPERIMENT_ID = "joint_five_target_forward_early_stopping_4_72h_v2"
OUTPUT_DIR = protocol.OUTPUT_ROOT / "多指标联合水质预测"
VALIDATION_DIR = OUTPUT_DIR / "验证集" / "时间前向早停输入尺度消融"

OBSERVED_DATA_PATH = protocol.OBSERVED_DATA_PATH
QUALITY_DATA_PATH = protocol.QUALITY_DATA_PATH
START_DATE = protocol.START_DATE
TRAIN_END = protocol.TRAIN_END
VAL_END = protocol.VAL_END

STEP_HOURS = 4
HORIZON_STEPS = tuple(range(1, 19))
HORIZON_HOURS = tuple(step * STEP_HOURS for step in HORIZON_STEPS)
OUTPUT_STEPS = len(HORIZON_STEPS)
INPUT_FEATURES = protocol.INPUT_FEATURE_COLUMNS
TARGETS = protocol.TARGET_FEATURE_COLUMNS
TARGET_MODES = ("absolute", "delta")
TARGET_MODE_LABELS = {"absolute": "原值", "delta": "变化量"}

# All variants use the same seven-day sample set. Shorter variants see only the
# last part of that sequence, which makes validation rows exactly paired.
MAX_SEQUENCE_STEPS = 42
CONTEXT_STEPS = {
    "24h": 6,
    "72h": 18,
    "7d": 42,
    "multiscale": 6,
}
CONTEXT_LABELS = {
    "24h": "24小时原始历史",
    "72h": "72小时原始历史",
    "7d": "7天原始历史",
    "multiscale": "24小时加多尺度周期特征",
}
CONTEXTS = tuple(CONTEXT_STEPS)

LAG_HOURS = (24, 48, 168, 720, 8760)
ROLLING_HOURS = (24, 72, 168, 720)

FORMAL_SEEDS = protocol.FORMAL_SEEDS
SCREENING_SEED = 42
GRU_HIDDEN_SIZE = 128
CONTEXT_HIDDEN_SIZE = 128
FUSION_HIDDEN_SIZE = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
MIN_EPOCHS = 20
EVALUATION_EVERY = 5
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4
INTERNAL_VAL_START = "2023-07-01"

# These rules are for a methodological event-warning validation only. They are
# fitted from the training split and are not regulatory water-quality limits.
WARNING_QUANTILE = 0.10
WARNING_DIRECTIONS = {
    "pH(无量纲)": "two_sided",
    "溶解氧(mg/L)": "lower",
    "高锰酸盐指数(mg/L)": "upper",
    "氨氮(mg/L)": "upper",
    "总磷(mg/L)": "upper",
}


def prediction_path(
    station: str, context: str, target_mode: str, seed: int
) -> Path:
    from scripts.multitarget_forecasting.io import safe_filename

    filename = "__".join(
        (
            safe_filename(CONTEXT_LABELS[context]),
            f"{TARGET_MODE_LABELS[target_mode]}输出",
            f"种子{seed}",
            safe_filename(station),
            "五指标联合预测",
        )
    )
    return VALIDATION_DIR / "预测结果" / f"{filename}.npz"


def model_path(station: str, context: str, target_mode: str, seed: int) -> Path:
    return (
        VALIDATION_DIR
        / "模型"
        / prediction_path(station, context, target_mode, seed).with_suffix(".pt").name
    )
