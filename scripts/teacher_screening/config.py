"""Frozen development-only protocol for causal teacher screening."""

from __future__ import annotations

from pathlib import Path

from scripts.multitarget_forecasting import config as base
from scripts.multitarget_forecasting import preprocessing_ablation_config as prep
from scripts.multitarget_forecasting.io import safe_filename
from scripts.tabpfn_distillation import config as old_distillation


EXPERIMENT_ID = "causal_multiteacher_anchor_horizon_screening_v1"
OUTPUT_DIR = base.OUTPUT_DIR / "教师筛选" / "训练期因果OOF"
REPRESENTATIVE_STATIONS_PATH = (
    base.OUTPUT_DIR / "站点筛选" / "推荐跨站验证站点.json"
)

MODELS = ("tabpfn", "xgboost", "catboost_joint", "patch_transformer")
MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "xgboost": "XGBoost逐输出",
    "catboost_joint": "CatBoost联合多输出",
    "patch_transformer": "轻量补丁时序Transformer",
}

# The first gate deliberately uses four locations along the forecast curve.
# A shortlisted model must later be rerun on all 18 horizons; anchor winners are
# never silently extrapolated to the omitted horizons.
ANCHOR_HOURS = (4, 24, 48, 72)
ALL_HOURS = base.HORIZON_HOURS
TARGETS = base.TARGETS
TARGET_OUTPUT_MODES = prep.TARGET_OUTPUT_MODES
LOG_TARGETS = prep.LOG_TARGETS
OOF_FOLDS = old_distillation.OOF_FOLDS
SCREENING_SEED = 42
MIN_TRAIN_ROWS = 256

XGBOOST_ESTIMATORS = 400
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.03

CATBOOST_ITERATIONS = 500
CATBOOST_DEPTH = 8
CATBOOST_LEARNING_RATE = 0.03

PATCH_DIMENSION = 128
PATCH_LAYERS = 3
PATCH_HEADS = 4
PATCH_KERNEL = 2
PATCH_BATCH_SIZE = 128
PATCH_MAX_EPOCHS = 300
PATCH_EVALUATION_EVERY = 5
PATCH_PATIENCE = 12
PATCH_MIN_EPOCHS = 20
PATCH_LEARNING_RATE = 5e-4


def horizon_group(hours: int) -> str:
    if hours <= 24:
        return "短时距_4至24小时"
    if hours <= 48:
        return "中时距_28至48小时"
    return "长时距_52至72小时"


def prediction_path(
    station: str,
    model: str,
    seed: int,
    horizon_hours: tuple[int, ...],
) -> Path:
    horizon_label = "-".join(map(str, horizon_hours))
    filename = "__".join(
        (
            safe_filename(MODEL_LABELS[model]),
            f"种子{seed}",
            safe_filename(station),
            f"时距{horizon_label}小时",
        )
    )
    return OUTPUT_DIR / "教师OOF预测" / f"{filename}.npz"
