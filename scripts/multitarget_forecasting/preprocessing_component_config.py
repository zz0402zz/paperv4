"""Frozen protocol for separating robust scaling from Huber loss."""

from __future__ import annotations

from pathlib import Path

from scripts.multitarget_forecasting import config as base
from scripts.multitarget_forecasting import head_ablation_config as head
from scripts.multitarget_forecasting.io import safe_filename


EXPERIMENT_ID = "joint_five_target_preprocessing_component_ablation_24h_mixed_v1"
OUTPUT_DIR = base.OUTPUT_DIR / "验证集" / "预处理组件拆分消融"
CONTEXT = "24h"

# A、D、E已由既有实验生成；本模块只训练缺失的B、C。
TRAIN_VARIANTS = (
    "robust_mse",
    "standard_huber",
)
TRAIN_VARIANT_SPECS = {
    "robust_mse": {
        "input_scaler": "median_iqr",
        "target_scaler": "median_iqr",
        "loss": "mse",
        "log_targets": False,
    },
    "standard_huber": {
        "input_scaler": "mean_std",
        "target_scaler": "mean_std",
        "loss": "huber",
        "log_targets": False,
    },
}

REFERENCE_VARIANTS = (
    "original",
    "robust_huber",
    "robust_huber_log",
)
REPORT_VARIANTS = (
    "original",
    "robust_mse",
    "standard_huber",
    "robust_huber",
    "robust_huber_log",
)
VARIANT_LABELS = {
    "original": "A_原始均值标准化_MSE",
    "robust_mse": "B_中位数IQR标准化_MSE",
    "standard_huber": "C_均值标准化_Huber",
    "robust_huber": "D_中位数IQR标准化_Huber",
    "robust_huber_log": "E_稳健标准化_Huber_三指标对数",
}

CONTRASTS = (
    ("B对A_稳健标准化效果_MSE下", "original", "robust_mse"),
    ("C对A_Huber效果_均值标准化下", "original", "standard_huber"),
    ("D对B_Huber效果_稳健标准化下", "robust_mse", "robust_huber"),
    ("D对C_稳健标准化效果_Huber下", "standard_huber", "robust_huber"),
    ("D对A_稳健与Huber组合效果", "original", "robust_huber"),
    ("E对D_对数变换增量", "robust_huber", "robust_huber_log"),
    ("E对A_完整预处理效果", "original", "robust_huber_log"),
)

TARGET_OUTPUT_MODES = head.TARGET_OUTPUT_MODES
TARGETS = base.TARGETS
HORIZON_HOURS = base.HORIZON_HOURS
OUTPUT_STEPS = base.OUTPUT_STEPS
SCREENING_SEED = base.SCREENING_SEED
HUBER_DELTA = 1.0


def prediction_path(station: str, variant: str, seed: int) -> Path:
    return OUTPUT_DIR / "预测结果" / (
        "__".join(
            (
                safe_filename(VARIANT_LABELS[variant]),
                f"种子{seed}",
                safe_filename(station),
                "五指标联合预测",
            )
        )
        + ".npz"
    )


def model_path(station: str, variant: str, seed: int) -> Path:
    return OUTPUT_DIR / "模型" / prediction_path(station, variant, seed).with_suffix(
        ".pt"
    ).name
