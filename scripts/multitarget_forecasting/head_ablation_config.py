"""Frozen protocol for the target-specific forecasting-head ablation."""

from __future__ import annotations

from pathlib import Path

from scripts.multitarget_forecasting import config as base
from scripts.multitarget_forecasting.io import safe_filename


EXPERIMENT_ID = "joint_five_target_head_ablation_24h_mixed_v1"
OUTPUT_DIR = base.OUTPUT_DIR / "验证集" / "指标专属预测头消融"

CONTEXT = "24h"
VARIANTS = (
    "mixed_linear",
    "mixed_shared_mlp",
    "mixed_target_heads",
)
VARIANT_LABELS = {
    "mixed_linear": "共享线性头_指标混合表示",
    "mixed_shared_mlp": "共享非线性头_指标混合表示",
    "mixed_target_heads": "指标专属非线性头_指标混合表示",
}
REFERENCE_VARIANT = "uniform_delta_reference"
REFERENCE_LABEL = "原联合线性头_统一变化量"

# Selected from the preceding 25-station development-screening result. This
# mapping is frozen before the head ablation and must not be changed per station.
TARGET_OUTPUT_MODES = {
    "pH(无量纲)": "delta",
    "溶解氧(mg/L)": "delta",
    "高锰酸盐指数(mg/L)": "absolute",
    "氨氮(mg/L)": "delta",
    "总磷(mg/L)": "absolute",
}

# The two nonlinear alternatives have almost identical head capacity:
# shared MLP = 23,742 parameters; five target heads = 23,610 parameters.
SHARED_MLP_HIDDEN_SIZE = 108
TARGET_HEAD_HIDDEN_SIZE = 32

TARGETS = base.TARGETS
HORIZON_HOURS = base.HORIZON_HOURS
OUTPUT_STEPS = base.OUTPUT_STEPS
SCREENING_SEED = base.SCREENING_SEED


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
