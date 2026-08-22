"""Frozen validation-only protocol for preprocessing ablations."""

from __future__ import annotations

from pathlib import Path

from scripts.multitarget_forecasting import config as base
from scripts.multitarget_forecasting import head_ablation_config as head
from scripts.multitarget_forecasting.io import safe_filename


EXPERIMENT_ID = "joint_five_target_preprocessing_ablation_24h_mixed_v1"
OUTPUT_DIR = base.OUTPUT_DIR / "验证集" / "数据预处理消融"
CONTEXT = "24h"

REFERENCE_VARIANT = "current_baseline"
REFERENCE_LABEL = "当前均值标准化_MSE"
VARIANTS = (
    "robust_huber",
    "robust_huber_log",
    "robust_huber_quality",
    "robust_huber_log_quality",
)
VARIANT_LABELS = {
    REFERENCE_VARIANT: REFERENCE_LABEL,
    "robust_huber": "中位数IQR标准化_Huber",
    "robust_huber_log": "稳健标准化_Huber_三指标对数",
    "robust_huber_quality": "稳健标准化_Huber_质量标记",
    "robust_huber_log_quality": "稳健标准化_Huber_三指标对数_质量标记",
}
VARIANT_SPECS = {
    "robust_huber": {"log_targets": False, "quality_aware": False},
    "robust_huber_log": {"log_targets": True, "quality_aware": False},
    "robust_huber_quality": {"log_targets": False, "quality_aware": True},
    "robust_huber_log_quality": {"log_targets": True, "quality_aware": True},
}

TARGET_OUTPUT_MODES = head.TARGET_OUTPUT_MODES
LOG_TARGETS = (
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
TARGETS = base.TARGETS
HORIZON_HOURS = base.HORIZON_HOURS
OUTPUT_STEPS = base.OUTPUT_STEPS
SCREENING_SEED = base.SCREENING_SEED
HUBER_DELTA = 1.0

# 这些队列只用于敏感性报告，不改变训练数据或主分析结果。
SENSITIVITY_COHORTS = {
    "all_25": {"label": "全部25站", "excluded_stations": ()},
    "without_puyang_exit": {
        "label": "不含浦阳江出口",
        "excluded_stations": ("浦阳江出口",),
    },
    "without_zha_kou": {
        "label": "不含闸口",
        "excluded_stations": ("闸口",),
    },
    "without_fushidu": {
        "label": "不含浮石渡",
        "excluded_stations": ("浮石渡",),
    },
    "without_three_review_stations": {
        "label": "不含浦阳江出口_闸口_浮石渡",
        "excluded_stations": ("浦阳江出口", "闸口", "浮石渡"),
    },
}


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
    return OUTPUT_DIR / "模型" / prediction_path(station, variant, seed).with_suffix(".pt").name
