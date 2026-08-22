"""Protocol for preprocessing-by-output-representation interaction screening."""

from __future__ import annotations

from pathlib import Path

from scripts.multitarget_forecasting import config as base
from scripts.multitarget_forecasting import head_ablation_config as head
from scripts.multitarget_forecasting import preprocessing_ablation_config as prep
from scripts.multitarget_forecasting.io import safe_filename


EXPERIMENT_ID = "joint_five_target_preprocessing_mode_interaction_24h_v1"
OUTPUT_DIR = base.OUTPUT_DIR / "验证集" / "预处理与输出表示交互消融"
CONTEXT = "24h"

# A -> C只加Huber；C -> D再换稳健标准化；D -> E再加对数。
PREPROCESSINGS = ("A", "C", "D", "E")
PREPROCESSING_SPECS = {
    "A": {
        "label": "A_原始均值标准化_MSE",
        "input_scaler": "mean_std",
        "target_scaler": "mean_std",
        "loss": "mse",
        "log_targets": False,
    },
    "C": {
        "label": "C_均值标准化_Huber",
        "input_scaler": "mean_std",
        "target_scaler": "mean_std",
        "loss": "huber",
        "log_targets": False,
    },
    "D": {
        "label": "D_中位数IQR标准化_Huber",
        "input_scaler": "median_iqr",
        "target_scaler": "median_iqr",
        "loss": "huber",
        "log_targets": False,
    },
    "E": {
        "label": "E_稳健标准化_Huber_三指标对数",
        "input_scaler": "median_iqr",
        "target_scaler": "median_iqr",
        "loss": "huber",
        "log_targets": True,
    },
}

BASE_MODES = dict(head.TARGET_OUTPUT_MODES)
FLIP_TARGETS = {
    "ph": "pH(无量纲)",
    "do": "溶解氧(mg/L)",
    "codmn": "高锰酸盐指数(mg/L)",
    "nh3n": "氨氮(mg/L)",
    "tp": "总磷(mg/L)",
}
FLIPS = tuple(FLIP_TARGETS)
LOG_TARGETS = prep.LOG_TARGETS
TARGETS = base.TARGETS
HORIZON_HOURS = base.HORIZON_HOURS
OUTPUT_STEPS = base.OUTPUT_STEPS
SCREENING_SEED = base.SCREENING_SEED
HUBER_DELTA = prep.HUBER_DELTA


def flipped_modes(flip: str) -> dict[str, str]:
    if flip not in FLIP_TARGETS:
        raise ValueError(f"未知指标翻转: {flip}")
    modes = dict(BASE_MODES)
    target = FLIP_TARGETS[flip]
    modes[target] = "absolute" if modes[target] == "delta" else "delta"
    return modes


def flip_label(flip: str) -> str:
    target = FLIP_TARGETS[flip]
    destination = "原值" if flipped_modes(flip)[target] == "absolute" else "变化量"
    return f"仅{target}改为{destination}"


def prediction_path(station: str, preprocessing: str, flip: str, seed: int) -> Path:
    return OUTPUT_DIR / "预测结果" / (
        "__".join(
            (
                safe_filename(str(PREPROCESSING_SPECS[preprocessing]["label"])),
                safe_filename(flip_label(flip)),
                f"种子{seed}",
                safe_filename(station),
                "五指标联合预测",
            )
        )
        + ".npz"
    )
