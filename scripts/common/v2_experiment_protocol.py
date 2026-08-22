"""Frozen paths and time boundaries shared by the current V2 study."""

from __future__ import annotations

from pathlib import Path

from scripts.common.wq_gru_data import FEATURE_COLUMNS


DATA_VERSION = "v2_reprocessed_20260710"
OBSERVED_DATA_PATH = Path("data/processed/v2/quantity_4h_observed.csv")
QUALITY_DATA_PATH = Path("data/processed/v2/quantity_4h_quality.csv")
OUTPUT_ROOT = Path("outputs")

START_DATE = "2022-01-01"
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
RESAMPLE_RULE = "4h"

TARGET_FEATURE_COLUMNS = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
INPUT_FEATURE_COLUMNS = FEATURE_COLUMNS
FORMAL_SEEDS = (17, 42, 73, 101, 202)
