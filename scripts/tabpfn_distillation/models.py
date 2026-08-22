"""Pinned TabPFN teacher construction and numerical preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
import os

import numpy as np

from scripts.tabpfn_distillation import config


PINNED_TABPFN_VERSION = "8.1.0"
MODEL_IDENTITY = (
    "official TabPFN ModelVersion.V2 resolved through tabpfn==8.1.0; "
    "pretrained weights are loaded locally"
)

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


@lru_cache(maxsize=1)
def require_tabpfn() -> str:
    try:
        installed = metadata.version("tabpfn")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit("缺少 tabpfn，请使用项目的 .venv-tabpfn 环境。") from exc
    if installed != PINNED_TABPFN_VERSION:
        raise SystemExit(
            f"TabPFN版本不符合冻结协议: installed={installed}, "
            f"required={PINNED_TABPFN_VERSION}"
        )
    return installed


def make_teacher(seed: int = config.TEACHER_SEED):
    require_tabpfn()
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        device="auto",
        random_state=int(seed),
        fit_mode=config.TABPFN_FIT_MODE,
        memory_saving_mode=config.TABPFN_MEMORY_SAVING_MODE,
        show_progress_bar=False,
    )


def finite_feature_medians(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def apply_feature_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.broadcast_to(medians, values.shape)[invalid]
    return values


@dataclass(frozen=True)
class MaskedScaler:
    """Per-output standardization fitted only on approved true labels."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "MaskedScaler":
        values = np.asarray(values, dtype=float)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
        masked = np.where(valid, values, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(masked, axis=0)
            scale = np.nanstd(masked, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        values = np.asarray(values, dtype=float)
        flattened = values.reshape(-1, values.shape[-1])
        finite = np.where(np.isfinite(flattened), flattened, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(finite, axis=0)
            scale = np.nanstd(finite, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale
