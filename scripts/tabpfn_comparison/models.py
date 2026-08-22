"""Pinned local TabPFN constructors and inference helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
import os

import numpy as np

from scripts.tabpfn_comparison import config


PINNED_TABPFN_VERSION = "8.1.0"
PINNED_TABPFN_TS_VERSION = "1.2.0"
TS3_CHECKPOINT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
V2_CHECKPOINT_POLICY = (
    "official ModelVersion.V2 resolved through tabpfn==8.1.0's supported "
    "factory; weights are loaded locally by the TabPFN package"
)
TS3_CHECKPOINT_POLICY = f"official local TS-3 checkpoint: {TS3_CHECKPOINT}"

# Keep the experiment local and deterministic at the interface level. The
# official anonymous telemetry contains no model inputs, but is unnecessary here.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


@lru_cache(maxsize=2)
def require_dependencies(*, need_time_series: bool) -> dict[str, str]:
    packages = ["tabpfn"]
    if need_time_series:
        packages.append("tabpfn-time-series")
    versions: dict[str, str] = {}
    missing = []
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise SystemExit(
            f"缺少 {', '.join(missing)}。请按 "
            "`scripts/tabpfn_comparison/README.md` 安装专用环境；"
            "安装命令只安装代码依赖，"
            "首次创建本地模型时才会下载权重并要求接受许可证。"
        )
    expected = {"tabpfn": PINNED_TABPFN_VERSION}
    if need_time_series:
        expected["tabpfn-time-series"] = PINNED_TABPFN_TS_VERSION
    mismatched = {
        package: {"installed": versions[package], "required": version}
        for package, version in expected.items()
        if versions[package] != version
    }
    if mismatched:
        raise SystemExit(
            f"TabPFN依赖版本不符合冻结协议: {mismatched}。请使用项目专用"
            "`.venv-tabpfn`并按README中的系统对应命令重装。"
        )
    return versions


def model_identity(model: str) -> str | None:
    """Return an auditable identity statement without constructing a model."""
    if model in {"tabpfn_ts_v2", "delta_tabpfn_v2", "short_history_delta_tabpfn_v2"}:
        return V2_CHECKPOINT_POLICY
    if model == "tabpfn_ts3":
        return TS3_CHECKPOINT_POLICY
    return None


def make_v2_regressor(seed: int):
    require_dependencies(need_time_series=False)
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


def v2_regressor_config(seed: int) -> dict:
    """Resolve the v2 checkpoint through TabPFN's supported version factory."""
    regressor = make_v2_regressor(seed)
    return {
        "model_path": regressor.model_path,
        "device": "auto",
        "random_state": int(seed),
        "n_estimators": regressor.n_estimators,
        "softmax_temperature": regressor.softmax_temperature,
        "show_progress_bar": False,
    }


def make_native_pipeline(version: str, *, max_context_length: int):
    require_dependencies(need_time_series=True)
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    if version == "v2":
        from tabpfn_time_series.pipeline import TABPFN_TS_DEFAULT_FEATURES

        return TabPFNTSPipeline(
            max_context_length=max_context_length,
            temporal_features=TABPFN_TS_DEFAULT_FEATURES,
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection="mean",
            tabpfn_model_config=v2_regressor_config(0),
        )
    if version == "ts3":
        from tabpfn.model_loading import prepend_cache_path

        return TabPFNTSPipeline(
            max_context_length=max_context_length,
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection="mean",
            tabpfn_model_config={
                "model_path": prepend_cache_path(TS3_CHECKPOINT),
                "device": "auto",
                "random_state": 0,
                "show_progress_bar": False,
            },
        )
    raise ValueError(f"Unknown native TabPFN-TS version: {version}")


def finite_feature_medians(values: np.ndarray) -> np.ndarray:
    """Training-only median fill; preserve explicit mask columns as features."""
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
