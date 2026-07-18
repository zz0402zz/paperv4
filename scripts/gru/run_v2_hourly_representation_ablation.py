#!/usr/bin/env python3
"""Compare mean, endpoint, and endpoint-plus-stat inputs on common endpoint targets."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS
from scripts.data.build_hourly_representation_ablation import (
    HOURLY_FEATURE_COLUMNS,
    METADATA_PATH,
    QUALITY_PATH,
    STATISTICS,
    VALUES_PATH,
    prefixed_feature,
)
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_self_feature_ablation as self_ablation
from scripts.gru import run_wentu_window_level_ablation as window


OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "stage3d_hourly_representation_ablation" / "pilot_seed42"
INPUT_STEPS = 6
OUTPUT_STEPS = 1
SEED = protocol.PILOT_SEED
TARGET_FEATURE_COLUMNS = protocol.TARGET_FEATURE_COLUMNS
MODES = ("mean_history", "endpoint_history", "endpoint_plus_window_stats")
CONTROL_MODES = ("endpoint_plus_shifted_window_stats",)
ALL_MODES = (*MODES, *CONTROL_MODES)
CONTROL_SHIFT_STEPS = 137

TERMINAL_MODE_NAMES = {
    "mean_history": "4h mean",
    "endpoint_history": "4h endpoint",
    "endpoint_plus_window_stats": "endpoint + hourly stats",
    "endpoint_plus_shifted_window_stats": "shifted-stat control",
    "persistence": "Persistence",
}


def diff_column(prefix: str, feature: str) -> str:
    return f"{prefix}_diff1__{feature}"


def input_columns_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "mean_history":
        return tuple(diff_column("mean", feature) for feature in FEATURE_COLUMNS)
    endpoint = tuple(diff_column("endpoint", feature) for feature in FEATURE_COLUMNS)
    if mode == "endpoint_history":
        return endpoint
    if mode == "endpoint_plus_window_stats":
        statistics = tuple(
            prefixed_feature(f"window_{statistic}", feature)
            for feature in HOURLY_FEATURE_COLUMNS
            for statistic in STATISTICS
        )
        return (*endpoint, *statistics)
    if mode == "endpoint_plus_shifted_window_stats":
        statistics = tuple(
            f"shift{CONTROL_SHIFT_STEPS}__{prefixed_feature(f'window_{statistic}', feature)}"
            for feature in HOURLY_FEATURE_COLUMNS
            for statistic in STATISTICS
        )
        return (*endpoint, *statistics)
    raise ValueError(f"Unknown representation mode: {mode}")


def load_ablation_data(values_path: Path = VALUES_PATH, quality_path: Path = QUALITY_PATH) -> pd.DataFrame:
    values = pd.read_csv(values_path)
    quality = pd.read_csv(quality_path)
    values["time"] = pd.to_datetime(values["time"])
    quality["time"] = pd.to_datetime(quality["time"])
    quality_columns = [column for column in quality if column not in {"station", "time"}]
    data = values.merge(
        quality[["station", "time", *quality_columns]],
        on=["station", "time"],
        how="left",
        validate="one_to_one",
    )
    frames = []
    for _, group in data.groupby("station", sort=True):
        group = group.sort_values("time").copy()
        for feature in FEATURE_COLUMNS:
            group[diff_column("endpoint", feature)] = pd.to_numeric(group[feature], errors="coerce").diff()
            group[diff_column("mean", feature)] = pd.to_numeric(
                group[prefixed_feature("mean", feature)], errors="coerce"
            ).diff()
        frames.append(group)
    data = pd.concat(frames, ignore_index=True).sort_values(["station", "time"])
    data = data[data["time"] >= pd.Timestamp(protocol.START_DATE)].reset_index(drop=True)

    statistic_columns = [
        prefixed_feature(f"window_{statistic}", feature)
        for feature in HOURLY_FEATURE_COLUMNS
        for statistic in STATISTICS
    ]
    split_name = np.select(
        [data["time"] < pd.Timestamp(protocol.TRAIN_END), data["time"] < pd.Timestamp(protocol.VAL_END)],
        ["train", "val"],
        default="test",
    )
    data["_split_for_control"] = split_name
    shifted_parts = []
    for _, group in data.groupby(["station", "_split_for_control"], sort=False):
        group = group.sort_values("time").copy()
        for column in statistic_columns:
            group[f"shift{CONTROL_SHIFT_STEPS}__{column}"] = group[column].shift(CONTROL_SHIFT_STEPS)
        shifted_parts.append(group)
    return (
        pd.concat(shifted_parts, ignore_index=True)
        .drop(columns="_split_for_control")
        .sort_values(["station", "time"])
        .reset_index(drop=True)
    )


def _delta_arrays(arrays: dict[str, np.ndarray], raw_split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    last = raw_split["last_target"][:, None, :, :]
    return {
        "pred": arrays["pred"] - last,
        "true": raw_split["y"],
        "mask": arrays["mask"],
    }


def _persistence_arrays(raw_split: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    last = raw_split["last_target"][:, None, :, :]
    absolute = {
        "pred": np.repeat(last, raw_split["y_abs"].shape[1], axis=1),
        "true": raw_split["y_abs"],
        "mask": raw_split["y_mask"],
    }
    change = {
        "pred": np.zeros_like(raw_split["y"]),
        "true": raw_split["y"],
        "mask": raw_split["y_mask"],
    }
    return absolute, change


def _assert_common_targets(
    reference: dict[str, dict[str, np.ndarray]] | None,
    candidate: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    if reference is None:
        return candidate
    for split in ("train", "val", "test"):
        for key in ("y", "y_abs", "last_target", "y_mask", "target_start", "target_end"):
            left = reference[split][key]
            right = candidate[split][key]
            if left.shape != right.shape:
                raise ValueError(f"Target shape differs for {split}/{key}: {left.shape} != {right.shape}")
            if np.issubdtype(left.dtype, np.datetime64):
                equal = np.array_equal(left, right)
            else:
                equal = np.allclose(left, right, equal_nan=True) if left.dtype != bool else np.array_equal(left, right)
            if not equal:
                raise ValueError(f"Target content differs for {split}/{key}")
    return reference


def _macro_station_rmse(metrics: dict) -> float | None:
    values = [
        item.get("rmse")
        for item in metrics.get("station_metrics", {}).values()
        if item.get("rmse") is not None and np.isfinite(item.get("rmse"))
    ]
    return float(np.mean(values)) if values else None


def _skill(model_rmse: float | None, persistence_rmse: float | None) -> float | None:
    if model_rmse is None or persistence_rmse is None or persistence_rmse == 0:
        return None
    return float(100.0 * (persistence_rmse - model_rmse) / persistence_rmse)


def _sign_accuracy(arrays: dict[str, np.ndarray]) -> float | None:
    true = arrays["true"]
    pred = arrays["pred"]
    mask = arrays["mask"].astype(bool) & np.isfinite(true) & np.isfinite(pred) & (np.abs(true) > 1e-12)
    return float((np.sign(pred[mask]) == np.sign(true[mask])).mean()) if mask.any() else None


def _tail_rmse(arrays: dict[str, np.ndarray], threshold: float) -> tuple[float | None, int]:
    error = arrays["pred"] - arrays["true"]
    mask = (
        arrays["mask"].astype(bool)
        & np.isfinite(error)
        & np.isfinite(arrays["true"])
        & (np.abs(arrays["true"]) >= threshold)
    )
    return (float(np.sqrt(np.mean(error[mask] ** 2))), int(mask.sum())) if mask.any() else (None, 0)


def _aggregate(arrays: dict[str, dict[str, np.ndarray]], stations: tuple[str, ...]) -> dict:
    return self_ablation.aggregate_single_target_arrays(arrays, stations, TARGET_FEATURE_COLUMNS)


def _metric_rows(
    results: dict[str, dict[str, dict]],
    persistence: dict[str, dict[str, dict]],
) -> list[dict[str, object]]:
    rows = []
    for split in ("train", "val", "test"):
        persistence_abs = persistence[split]["absolute"]
        persistence_delta = persistence[split]["delta"]
        for mode in (*ALL_MODES, "persistence"):
            if mode == "persistence":
                absolute = persistence_abs
                change = persistence_delta
                parameters = 0
            else:
                absolute = results[mode][split]["absolute"]
                change = results[mode][split]["delta"]
                parameters = results[mode]["parameters"]
            rows.append(
                {
                    "split": split,
                    "mode": mode,
                    "input_steps": INPUT_STEPS,
                    "window_hours": INPUT_STEPS * 4,
                    "raw_input_channels": 0 if mode == "persistence" else len(input_columns_for_mode(mode)),
                    "parameters_per_target": parameters,
                    "valid_points": absolute.get("valid_points"),
                    "mae": absolute.get("mae"),
                    "rmse": absolute.get("rmse"),
                    "nse": absolute.get("nse"),
                    "macro_station_rmse": _macro_station_rmse(absolute),
                    "skill_vs_persistence_pct": _skill(absolute.get("rmse"), persistence_abs.get("rmse")),
                    "delta_mae": change.get("mae"),
                    "delta_rmse": change.get("rmse"),
                    "delta_nse": change.get("nse"),
                }
            )
    return rows


def _feature_rows(
    results: dict[str, dict[str, dict]],
    persistence: dict[str, dict[str, dict]],
) -> list[dict[str, object]]:
    rows = []
    for split in ("val", "test"):
        for mode in (*ALL_MODES, "persistence"):
            absolute = persistence[split]["absolute"] if mode == "persistence" else results[mode][split]["absolute"]
            change = persistence[split]["delta"] if mode == "persistence" else results[mode][split]["delta"]
            for feature in TARGET_FEATURE_COLUMNS:
                persistence_rmse = persistence[split]["absolute"]["feature_rmse"].get(feature)
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "feature": feature,
                        "valid_points": absolute["feature_valid_points"].get(feature, 0),
                        "mae": absolute["feature_mae"].get(feature),
                        "rmse": absolute["feature_rmse"].get(feature),
                        "nse": absolute["feature_nse"].get(feature),
                        "skill_vs_persistence_pct": _skill(absolute["feature_rmse"].get(feature), persistence_rmse),
                        "delta_mae": change["feature_mae"].get(feature),
                        "delta_rmse": change["feature_rmse"].get(feature),
                        "delta_nse": change["feature_nse"].get(feature),
                    }
                )
    return rows


def _station_rows(results: dict[str, dict[str, dict]]) -> list[dict[str, object]]:
    rows = []
    for split in ("val", "test"):
        for mode in ALL_MODES:
            metrics = results[mode][split]["absolute"]["station_metrics"]
            for station, item in metrics.items():
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "station": station,
                        "valid_points": item.get("valid_points", 0),
                        "mae": item.get("mae"),
                        "rmse": item.get("rmse"),
                        "nse": item.get("nse"),
                    }
                )
    return rows


def _station_feature_rows(results: dict[str, dict[str, dict]]) -> list[dict[str, object]]:
    rows = []
    for split in ("val", "test"):
        for mode in ALL_MODES:
            station_metrics = results[mode][split]["absolute"]["station_metrics"]
            for station, item in station_metrics.items():
                for feature in TARGET_FEATURE_COLUMNS:
                    rows.append(
                        {
                            "split": split,
                            "mode": mode,
                            "station": station,
                            "feature": feature,
                            "valid_points": item["feature_valid_points"].get(feature, 0),
                            "mae": item["feature_mae"].get(feature),
                            "rmse": item["feature_rmse"].get(feature),
                            "nse": item["feature_nse"].get(feature),
                        }
                    )
    return rows


def _tail_rows(
    arrays_by_mode: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    persistence_arrays: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, object]]:
    thresholds = {}
    for feature in TARGET_FEATURE_COLUMNS:
        train = persistence_arrays["train"][feature]["delta"]
        mask = train["mask"].astype(bool) & np.isfinite(train["true"])
        thresholds[feature] = float(np.quantile(np.abs(train["true"][mask]), 0.90))

    rows = []
    for split in ("val", "test"):
        for mode in (*ALL_MODES, "persistence"):
            for feature in TARGET_FEATURE_COLUMNS:
                arrays = (
                    persistence_arrays[split][feature]["delta"]
                    if mode == "persistence"
                    else arrays_by_mode[mode][split][feature]["delta"]
                )
                rmse, points = _tail_rmse(arrays, thresholds[feature])
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "feature": feature,
                        "train_p90_abs_delta_threshold": thresholds[feature],
                        "tail_points": points,
                        "tail_delta_rmse": rmse,
                        "sign_accuracy": _sign_accuracy(arrays),
                    }
                )
    return rows


def _write_report(overall: pd.DataFrame, features: pd.DataFrame, stations: pd.DataFrame) -> None:
    validation = overall[overall["split"].eq("val")].sort_values("macro_station_rmse")
    test = overall[overall["split"].eq("test")].sort_values("macro_station_rmse")
    candidates = validation[validation["mode"].isin(MODES)]
    winner = str(candidates.iloc[0]["mode"])
    station_pivot = stations[stations["split"].eq("val")].pivot(index="station", columns="mode", values="rmse")
    endpoint_wins = int((station_pivot["endpoint_history"] < station_pivot["mean_history"]).sum())
    stats_wins = int((station_pivot["endpoint_plus_window_stats"] < station_pivot["endpoint_history"]).sum())
    control_wins = int(
        (station_pivot["endpoint_plus_window_stats"] < station_pivot["endpoint_plus_shifted_window_stats"]).sum()
    )
    feature_validation = features[features["split"].eq("val")].sort_values(["feature", "rmse"])
    lines = [
        "# 小时数据表示消融：共同端点真值",
        "",
        "## 固定口径",
        "- 25 个站点，2022-2023 训练、2024 验证、2025 测试。",
        "- 过去 6 个四小时时间步（24h）预测未来 1 步（4h）。",
        "- 五个目标分别训练 D-GRU：历史变化序列走 GRU，当前端点目标值走 MLP。",
        "- 三种方案使用完全相同的端点当前值和端点未来标签；只改变历史输入。",
        "- mean_history：9 个指标的四小时均值/最新值变化。",
        "- endpoint_history：9 个指标的四小时端点变化。",
        "- endpoint_plus_window_stats：端点变化加小时指标窗口 mean/max/std/slope。",
        f"- endpoint_plus_shifted_window_stats：同维度负对照，窗口统计在同站同划分内只向过去错位 {CONTROL_SHIFT_STEPS} 步。",
        "- 特征固定使用 all9，不重新筛 corr-top3，避免特征选择混淆预处理效应。",
        "",
        "## 验证集决策",
        f"- 验证集宏平均站点 RMSE 最优方案：`{winner}`。",
        f"- 端点历史相对均值历史改善站点：{endpoint_wins}/25。",
        f"- 加窗口统计相对纯端点改善站点：{stats_wins}/25。",
        f"- 正确对齐统计优于同容量错位对照的站点：{control_wins}/25。",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## 锁定后的测试集读数",
        "```text",
        test.to_string(index=False),
        "```",
        "",
        "## 验证集分指标",
        "```text",
        feature_validation.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = SEED) -> int:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper.SEED = int(seed)
    dual.INPUT_STEPS = INPUT_STEPS
    dual.OUTPUT_STEPS = OUTPUT_STEPS
    random.seed(seed)
    np.random.seed(seed)

    data = load_ablation_data()
    stations = tuple(sorted(data["station"].astype(str).unique()))
    torch = base.require_torch()
    torch.manual_seed(seed)
    device = base.choose_device(torch)
    console.phase("hourly representation ablation")
    console.info(
        "dataset",
        stations=len(stations),
        input=f"{INPUT_STEPS} steps / {INPUT_STEPS * 4}h",
        output=f"{OUTPUT_STEPS} step / {OUTPUT_STEPS * 4}h",
        targets=len(TARGET_FEATURE_COLUMNS),
    )
    console.info("runtime", device=device, seed=seed)

    manifest = protocol.build_run_manifest(
        experiment="stage3d_hourly_representation_common_endpoint_6to1",
        output_dir=OUTPUT_DIR,
        seed=seed,
        observed_path=VALUES_PATH,
        quality_path=QUALITY_PATH,
        code_paths=(
            Path("scripts/data/build_hourly_representation_ablation.py"),
            Path("scripts/gru/run_v2_hourly_representation_ablation.py"),
        ),
    )
    manifest["representation_metadata"] = str(METADATA_PATH)
    manifest["common_target"] = "4h endpoint observation"
    manifest["modes"] = {mode: list(input_columns_for_mode(mode)) for mode in ALL_MODES}
    manifest["negative_control"] = {
        "mode": CONTROL_MODES[0],
        "shift_steps": CONTROL_SHIFT_STEPS,
        "shift_hours": CONTROL_SHIFT_STEPS * 4,
        "policy": "station-and-split-local past-only shift",
    }
    manifest["input_steps"] = INPUT_STEPS
    base.save_json(OUTPUT_DIR / "run_manifest.json", manifest)

    results: dict[str, dict[str, dict]] = {}
    arrays_by_mode: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    persistence_arrays: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
        split: {} for split in ("train", "val", "test")
    }
    target_references: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    history_rows = []
    model_rows = []

    for mode_index, mode in enumerate(ALL_MODES, start=1):
        console.phase(
            f"train {TERMINAL_MODE_NAMES[mode]}",
            current=mode_index,
            total=len(ALL_MODES),
        )
        mode_dir = OUTPUT_DIR / "checkpoints" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        dual.OUTPUT_DIR = mode_dir
        input_columns = input_columns_for_mode(mode)
        mode_abs_arrays = {split: {} for split in ("train", "val", "test")}
        mode_delta_arrays = {split: {} for split in ("train", "val", "test")}
        mode_raw_arrays = {split: {} for split in ("train", "val", "test")}
        parameter_count = None

        for target in TARGET_FEATURE_COLUMNS:
            raw_splits, scaled_splits, scalers = window.build_target_splits(
                data,
                stations,
                input_columns,
                target,
                INPUT_STEPS,
                include_current_level=False,
            )
            target_references[target] = _assert_common_targets(target_references.get(target), raw_splits)
            scaled_splits, current_scaler = dual.attach_scaled_current_level(raw_splits, scaled_splits)
            if parameter_count is None:
                model = dual.make_dual_branch_model(
                    torch,
                    sequence_input_dim=scaled_splits["train"]["self_x"].shape[-1],
                    current_input_dim=scaled_splits["train"]["current_level"].shape[-1],
                    target_dim=1,
                    output_steps=OUTPUT_STEPS,
                )
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

            name = f"{mode}_all9_D_{INPUT_STEPS}to{OUTPUT_STEPS}"
            result, arrays_by_split = dual.fit_dual_target_delta_gru(
                torch,
                name,
                target,
                input_columns,
                scaled_splits,
                scalers,
                current_scaler,
                stations,
                device,
            )
            model_rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "raw_input_channels": len(input_columns),
                    "model_input_channels_with_masks": scaled_splits["train"]["self_x"].shape[-1],
                    "parameters": parameter_count,
                    "best_epoch": result["best_epoch"]["epoch"],
                    "best_val_rmse": result["best_epoch"]["val_rmse"],
                    "checkpoint": result["best_model_path"],
                }
            )
            history_rows.extend({"mode": mode, "target": target, **row} for row in result["history"])
            for split, arrays in arrays_by_split.items():
                mode_abs_arrays[split][target] = arrays
                mode_delta_arrays[split][target] = _delta_arrays(arrays, raw_splits[split])
                mode_raw_arrays[split][target] = {
                    "absolute": arrays,
                    "delta": mode_delta_arrays[split][target],
                }
                if target not in persistence_arrays[split]:
                    absolute, change = _persistence_arrays(raw_splits[split])
                    persistence_arrays[split][target] = {"absolute": absolute, "delta": change}

        arrays_by_mode[mode] = mode_raw_arrays
        results[mode] = {"parameters": parameter_count}
        for split in ("train", "val", "test"):
            results[mode][split] = {
                "absolute": _aggregate(mode_abs_arrays[split], stations),
                "delta": _aggregate(mode_delta_arrays[split], stations),
            }

    persistence = {}
    for split in ("train", "val", "test"):
        persistence[split] = {
            "absolute": _aggregate(
                {feature: persistence_arrays[split][feature]["absolute"] for feature in TARGET_FEATURE_COLUMNS},
                stations,
            ),
            "delta": _aggregate(
                {feature: persistence_arrays[split][feature]["delta"] for feature in TARGET_FEATURE_COLUMNS},
                stations,
            ),
        }

    overall = pd.DataFrame(_metric_rows(results, persistence))
    features = pd.DataFrame(_feature_rows(results, persistence))
    station_metrics = pd.DataFrame(_station_rows(results))
    station_feature_metrics = pd.DataFrame(_station_feature_rows(results))
    tail_metrics = pd.DataFrame(_tail_rows(arrays_by_mode, persistence_arrays))
    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig")
    station_metrics.to_csv(OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig")
    station_feature_metrics.to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    tail_metrics.to_csv(OUTPUT_DIR / "tail_change_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_rows).to_csv(OUTPUT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")
    base.save_json(
        OUTPUT_DIR / "metrics.json",
        {
            "manifest": manifest,
            "overall": overall.to_dict(orient="records"),
            "model_summary": model_rows,
        },
    )
    _write_report(overall, features, station_metrics)
    display = overall.copy()
    display["representation"] = display["mode"].map(TERMINAL_MODE_NAMES)
    columns = ("representation", "macro_station_rmse", "rmse", "nse", "skill_vs_persistence_pct")
    console.table(
        "validation summary",
        display[display["split"].eq("val")].sort_values("macro_station_rmse"),
        columns=columns,
    )
    console.table(
        "test summary",
        display[display["split"].eq("test")].sort_values("macro_station_rmse"),
        columns=columns,
    )
    console.done(OUTPUT_DIR, report="run_report.md", details="CSV/JSON files")
    return 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
