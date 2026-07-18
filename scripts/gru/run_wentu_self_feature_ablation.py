#!/usr/bin/env python3
"""Ablate whether same-station feature coupling helps GRU forecasts."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_wentu_physical_lag_gru as lag
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS, add_feature_enhancements, feature_enhancement_columns


# [01] 实验配置：只看本站历史，逐步加入目标间耦合和额外指标差分。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_self_feature_ablation_2023_6to1_chain"
INPUT_STEPS = lag.INPUT_STEPS
OUTPUT_STEPS = lag.OUTPUT_STEPS
TARGET_FEATURE_COLUMNS = lag.TARGET_FEATURE_COLUMNS
ALL9_DIFF_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, *feature_enhancement_columns(("diff1",)))


@dataclass(frozen=True)
class ExperimentSpec:
    """[02] 单个消融实验的输入特征定义。"""

    per_target: bool
    raw_features: tuple[str, ...]
    include_diff: bool
    description: str


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "target_self_raw": ExperimentSpec(
        per_target=True,
        raw_features=(),
        include_diff=False,
        description="每个目标只看自己的原始历史。",
    ),
    "target_self_raw_diff": ExperimentSpec(
        per_target=True,
        raw_features=(),
        include_diff=True,
        description="每个目标只看自己的原始历史和自身 diff1。",
    ),
    "target5_raw": ExperimentSpec(
        per_target=False,
        raw_features=TARGET_FEATURE_COLUMNS,
        include_diff=False,
        description="5 个预测目标的原始历史共同输入。",
    ),
    "target5_raw_diff": ExperimentSpec(
        per_target=False,
        raw_features=TARGET_FEATURE_COLUMNS,
        include_diff=True,
        description="5 个预测目标的原始历史和 diff1 共同输入。",
    ),
    "all9_raw_diff": ExperimentSpec(
        per_target=False,
        raw_features=FEATURE_COLUMNS,
        include_diff=True,
        description="9 个水质指标的原始历史和 diff1 共同输入。",
    ),
}


def load_diff1_chain_data() -> pd.DataFrame:
    """[03] 读取三站链条数据，并加入一阶差分特征。"""
    return add_feature_enhancements(lag.load_chain_data(), ("diff1",))


def input_columns_for_spec(spec: ExperimentSpec, target_feature: str | None = None) -> tuple[str, ...]:
    """[04] 根据实验定义返回输入列；per-target 只允许目标自身历史。"""
    if spec.per_target:
        if target_feature is None:
            raise ValueError("target_feature is required for per-target experiments.")
        raw_features = (target_feature,)
    else:
        raw_features = spec.raw_features
    if not spec.include_diff:
        return tuple(raw_features)
    return (*raw_features, *(f"{feature}_diff1" for feature in raw_features))


def build_splits(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], base.GraphForecastScalers]:
    """[05] 构造窗口并标准化；上游张量保留但模型不使用。"""
    dataset = lag.build_physical_lag_dataset(
        data,
        stations=stations,
        edges=lag.PHYSICAL_EDGES,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=input_columns,
        target_columns=target_columns,
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    scaled_splits, scalers = lag.scale_physical_lag_splits(raw_splits)
    return raw_splits, scaled_splits, scalers


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[06] DataLoader：沿用 physical-lag 格式，但 use_upstream=False。"""
    return lag.make_loader(torch, split, shuffle)


def evaluate_model(
    torch,
    model,
    loader,
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    target_columns: tuple[str, ...],
    device,
) -> dict:
    """[07] 在原始量纲上评估可变目标维度模型。"""
    arrays = collect_prediction_arrays(torch, model, loader, scalers, device)
    return base.masked_error_metrics(
        arrays["pred"] - arrays["true"],
        arrays["mask"],
        target_columns,
        stations,
        truth=arrays["true"],
    )


def collect_prediction_arrays(torch, model, loader, scalers: base.GraphForecastScalers, device) -> dict[str, np.ndarray]:
    """[08] 收集某个 split 的预测、真值和 mask。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for self_x, upstream_x, y, y_mask in loader:
            preds.append(model(self_x.to(device), upstream_x.to(device)).cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())
    pred = scalers.inverse_transform_target(np.concatenate(preds))
    true = scalers.inverse_transform_target(np.concatenate(trues))
    mask = np.concatenate(masks).astype(bool)
    return {"pred": pred, "true": true, "mask": mask}


def safe_filename(value: str) -> str:
    """[09] 把中文指标名安全转成 checkpoint 文件名片段。"""
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def fit_self_gru(
    torch,
    name: str,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    target_columns: tuple[str, ...],
    device,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """[10] 训练一个 self-only GRU，并返回指标与预测数组。"""
    torch.manual_seed(paper.SEED)
    loaders = {
        split_name: make_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = lag.make_model(
        torch,
        self_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        upstream_input_dim=scaled_splits["train"]["upstream_x"].shape[-1],
        target_dim=len(target_columns),
        output_steps=OUTPUT_STEPS,
        use_upstream=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=paper.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=paper.LR_DECAY_FACTOR,
        patience=paper.LR_DECAY_PATIENCE,
    )
    loss_fn = paper.make_loss_fn(torch)
    best_rmse = float("inf")
    bad_epochs = 0
    best_model_path = OUTPUT_DIR / f"{safe_filename(name)}_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    history = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = lag.train_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = evaluate_model(torch, model, loaders["val"], scalers, stations, target_columns, device)
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_rmse < best_rmse - paper.MIN_DELTA
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "learning_rate": current_lr,
                "improved": improved,
            }
        )
        console.print(
            f"{name} epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_rmse={val_rmse:.6f} lr={current_lr:.6g}",
            flush=True,
        )
        if improved:
            best_rmse = val_rmse
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "target_columns": list(target_columns)}, best_model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                console.print(f"{name} early_stop epoch={epoch:03d} best_val_rmse={best_rmse:.6f}", flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    arrays_by_split = {
        split_name: collect_prediction_arrays(torch, model, loaders[split_name], scalers, device)
        for split_name in ["train", "val", "test"]
    }
    metrics = {
        split_name: base.masked_error_metrics(
            arrays["pred"] - arrays["true"],
            arrays["mask"],
            target_columns,
            stations,
            truth=arrays["true"],
        )
        for split_name, arrays in arrays_by_split.items()
    }
    return {
        "experiment": name,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["val_rmse"]),
        "best_checkpoint": metrics,
        "best_model_path": str(best_model_path),
    }, arrays_by_split


def _safe_mean(values: np.ndarray) -> float | None:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if valid.size else None


def _safe_rmse(values: np.ndarray) -> float | None:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(np.sqrt(np.mean(valid**2))) if valid.size else None


def _safe_nse(error: np.ndarray, truth: np.ndarray) -> float | None:
    error = np.asarray(error, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(error) & np.isfinite(truth)
    if not valid.any():
        return None
    denominator = float(np.sum((truth[valid] - np.mean(truth[valid])) ** 2))
    if denominator <= 0:
        return None
    return float(1.0 - np.sum(error[valid] ** 2) / denominator)


def aggregate_single_target_arrays(
    arrays_by_target: dict[str, dict[str, np.ndarray]],
    stations: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> dict:
    """[11] 聚合 per-target 模型结果，得到与多输出模型同口径的指标。"""
    all_errors, all_truths = [], []
    metrics = {
        "windows": int(sum(arrays["pred"].shape[0] for arrays in arrays_by_target.values())),
        "valid_points": 0,
        "mae": None,
        "rmse": None,
        "nse": None,
        "feature_valid_points": {},
        "feature_mae": {},
        "feature_rmse": {},
        "feature_nse": {},
        "station_metrics": {},
    }

    for feature in target_columns:
        arrays = arrays_by_target[feature]
        error = arrays["pred"] - arrays["true"]
        mask = arrays["mask"].astype(bool) & np.isfinite(error) & np.isfinite(arrays["true"])
        feature_error = error[mask]
        feature_truth = arrays["true"][mask]
        all_errors.append(feature_error)
        all_truths.append(feature_truth)
        metrics["feature_valid_points"][feature] = int(mask.sum())
        metrics["feature_mae"][feature] = _safe_mean(np.abs(feature_error))
        metrics["feature_rmse"][feature] = _safe_rmse(feature_error)
        metrics["feature_nse"][feature] = _safe_nse(feature_error, feature_truth)

    flat_error = np.concatenate(all_errors) if all_errors else np.asarray([])
    flat_truth = np.concatenate(all_truths) if all_truths else np.asarray([])
    metrics["valid_points"] = int(flat_error.size)
    metrics["mae"] = _safe_mean(np.abs(flat_error))
    metrics["rmse"] = _safe_rmse(flat_error)
    metrics["nse"] = _safe_nse(flat_error, flat_truth)

    for station_idx, station in enumerate(stations):
        station_errors, station_truths = [], []
        station_item = {
            "valid_points": 0,
            "mae": None,
            "rmse": None,
            "nse": None,
            "feature_valid_points": {},
            "feature_mae": {},
            "feature_rmse": {},
            "feature_nse": {},
        }
        for feature in target_columns:
            arrays = arrays_by_target[feature]
            error = arrays["pred"][:, :, station_idx, :] - arrays["true"][:, :, station_idx, :]
            truth = arrays["true"][:, :, station_idx, :]
            mask = arrays["mask"][:, :, station_idx, :].astype(bool) & np.isfinite(error) & np.isfinite(truth)
            item_error = error[mask]
            item_truth = truth[mask]
            station_errors.append(item_error)
            station_truths.append(item_truth)
            station_item["feature_valid_points"][feature] = int(mask.sum())
            station_item["feature_mae"][feature] = _safe_mean(np.abs(item_error))
            station_item["feature_rmse"][feature] = _safe_rmse(item_error)
            station_item["feature_nse"][feature] = _safe_nse(item_error, item_truth)
        station_error = np.concatenate(station_errors) if station_errors else np.asarray([])
        station_truth = np.concatenate(station_truths) if station_truths else np.asarray([])
        station_item["valid_points"] = int(station_error.size)
        station_item["mae"] = _safe_mean(np.abs(station_error))
        station_item["rmse"] = _safe_rmse(station_error)
        station_item["nse"] = _safe_nse(station_error, station_truth)
        metrics["station_metrics"][station] = station_item
    return metrics


def run_per_target_experiment(
    torch,
    name: str,
    spec: ExperimentSpec,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[12] 每个目标单独训练一个只看自身历史的小 GRU。"""
    target_results = {}
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    histories = []
    for target in TARGET_FEATURE_COLUMNS:
        input_columns = input_columns_for_spec(spec, target_feature=target)
        _, scaled_splits, scalers = build_splits(data, stations, input_columns, (target,))
        result, arrays_by_split = fit_self_gru(
            torch,
            f"{name}_{target}",
            scaled_splits,
            scalers,
            stations,
            (target,),
            device,
        )
        target_results[target] = result
        for split_name, arrays in arrays_by_split.items():
            arrays_by_split_and_target[split_name][target] = arrays
        for row in result["history"]:
            histories.append({"sub_experiment": f"{name}_{target}", **row})

    metrics = {
        split_name: aggregate_single_target_arrays(arrays_by_target, stations, TARGET_FEATURE_COLUMNS)
        for split_name, arrays_by_target in arrays_by_split_and_target.items()
    }
    best_epochs = {target: result["best_epoch"] for target, result in target_results.items()}
    return {
        "experiment": name,
        "history": histories,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }


def run_multi_output_experiment(
    torch,
    name: str,
    spec: ExperimentSpec,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[13] 多目标共同输入、共同输出的 self-GRU。"""
    input_columns = input_columns_for_spec(spec)
    _, scaled_splits, scalers = build_splits(data, stations, input_columns, TARGET_FEATURE_COLUMNS)
    result, _ = fit_self_gru(torch, name, scaled_splits, scalers, stations, TARGET_FEATURE_COLUMNS, device)
    return result


def evaluate_persistence_baseline(raw_splits: dict[str, dict[str, np.ndarray]], stations: tuple[str, ...]) -> dict[str, dict]:
    """[14] 持久性 baseline：未来一步等于本站输入窗口最后一个目标观测。"""
    indices = [ALL9_DIFF_COLUMNS.index(feature) for feature in TARGET_FEATURE_COLUMNS]
    metrics = {}
    for split_name, split in raw_splits.items():
        pred = np.repeat(np.take(split["self_x"][:, -1:, :, :], indices, axis=-1), split["y"].shape[1], axis=1)
        valid = split["y_mask"] & np.isfinite(pred)
        metrics[split_name] = base.masked_error_metrics(
            pred - split["y"],
            valid,
            TARGET_FEATURE_COLUMNS,
            stations,
            truth=split["y"],
        )
    return metrics


def overall_rows(results: dict[str, dict], persistence: dict[str, dict]) -> list[dict[str, object]]:
    """[15] 整体 test 指标。"""
    rows = []
    for experiment, result in results.items():
        test = result["best_checkpoint"]["test"]
        best_epoch = result.get("best_epoch", {})
        rows.append(
            {
                "experiment": experiment,
                "best_epoch": best_epoch.get("epoch", ""),
                "val_rmse": best_epoch.get("val_rmse", ""),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    test = persistence["test"]
    rows.append(
        {
            "experiment": "persistence",
            "best_epoch": "",
            "val_rmse": "",
            "test_mae": test.get("mae"),
            "test_rmse": test.get("rmse"),
            "test_nse": test.get("nse"),
            "valid_points": test.get("valid_points"),
        }
    )
    return rows


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[16] 逐目标指标结果。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]
        for feature in TARGET_FEATURE_COLUMNS:
            rows.append(
                {
                    "experiment": experiment,
                    "feature": feature,
                    "valid_points": metrics["feature_valid_points"].get(feature, 0),
                    "test_mae": metrics["feature_mae"].get(feature),
                    "test_rmse": metrics["feature_rmse"].get(feature),
                    "test_nse": metrics["feature_nse"].get(feature),
                }
            )
    return rows


def station_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[17] 逐站点整体指标。"""
    rows = []
    for experiment, result in results.items():
        station_metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            item = station_metrics.get(station, {})
            rows.append(
                {
                    "experiment": experiment,
                    "station": station,
                    "valid_points": item.get("valid_points", 0),
                    "test_mae": item.get("mae"),
                    "test_rmse": item.get("rmse"),
                    "test_nse": item.get("nse"),
                }
            )
    return rows


def station_feature_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[18] 逐站点逐目标指标。"""
    rows = []
    for experiment, result in results.items():
        station_metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            item = station_metrics.get(station, {})
            for feature in TARGET_FEATURE_COLUMNS:
                rows.append(
                    {
                        "experiment": experiment,
                        "station": station,
                        "feature": feature,
                        "valid_points": item.get("feature_valid_points", {}).get(feature, 0),
                        "test_mae": item.get("feature_mae", {}).get(feature),
                        "test_rmse": item.get("feature_rmse", {}).get(feature),
                        "test_nse": item.get("feature_nse", {}).get(feature),
                    }
                )
    return rows


def save_tables(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> None:
    """[19] 保存结果表。"""
    pd.DataFrame(overall_rows(results, persistence)).to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(feature_metric_rows(results)).to_csv(
        OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(station_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(station_feature_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def write_report(results: dict[str, dict], persistence: dict[str, dict]) -> None:
    """[20] 写简短报告。"""
    overall = pd.DataFrame(overall_rows(results, persistence))
    lines = [
        "# 单站点特征耦合消融",
        "",
        "## 实验",
        "- target_self_raw：每个目标只看自己的原始历史。",
        "- target_self_raw_diff：每个目标只看自己的原始历史和自身 diff1。",
        "- target5_raw：5 个预测目标原始历史共同输入。",
        "- target5_raw_diff：5 个预测目标原始历史和 diff1 共同输入。",
        "- all9_raw_diff：9 个水质指标原始历史和 diff1 共同输入。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR) -> int:
    """[21] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    random.seed(paper.SEED)
    np.random.seed(paper.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = tuple(station for station, _ in lag.STATION_FILES)
    data = load_diff1_chain_data()
    all9_raw_splits, _, _ = build_splits(data, stations, ALL9_DIFF_COLUMNS, TARGET_FEATURE_COLUMNS)
    dataset_summary = {
        "model": "same_station_feature_coupling_ablation",
        "output_dir": str(OUTPUT_DIR),
        "start_date": paper.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": paper.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "stations": list(stations),
        "experiments": {
            name: {
                "per_target": spec.per_target,
                "description": spec.description,
                "input_columns": list(input_columns_for_spec(spec, TARGET_FEATURE_COLUMNS[0]) if spec.per_target else input_columns_for_spec(spec)),
            }
            for name, spec in EXPERIMENT_SPECS.items()
        },
        "split_summary": lag.graph_split_summary(all9_raw_splits),
    }
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = lag.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    for name, spec in EXPERIMENT_SPECS.items():
        if spec.per_target:
            results[name] = run_per_target_experiment(torch, name, spec, data, stations, device)
        else:
            results[name] = run_multi_output_experiment(torch, name, spec, data, stations, device)

    persistence = evaluate_persistence_baseline(all9_raw_splits, stations)
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence,
        "experiments": results,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence)
    write_report(results, persistence)
    console.print(pd.DataFrame(overall_rows(results, persistence)).to_string(index=False), flush=True)
    console.print(pd.DataFrame(feature_metric_rows(results)).to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
