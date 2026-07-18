#!/usr/bin/env python3
"""Run GRU experiments that predict target deltas from historical feature diffs."""

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


# [01] 实验配置：只看本站，使用历史变化量预测未来目标变化量。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_diff_delta_2023_6to1_chain"
INPUT_STEPS = lag.INPUT_STEPS
OUTPUT_STEPS = lag.OUTPUT_STEPS
TARGET_FEATURE_COLUMNS = lag.TARGET_FEATURE_COLUMNS
DIFF_COLUMNS: tuple[str, ...] = feature_enhancement_columns(("diff1",))


@dataclass(frozen=True)
class ExperimentSpec:
    """[02] 单个 delta 实验的输入列定义。"""

    raw_features: tuple[str, ...]
    diff_features: tuple[str, ...]
    description: str


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "target5_diff_delta": ExperimentSpec(
        raw_features=(),
        diff_features=TARGET_FEATURE_COLUMNS,
        description="5 个目标指标历史 diff1 -> 5 个目标未来 delta。",
    ),
    "all9_diff_delta": ExperimentSpec(
        raw_features=(),
        diff_features=FEATURE_COLUMNS,
        description="9 个水质指标历史 diff1 -> 5 个目标未来 delta。",
    ),
    "all9_raw_diff_delta": ExperimentSpec(
        raw_features=FEATURE_COLUMNS,
        diff_features=FEATURE_COLUMNS,
        description="9 个原始指标 + 9 个历史 diff1 -> 5 个目标未来 delta。",
    ),
}


def input_columns_for_spec(spec: ExperimentSpec) -> tuple[str, ...]:
    """[03] 返回模型输入列；diff 输入统一使用 *_diff1。"""
    return (*spec.raw_features, *(f"{feature}_diff1" for feature in spec.diff_features))


def load_diff1_chain_data() -> pd.DataFrame:
    """[04] 读取三站链条数据，并加入一阶差分特征。"""
    return add_feature_enhancements(lag.load_chain_data(), ("diff1",))


def _empty_delta_dataset(
    input_steps: int,
    output_steps: int,
    station_count: int,
    input_dim: int,
    target_dim: int,
) -> dict[str, np.ndarray]:
    return {
        "self_x": np.empty((0, input_steps, station_count, input_dim), dtype=float),
        "upstream_x": np.empty((0, station_count, input_dim), dtype=float),
        "y": np.empty((0, output_steps, station_count, target_dim), dtype=float),
        "y_abs": np.empty((0, output_steps, station_count, target_dim), dtype=float),
        "last_target": np.empty((0, station_count, target_dim), dtype=float),
        "self_mask": np.empty((0, input_steps, station_count, input_dim), dtype=bool),
        "upstream_mask": np.empty((0, station_count, input_dim), dtype=bool),
        "y_mask": np.empty((0, output_steps, station_count, target_dim), dtype=bool),
        "target_start": np.asarray([], dtype="datetime64[ns]"),
        "target_end": np.asarray([], dtype="datetime64[ns]"),
        "upstream_time": np.empty((0, station_count), dtype="datetime64[ns]"),
    }


def target_delta_from_future_and_last(
    future: np.ndarray,
    last_target: np.ndarray,
    future_ok: np.ndarray | None = None,
    last_ok: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """[05] 构造监督信号：delta = future target - 当前最后一次 target。"""
    future = np.asarray(future, dtype=float)
    last_target = np.asarray(last_target, dtype=float)
    delta = future - last_target[:, None, :, :]
    mask = np.isfinite(future) & np.isfinite(last_target[:, None, :, :])
    if future_ok is not None:
        mask &= np.asarray(future_ok, dtype=bool)
    if last_ok is not None:
        mask &= np.asarray(last_ok, dtype=bool)[:, None, :, :]
    return delta, mask


def restore_absolute_from_delta(delta_prediction: np.ndarray, last_target: np.ndarray) -> np.ndarray:
    """[06] 将预测变化量还原为真实量纲预测值。"""
    return np.asarray(last_target, dtype=float)[:, None, :, :] + np.asarray(delta_prediction, dtype=float)


def build_delta_dataset(
    data: pd.DataFrame,
    stations: tuple[str, ...] | list[str],
    input_steps: int = INPUT_STEPS,
    output_steps: int = OUTPUT_STEPS,
    input_columns: tuple[str, ...] = DIFF_COLUMNS,
    target_columns: tuple[str, ...] = TARGET_FEATURE_COLUMNS,
    freq: str = paper.RESAMPLE_RULE,
) -> dict[str, np.ndarray]:
    """[07] 构造 self-only 窗口，y 为未来变化量，同时保留真实 y_abs。"""
    stations = tuple(stations)
    if not stations:
        return _empty_delta_dataset(input_steps, output_steps, 0, len(input_columns), len(target_columns))

    quality_columns = tuple(
        f"{column}__target_ok" for column in target_columns if f"{column}__target_ok" in data.columns
    )
    required = ["station", "time", *dict.fromkeys([*input_columns, *target_columns, *quality_columns])]
    work = data[required].copy()
    work["station"] = work["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    for column in dict.fromkeys([*input_columns, *target_columns]):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    start = work["time"].min().ceil(freq)
    end = work["time"].max().floor(freq)
    times = pd.date_range(start, end, freq=freq)
    if len(times) < input_steps + output_steps:
        return _empty_delta_dataset(input_steps, output_steps, len(stations), len(input_columns), len(target_columns))

    grouped = work.groupby(["time", "station"], sort=True).mean(numeric_only=True)
    full_index = pd.MultiIndex.from_product([times, stations], names=["time", "station"])
    panel = grouped.reindex(full_index)
    input_values = panel.loc[:, input_columns].to_numpy(float).reshape(len(times), len(stations), len(input_columns))
    target_values = panel.loc[:, target_columns].to_numpy(float).reshape(len(times), len(stations), len(target_columns))
    if len(quality_columns) == len(target_columns):
        target_quality = (
            panel.loc[:, quality_columns]
            .fillna(False)
            .to_numpy(bool)
            .reshape(len(times), len(stations), len(target_columns))
        )
    else:
        target_quality = np.ones_like(target_values, dtype=bool)

    total_steps = input_steps + output_steps
    self_xs, upstream_xs, ys, y_abss, last_targets = [], [], [], [], []
    self_masks, upstream_masks, y_masks = [], [], []
    target_starts, target_ends, upstream_times = [], [], []
    for start_idx in range(len(times) - total_steps + 1):
        target_idx = start_idx + input_steps
        self_x = input_values[start_idx:target_idx]
        future = target_values[target_idx : start_idx + total_steps]
        last_target = target_values[target_idx - 1]
        future_ok = target_quality[target_idx : start_idx + total_steps]
        last_ok = target_quality[target_idx - 1]
        delta, delta_mask = target_delta_from_future_and_last(
            future[None, ...],
            last_target[None, ...],
            future_ok=future_ok[None, ...],
            last_ok=last_ok[None, ...],
        )
        delta = delta[0]
        delta_mask = delta_mask[0]
        if not delta_mask.any():
            continue

        upstream_x = np.full((len(stations), len(input_columns)), np.nan, dtype=float)
        upstream_mask = np.zeros((len(stations), len(input_columns)), dtype=bool)
        upstream_time = np.full((len(stations),), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

        self_xs.append(self_x)
        upstream_xs.append(upstream_x)
        ys.append(delta)
        y_abss.append(future)
        last_targets.append(last_target)
        self_masks.append(np.isfinite(self_x))
        upstream_masks.append(upstream_mask)
        y_masks.append(delta_mask)
        target_starts.append(times[target_idx].to_datetime64())
        target_ends.append(times[start_idx + total_steps - 1].to_datetime64())
        upstream_times.append(upstream_time)

    if not self_xs:
        return _empty_delta_dataset(input_steps, output_steps, len(stations), len(input_columns), len(target_columns))

    return {
        "self_x": np.stack(self_xs),
        "upstream_x": np.stack(upstream_xs),
        "y": np.stack(ys),
        "y_abs": np.stack(y_abss),
        "last_target": np.stack(last_targets),
        "self_mask": np.stack(self_masks),
        "upstream_mask": np.stack(upstream_masks),
        "y_mask": np.stack(y_masks),
        "target_start": np.asarray(target_starts, dtype="datetime64[ns]"),
        "target_end": np.asarray(target_ends, dtype="datetime64[ns]"),
        "upstream_time": np.stack(upstream_times).astype("datetime64[ns]"),
    }


def build_splits(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_columns: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], base.GraphForecastScalers]:
    """[08] 构造 delta 数据集并按时间切分、标准化。"""
    dataset = build_delta_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=input_columns,
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    scaled_splits, scalers = lag.scale_physical_lag_splits(raw_splits)
    return raw_splits, scaled_splits, scalers


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[09] 复用 physical-lag DataLoader；上游输入为空且模型不使用。"""
    return lag.make_loader(torch, split, shuffle)


def collect_abs_prediction_arrays(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    device,
) -> dict[str, np.ndarray]:
    """[10] 收集 delta 预测，并还原为绝对值预测。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for self_x, upstream_x, y, y_mask in loader:
            preds.append(model(self_x.to(device), upstream_x.to(device)).cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())
    if not preds:
        return {
            "pred_abs": split["y_abs"][:0],
            "true_abs": split["y_abs"][:0],
            "pred_delta": split["y"][:0],
            "true_delta": split["y"][:0],
            "mask": split["y_mask"][:0],
        }

    pred_delta = scalers.inverse_transform_target(np.concatenate(preds))
    true_delta = scalers.inverse_transform_target(np.concatenate(trues))
    mask = np.concatenate(masks).astype(bool)
    pred_abs = restore_absolute_from_delta(pred_delta, split["last_target"])
    true_abs = split["y_abs"]
    mask = mask & np.isfinite(pred_abs) & np.isfinite(true_abs)
    return {
        "pred_abs": pred_abs,
        "true_abs": true_abs,
        "pred_delta": pred_delta,
        "true_delta": true_delta,
        "mask": mask,
    }


def evaluate_model(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[11] 在还原后的绝对值上计算 MAE/RMSE/NSE。"""
    arrays = collect_abs_prediction_arrays(torch, model, loader, split, scalers, device)
    return base.masked_error_metrics(
        arrays["pred_abs"] - arrays["true_abs"],
        arrays["mask"],
        TARGET_FEATURE_COLUMNS,
        stations,
        truth=arrays["true_abs"],
    )


def fit_delta_gru(
    torch,
    name: str,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[12] 训练一个 self-only delta GRU。"""
    torch.manual_seed(paper.SEED)
    loaders = {
        split_name: make_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = lag.make_model(
        torch,
        self_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        upstream_input_dim=scaled_splits["train"]["upstream_x"].shape[-1],
        target_dim=len(TARGET_FEATURE_COLUMNS),
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
    best_model_path = OUTPUT_DIR / f"{name}_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    history = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = lag.train_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = evaluate_model(torch, model, loaders["val"], scaled_splits["val"], scalers, stations, device)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": name,
                    "architecture": "self_gru_predict_delta_then_restore_absolute",
                    "target_columns": list(TARGET_FEATURE_COLUMNS),
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "scalers": scalers.to_dict(),
                },
                best_model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                console.print(f"{name} early_stop epoch={epoch:03d} best_val_rmse={best_rmse:.6f}", flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {
        split_name: evaluate_model(torch, model, loaders[split_name], scaled_splits[split_name], scalers, stations, device)
        for split_name in ["train", "val", "test"]
    }
    return {
        "experiment": name,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["val_rmse"]),
        "best_checkpoint": metrics,
        "best_model_path": str(best_model_path),
    }


def evaluate_persistence_baseline(raw_splits: dict[str, dict[str, np.ndarray]], stations: tuple[str, ...]) -> dict[str, dict]:
    """[13] 持久性 baseline：未来绝对值等于当前最后一次观测值。"""
    metrics = {}
    for split_name, split in raw_splits.items():
        pred = np.repeat(split["last_target"][:, None, :, :], split["y_abs"].shape[1], axis=1)
        valid = split["y_mask"] & np.isfinite(pred) & np.isfinite(split["y_abs"])
        metrics[split_name] = base.masked_error_metrics(
            pred - split["y_abs"],
            valid,
            TARGET_FEATURE_COLUMNS,
            stations,
            truth=split["y_abs"],
        )
    return metrics


def overall_rows(results: dict[str, dict], persistence: dict[str, dict]) -> list[dict[str, object]]:
    """[14] 汇总整体 test 指标。"""
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
    """[15] 汇总逐目标 test 指标。"""
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
    """[16] 汇总逐站点整体 test 指标。"""
    rows = []
    for experiment, result in results.items():
        for station in stations:
            item = result["best_checkpoint"]["test"]["station_metrics"].get(station, {})
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
    """[17] 汇总逐站点逐目标 test 指标。"""
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
    """[18] 保存所有结果表。"""
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
    """[19] 写简短报告。"""
    overall = pd.DataFrame(overall_rows(results, persistence))
    lines = [
        "# diff-delta GRU 消融",
        "",
        "## 实验",
        "- target5_diff_delta：5 个目标指标历史 diff1 预测 5 个目标未来 delta。",
        "- all9_diff_delta：9 个水质指标历史 diff1 预测 5 个目标未来 delta。",
        "- all9_raw_diff_delta：9 个原始指标 + 9 个 diff1 预测 5 个目标未来 delta。",
        "",
        "## 还原方式",
        "预测真实值 = 当前最后一次目标观测值 + 模型预测 delta。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR) -> int:
    """[20] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    random.seed(paper.SEED)
    np.random.seed(paper.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = tuple(station for station, _ in lag.STATION_FILES)
    data = load_diff1_chain_data()
    reference_columns = input_columns_for_spec(EXPERIMENT_SPECS["all9_raw_diff_delta"])
    reference_raw_splits, _, _ = build_splits(data, stations, reference_columns)
    dataset_summary = {
        "model": "same_station_diff_delta_gru",
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
                "description": spec.description,
                "input_columns": list(input_columns_for_spec(spec)),
                "target": "future_target_delta = future_target - last_observed_target",
            }
            for name, spec in EXPERIMENT_SPECS.items()
        },
        "split_summary": lag.graph_split_summary(reference_raw_splits),
    }
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = lag.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    for name, spec in EXPERIMENT_SPECS.items():
        input_columns = input_columns_for_spec(spec)
        raw_splits, scaled_splits, scalers = build_splits(data, stations, input_columns)
        results[name] = fit_delta_gru(torch, name, scaled_splits, scalers, stations, device)
        if name == "all9_raw_diff_delta":
            reference_raw_splits = raw_splits

    persistence = evaluate_persistence_baseline(reference_raw_splits, stations)
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
