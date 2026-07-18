#!/usr/bin/env python3
"""Ablate input windows and current-target-level inputs for diff-delta GRU."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_wentu_diff_delta_feature_selection as feature_select
from scripts.gru import run_wentu_diff_delta_gru as delta
from scripts.gru import run_wentu_physical_lag_gru as lag
from scripts.gru import run_wentu_self_feature_ablation as self_ablation
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS


# [01] 实验配置：窗口从 12h 到 72h；比较 diff-only 与 diff + 当前目标原始值。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_window_level_ablation_2023_1step_chain"
WINDOW_STEPS: tuple[int, ...] = (3, 6, 9, 12, 18)
OUTPUT_STEPS = 1
TARGET_FEATURE_COLUMNS = delta.TARGET_FEATURE_COLUMNS
CORR_TOP_K = feature_select.CORR_TOP_K


def steps_to_hours(input_steps: int) -> int:
    """[02] 4 小时频率下，将步数转换为小时数。"""
    return int(input_steps * 4)


def experiment_name(input_steps: int, include_current_level: bool) -> str:
    """[03] 统一实验命名。"""
    if include_current_level:
        return f"corr_top3_diff_current_level_{input_steps}step_delta"
    return f"corr_top3_diff_{input_steps}step_delta"


def diff_columns_from_features(features: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """[04] 指标名转 diff1 输入列。"""
    return tuple(f"{feature}_diff1" for feature in features)


def add_current_target_level_channel(
    split: dict[str, np.ndarray],
    target_input_index: int | None = None,
) -> dict[str, np.ndarray]:
    """[05] 把当前目标原始值作为额外通道拼到每个历史步后面。

    训练时没有传未来值：这个当前值来自输入窗口最后一步，也就是 t 时刻。
    """
    augmented = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in split.items()}
    self_x = augmented["self_x"]
    self_mask = augmented["self_mask"]
    if target_input_index is None:
        if "last_target" not in augmented:
            raise ValueError("last_target is required when target_input_index is not provided.")
        last_level = augmented["last_target"][..., 0]
        last_mask = np.isfinite(last_level)
    else:
        last_level = self_x[:, -1, :, target_input_index]
        last_mask = self_mask[:, -1, :, target_input_index].astype(bool) & np.isfinite(last_level)

    steps = self_x.shape[1]
    level_values = np.repeat(last_level[:, None, :, None], steps, axis=1)
    level_masks = np.repeat(last_mask[:, None, :, None], steps, axis=1)
    augmented["self_x"] = np.concatenate([self_x, level_values], axis=-1)
    augmented["self_mask"] = np.concatenate([self_mask, level_masks], axis=-1)

    if "upstream_x" in augmented:
        extra_upstream = np.full((*augmented["upstream_x"].shape[:-1], 1), np.nan, dtype=float)
        extra_upstream_mask = np.zeros((*augmented["upstream_mask"].shape[:-1], 1), dtype=bool)
        augmented["upstream_x"] = np.concatenate([augmented["upstream_x"], extra_upstream], axis=-1)
        augmented["upstream_mask"] = np.concatenate([augmented["upstream_mask"], extra_upstream_mask], axis=-1)
    return augmented


def maybe_add_current_level_to_splits(
    splits: dict[str, dict[str, np.ndarray]],
    include_current_level: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """[06] 对 train/val/test 同步添加当前目标原始值通道。"""
    if not include_current_level:
        return splits
    return {name: add_current_target_level_channel(split) for name, split in splits.items()}


def build_target_splits(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_columns: tuple[str, ...],
    target_feature: str,
    input_steps: int,
    include_current_level: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], base.GraphForecastScalers]:
    """[07] 构造单目标 delta 数据，并可选加入当前目标原始值。"""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=input_steps,
        output_steps=OUTPUT_STEPS,
        input_columns=input_columns,
        target_columns=(target_feature,),
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    raw_splits = maybe_add_current_level_to_splits(raw_splits, include_current_level)
    scaled_splits, scalers = lag.scale_physical_lag_splits(raw_splits)
    return raw_splits, scaled_splits, scalers


def select_corr_features_for_target(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    target_feature: str,
    input_steps: int,
    experiment: str,
    include_current_level: bool,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """[08] 只用训练集，为某目标和某窗口选择 corr-top3 diff 特征。"""
    all9_diff_columns = diff_columns_from_features(FEATURE_COLUMNS)
    raw_splits, _, _ = build_target_splits(
        data,
        stations,
        all9_diff_columns,
        target_feature,
        input_steps,
        include_current_level=False,
    )
    lagged, target_delta = feature_select.flatten_lagged_training_signals(raw_splits["train"], all9_diff_columns)
    selected, rows = feature_select.select_corr_topk_features(lagged, target_delta, target_feature, CORR_TOP_K)
    selected_set = set(selected)
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "experiment": experiment,
                "input_steps": input_steps,
                "window_hours": steps_to_hours(input_steps),
                "include_current_level": include_current_level,
                "target": target_feature,
                "feature": row["feature"],
                "score": row["score"],
                "best_lag_position": row["best_lag_position"],
                "selection_reason": row["selection_reason"],
                "selected": row["feature"] in selected_set,
            }
        )
    return selected, output_rows


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[09] DataLoader 复用 physical-lag 数据格式。"""
    return lag.make_loader(torch, split, shuffle)


def collect_target_arrays(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    device,
) -> dict[str, np.ndarray]:
    """[10] 收集单目标绝对值预测。"""
    arrays = delta.collect_abs_prediction_arrays(torch, model, loader, split, scalers, device)
    return {"pred": arrays["pred_abs"], "true": arrays["true_abs"], "mask": arrays["mask"]}


def evaluate_target_model(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    target_feature: str,
    device,
) -> dict:
    """[11] 按还原后的绝对值预测误差评分。"""
    arrays = collect_target_arrays(torch, model, loader, split, scalers, device)
    return base.masked_error_metrics(
        arrays["pred"] - arrays["true"],
        arrays["mask"],
        (target_feature,),
        stations,
        truth=arrays["true"],
    )


def fit_target_delta_gru(
    torch,
    name: str,
    target_feature: str,
    input_columns: tuple[str, ...],
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    input_steps: int,
    include_current_level: bool,
    device,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """[12] 训练单目标 GRU；输出 delta，评分时还原绝对值。"""
    torch.manual_seed(paper.SEED)
    loaders = {
        split_name: make_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = lag.make_model(
        torch,
        self_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        upstream_input_dim=scaled_splits["train"]["upstream_x"].shape[-1],
        target_dim=1,
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
    best_model_path = (
        OUTPUT_DIR
        / f"{feature_select.safe_filename(name)}_{feature_select.safe_filename(target_feature)}_{input_steps}to{OUTPUT_STEPS}_best.pt"
    )
    history = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = lag.train_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = evaluate_target_model(
            torch, model, loaders["val"], scaled_splits["val"], scalers, stations, target_feature, device
        )
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_rmse < best_rmse - paper.MIN_DELTA
        history.append(
            {
                "epoch": epoch,
                "target": target_feature,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "learning_rate": current_lr,
                "improved": improved,
            }
        )
        if improved:
            best_rmse = val_rmse
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": name,
                    "target": target_feature,
                    "architecture": "per_target_corr_top3_diff_delta_gru",
                    "input_columns": list(input_columns),
                    "include_current_level": include_current_level,
                    "input_steps": input_steps,
                    "output_steps": OUTPUT_STEPS,
                    "scalers": scalers.to_dict(),
                },
                best_model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    arrays_by_split = {
        split_name: collect_target_arrays(torch, model, loaders[split_name], scaled_splits[split_name], scalers, device)
        for split_name in ["train", "val", "test"]
    }
    metrics = {
        split_name: evaluate_target_model(
            torch, model, loaders[split_name], scaled_splits[split_name], scalers, stations, target_feature, device
        )
        for split_name in ["train", "val", "test"]
    }
    best_epoch = min(history, key=lambda item: item["val_rmse"])
    console.model_result(
        target_feature,
        best_epoch=best_epoch["epoch"],
        val_rmse=best_epoch["val_rmse"],
        test_rmse=metrics["test"]["rmse"],
    )
    return {
        "experiment": name,
        "target": target_feature,
        "history": history,
        "best_epoch": best_epoch,
        "best_checkpoint": metrics,
        "best_model_path": str(best_model_path),
        "input_columns": list(input_columns),
    }, arrays_by_split


def run_one_experiment(
    torch,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_steps: int,
    include_current_level: bool,
    device,
) -> tuple[dict, list[dict[str, object]]]:
    """[13] 跑一个窗口和一种输入模式的五个 per-target 模型。"""
    name = experiment_name(input_steps, include_current_level)
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []
    for target_feature in TARGET_FEATURE_COLUMNS:
        selected_features, rows = select_corr_features_for_target(
            data, stations, target_feature, input_steps, name, include_current_level
        )
        selected_rows.extend(rows)
        input_columns = diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = build_target_splits(
            data,
            stations,
            input_columns,
            target_feature,
            input_steps,
            include_current_level,
        )
        result, arrays_by_split = fit_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            stations,
            input_steps,
            include_current_level,
            device,
        )
        target_results[target_feature] = result
        history_rows.extend({"sub_experiment": f"{name}_{target_feature}", **row} for row in result["history"])
        for split_name, arrays in arrays_by_split.items():
            arrays_by_split_and_target[split_name][target_feature] = arrays

    metrics = {
        split_name: self_ablation.aggregate_single_target_arrays(arrays_by_target, stations, TARGET_FEATURE_COLUMNS)
        for split_name, arrays_by_target in arrays_by_split_and_target.items()
    }
    best_epochs = {target: result["best_epoch"] for target, result in target_results.items()}
    return {
        "experiment": name,
        "input_steps": input_steps,
        "window_hours": steps_to_hours(input_steps),
        "include_current_level": include_current_level,
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, selected_rows


def persistence_for_window(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_steps: int,
) -> dict:
    """[14] 给每个窗口单独算持久性 baseline，保证有效样本口径一致。"""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=input_steps,
        output_steps=OUTPUT_STEPS,
        input_columns=diff_columns_from_features(FEATURE_COLUMNS),
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    return delta.evaluate_persistence_baseline(raw_splits, stations)["test"]


def overall_rows(results: dict[str, dict], persistence_by_steps: dict[int, dict]) -> list[dict[str, object]]:
    """[15] 整体指标表。"""
    rows = []
    for experiment, result in results.items():
        test = result["best_checkpoint"]["test"]
        rows.append(
            {
                "experiment": experiment,
                "input_steps": result["input_steps"],
                "window_hours": result["window_hours"],
                "include_current_level": result["include_current_level"],
                "val_rmse": result["best_epoch"].get("val_rmse"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    for input_steps, test in persistence_by_steps.items():
        rows.append(
            {
                "experiment": f"persistence_{input_steps}step",
                "input_steps": input_steps,
                "window_hours": steps_to_hours(input_steps),
                "include_current_level": False,
                "val_rmse": "",
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    return rows


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[16] 分指标 test 指标。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]
        for feature in TARGET_FEATURE_COLUMNS:
            rows.append(
                {
                    "experiment": experiment,
                    "input_steps": result["input_steps"],
                    "window_hours": result["window_hours"],
                    "include_current_level": result["include_current_level"],
                    "feature": feature,
                    "valid_points": metrics["feature_valid_points"].get(feature, 0),
                    "test_mae": metrics["feature_mae"].get(feature),
                    "test_rmse": metrics["feature_rmse"].get(feature),
                    "test_nse": metrics["feature_nse"].get(feature),
                }
            )
    return rows


def station_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[17] 分站点 test 指标。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            item = metrics.get(station, {})
            rows.append(
                {
                    "experiment": experiment,
                    "input_steps": result["input_steps"],
                    "window_hours": result["window_hours"],
                    "include_current_level": result["include_current_level"],
                    "station": station,
                    "valid_points": item.get("valid_points", 0),
                    "test_mae": item.get("mae"),
                    "test_rmse": item.get("rmse"),
                    "test_nse": item.get("nse"),
                }
            )
    return rows


def station_feature_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[18] 分站点分指标 test 指标。"""
    rows = []
    for experiment, result in results.items():
        station_metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            item = station_metrics.get(station, {})
            for feature in TARGET_FEATURE_COLUMNS:
                rows.append(
                    {
                        "experiment": experiment,
                        "input_steps": result["input_steps"],
                        "window_hours": result["window_hours"],
                        "include_current_level": result["include_current_level"],
                        "station": station,
                        "feature": feature,
                        "valid_points": item.get("feature_valid_points", {}).get(feature, 0),
                        "test_mae": item.get("feature_mae", {}).get(feature),
                        "test_rmse": item.get("feature_rmse", {}).get(feature),
                        "test_nse": item.get("feature_nse", {}).get(feature),
                    }
                )
    return rows


def save_tables(
    results: dict[str, dict],
    stations: tuple[str, ...],
    persistence_by_steps: dict[int, dict],
    selected_rows: list[dict[str, object]],
) -> None:
    """[19] 保存所有结果表。"""
    pd.DataFrame(overall_rows(results, persistence_by_steps)).to_csv(
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
    pd.DataFrame(selected_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def _json_safe_results(results: dict[str, dict]) -> dict[str, dict]:
    safe = {}
    for name, result in results.items():
        safe[name] = {
            "input_steps": result["input_steps"],
            "window_hours": result["window_hours"],
            "include_current_level": result["include_current_level"],
            "best_epoch": result["best_epoch"],
            "best_checkpoint": result["best_checkpoint"],
            "targets": {
                target: {
                    "best_epoch": target_result["best_epoch"],
                    "best_model_path": target_result["best_model_path"],
                    "input_columns": target_result["input_columns"],
                }
                for target, target_result in result["targets"].items()
            },
        }
    return safe


def write_report(results: dict[str, dict], persistence_by_steps: dict[int, dict]) -> None:
    """[20] 写简短报告。"""
    overall = pd.DataFrame(overall_rows(results, persistence_by_steps))
    lines = [
        "# 时间窗口与当前目标原始值消融",
        "",
        "## 设计",
        "- 窗口：3/6/9/12/18 步，对应 12/24/36/48/72 小时。",
        "- diff-only：每个目标使用训练集 corr-top3 筛出的历史 diff 序列。",
        "- current-level：在 diff-only 基础上额外加入当前目标原始值，作为重复静态通道输入 GRU。",
        "- 输出：未来 1 步目标 delta，最终预测值 = 当前目标值 + 预测 delta。",
        "",
        "## 整体结果",
        "```text",
        overall.sort_values(["test_rmse", "experiment"]).to_string(index=False),
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
    data = delta.load_diff1_chain_data()
    dataset_summary = {
        "model": "per_target_corr_top3_window_and_current_level_ablation",
        "output_dir": str(OUTPUT_DIR),
        "start_date": paper.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": paper.RESAMPLE_RULE,
        "window_steps": list(WINDOW_STEPS),
        "window_hours": [steps_to_hours(steps) for steps in WINDOW_STEPS],
        "output_steps": OUTPUT_STEPS,
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "stations": list(stations),
        "corr_top_k": CORR_TOP_K,
        "experiments": [
            experiment_name(steps, include_current_level)
            for steps in WINDOW_STEPS
            for include_current_level in (False, True)
        ],
    }
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = lag.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    selected_rows = []
    for input_steps in WINDOW_STEPS:
        for include_current_level in (False, True):
            result, rows = run_one_experiment(torch, data, stations, input_steps, include_current_level, device)
            results[result["experiment"]] = result
            selected_rows.extend(rows)

    persistence_by_steps = {input_steps: persistence_for_window(data, stations, input_steps) for input_steps in WINDOW_STEPS}
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence_by_steps,
        "experiments": _json_safe_results(results),
        "selected_features": selected_rows,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence_by_steps, selected_rows)
    write_report(results, persistence_by_steps)
    overall = pd.DataFrame(overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    console.print(overall.to_string(index=False), flush=True)
    console.print(pd.DataFrame(feature_metric_rows(results)).to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
