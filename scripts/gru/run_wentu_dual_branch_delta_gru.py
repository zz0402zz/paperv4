#!/usr/bin/env python3
"""Compare dual-branch current-level MLP against 36h diff-delta baselines."""

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
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS, StandardScaler


# [01] 三站点小实验：只先比较 36h 输入窗口下 A/B/C/D 四种当前状态使用方式。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_dual_branch_2023_9to1_chain"
INPUT_STEPS = 9
OUTPUT_STEPS = 1
CURRENT_HIDDEN_SIZE = 16
TARGET_FEATURE_COLUMNS = window.TARGET_FEATURE_COLUMNS


def steps_to_hours(input_steps: int) -> int:
    """[02] 4 小时粒度下把步数换成小时数。"""
    return int(input_steps * 4)


def experiment_names() -> tuple[str, ...]:
    """[03] 本轮同口径比较的四个实验。"""
    return (
        window.experiment_name(INPUT_STEPS, False),
        window.experiment_name(INPUT_STEPS, True),
        step_level_experiment_name(),
        dual_branch_experiment_name(),
    )


def step_level_experiment_name() -> str:
    """[04] C 方案：每个历史步输入该步 raw level + diff1。"""
    return f"corr_top3_step_level_{INPUT_STEPS}step_delta"


def dual_branch_experiment_name() -> str:
    """[05] D 方案：diff 序列走 GRU，当前目标值走小 MLP。"""
    return f"corr_top3_dual_branch_current_mlp_{INPUT_STEPS}step_delta"


def raw_diff_columns_from_features(features: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """[06] C 方案输入列：每个选中特征的 raw level 和同一步 diff1。"""
    features = tuple(features)
    return (*features, *(f"{feature}_diff1" for feature in features))


def _stabilize_scaler(scaler: StandardScaler) -> StandardScaler:
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("Current-level scaler was not fitted.")
    scaler.mean_[~np.isfinite(scaler.mean_)] = 0.0
    scaler.scale_[~np.isfinite(scaler.scale_) | (scaler.scale_ == 0)] = 1.0
    return scaler


def attach_scaled_current_level(
    raw_splits: dict[str, dict[str, np.ndarray]],
    scaled_splits: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], StandardScaler]:
    """[07] D 方案当前值分支：当前目标值单独标准化，并附带观测 mask。"""
    current_scaler = _stabilize_scaler(StandardScaler().fit(raw_splits["train"]["last_target"]))
    attached = {}
    for split_name, split in scaled_splits.items():
        current_raw = raw_splits[split_name]["last_target"]
        current_mask = np.isfinite(current_raw)
        current_scaled = current_scaler.transform(current_raw)
        current_input = np.concatenate(
            [
                np.nan_to_num(current_scaled, nan=0.0, posinf=0.0, neginf=0.0),
                current_mask.astype(float),
            ],
            axis=-1,
        ).astype(np.float32)
        attached[split_name] = {**split, "current_level": current_input}
    return attached, current_scaler


def make_dual_branch_model(
    torch,
    sequence_input_dim: int,
    current_input_dim: int,
    target_dim: int,
    output_steps: int = OUTPUT_STEPS,
    hidden_size: int = paper.HIDDEN_SIZE,
    current_hidden_size: int = CURRENT_HIDDEN_SIZE,
):
    """[08] D 模型：GRU 编码历史 diff，MLP 编码当前目标原始水平。"""

    class DualBranchDeltaGru(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = paper.GRU_DROPOUT if paper.NUM_GRU_LAYERS > 1 else 0.0
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_input_dim,
                hidden_size=hidden_size,
                num_layers=paper.NUM_GRU_LAYERS,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(current_input_dim, current_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(current_hidden_size, hidden_size),
                torch.nn.ReLU(),
            )
            self.dropout = torch.nn.Dropout(paper.HEAD_DROPOUT)
            self.head = torch.nn.Linear(hidden_size * 2, output_steps * target_dim)

        def forward(self, sequence_x, current_level):
            batch_size, steps, node_count, _ = sequence_x.shape
            encoded_input = sequence_x.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, -1)
            encoded, _ = self.sequence_encoder(encoded_input)
            sequence_state = encoded[:, -1, :].reshape(batch_size, node_count, hidden_size)
            current_state = self.current_encoder(current_level)
            state = torch.cat([sequence_state, current_state], dim=-1)
            prediction = self.head(self.dropout(state)).reshape(batch_size, node_count, output_steps, target_dim)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return DualBranchDeltaGru()


def make_dual_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[09] D 方案 DataLoader：序列、当前值、目标 delta、mask。"""
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(split["self_x"], dtype=torch.float32),
        torch.as_tensor(split["current_level"], dtype=torch.float32),
        torch.as_tensor(split["y"], dtype=torch.float32),
        torch.as_tensor(split["y_mask"], dtype=torch.bool),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=paper.BATCH_SIZE, shuffle=shuffle)


def train_dual_epoch(torch, model, loader, optimizer, loss_fn, device) -> float:
    """[10] 训练 D 模型一个 epoch。"""
    model.train()
    losses = []
    for sequence_x, current_level, y, y_mask in loader:
        sequence_x = sequence_x.to(device)
        current_level = current_level.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(sequence_x, current_level)
        loss = base.masked_loss(torch, prediction, y, y_mask, loss_fn)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def collect_dual_target_arrays(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    device,
) -> dict[str, np.ndarray]:
    """[11] 收集 D 模型预测，并把 delta 还原成绝对值。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for sequence_x, current_level, y, y_mask in loader:
            preds.append(model(sequence_x.to(device), current_level.to(device)).cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())
    if not preds:
        return {
            "pred": split["y_abs"][:0],
            "true": split["y_abs"][:0],
            "mask": split["y_mask"][:0],
        }

    pred_delta = scalers.inverse_transform_target(np.concatenate(preds))
    true_delta = scalers.inverse_transform_target(np.concatenate(trues))
    mask = np.concatenate(masks).astype(bool)
    pred_abs = delta.restore_absolute_from_delta(pred_delta, split["last_target"])
    true_abs = split["y_abs"]
    mask = mask & np.isfinite(pred_abs) & np.isfinite(true_abs) & np.isfinite(true_delta)
    return {"pred": pred_abs, "true": true_abs, "mask": mask}


def evaluate_dual_target_model(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    target_feature: str,
    device,
) -> dict:
    """[12] D 模型按还原后的绝对值预测误差评分。"""
    arrays = collect_dual_target_arrays(torch, model, loader, split, scalers, device)
    return base.masked_error_metrics(
        arrays["pred"] - arrays["true"],
        arrays["mask"],
        (target_feature,),
        stations,
        truth=arrays["true"],
    )


def fit_dual_target_delta_gru(
    torch,
    name: str,
    target_feature: str,
    input_columns: tuple[str, ...],
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    current_scaler: StandardScaler,
    stations: tuple[str, ...],
    device,
    evaluation_splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """[13] 训练一个 D 方案单目标模型。"""
    if "val" not in evaluation_splits:
        raise ValueError("evaluation_splits must include val")
    unknown_splits = set(evaluation_splits) - set(scaled_splits)
    if unknown_splits:
        raise ValueError(f"Unknown evaluation splits: {sorted(unknown_splits)}")
    torch.manual_seed(paper.SEED)
    loaders = {
        split_name: make_dual_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = make_dual_branch_model(
        torch,
        sequence_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        current_input_dim=scaled_splits["train"]["current_level"].shape[-1],
        target_dim=1,
        output_steps=OUTPUT_STEPS,
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
        / f"{feature_select.safe_filename(name)}_{feature_select.safe_filename(target_feature)}_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    )
    history = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = train_dual_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = evaluate_dual_target_model(
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
                    "architecture": "dual_branch_diff_gru_current_level_mlp_delta",
                    "input_columns": list(input_columns),
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "current_scaler": current_scaler.to_dict(),
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
        split_name: collect_dual_target_arrays(torch, model, loaders[split_name], scaled_splits[split_name], scalers, device)
        for split_name in evaluation_splits
    }
    metrics = {
        split_name: evaluate_dual_target_model(
            torch, model, loaders[split_name], scaled_splits[split_name], scalers, stations, target_feature, device
        )
        for split_name in evaluation_splits
    }
    best_epoch = min(history, key=lambda item: item["val_rmse"])
    if "test" in metrics:
        console.model_result(
            target_feature,
            best_epoch=best_epoch["epoch"],
            val_rmse=best_epoch["val_rmse"],
            test_rmse=metrics["test"]["rmse"],
        )
    else:
        console.info(
            target_feature,
            epoch=best_epoch["epoch"],
            val_rmse=best_epoch["val_rmse"],
            test="sealed",
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


def run_step_level_experiment(torch, data: pd.DataFrame, stations: tuple[str, ...], device) -> tuple[dict, list[dict[str, object]]]:
    """[14] 跑 C 方案：每个历史步 raw+diff，经同一个 GRU 编码。"""
    name = step_level_experiment_name()
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []
    for target_feature in TARGET_FEATURE_COLUMNS:
        selected_features, rows = window.select_corr_features_for_target(
            data, stations, target_feature, INPUT_STEPS, name, include_current_level=False
        )
        for row in rows:
            item = dict(row)
            item["input_mode"] = "step_raw_plus_diff"
            selected_rows.append(item)
        input_columns = raw_diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = window.build_target_splits(
            data, stations, input_columns, target_feature, INPUT_STEPS, include_current_level=False
        )
        result, arrays_by_split = window.fit_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            stations,
            INPUT_STEPS,
            include_current_level=False,
            device=device,
        )
        result["input_mode"] = "step_raw_plus_diff"
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
        "input_steps": INPUT_STEPS,
        "window_hours": steps_to_hours(INPUT_STEPS),
        "include_current_level": False,
        "input_mode": "step_raw_plus_diff",
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, selected_rows


def run_dual_branch_experiment(torch, data: pd.DataFrame, stations: tuple[str, ...], device) -> tuple[dict, list[dict[str, object]]]:
    """[15] 跑 D 方案：GRU 学 top3 diff 序列，MLP 学当前目标原始值。"""
    name = dual_branch_experiment_name()
    target_results = {}
    selected_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []
    for target_feature in TARGET_FEATURE_COLUMNS:
        selected_features, rows = window.select_corr_features_for_target(
            data, stations, target_feature, INPUT_STEPS, name, include_current_level=False
        )
        for row in rows:
            item = dict(row)
            item["input_mode"] = "dual_branch_current_mlp"
            selected_rows.append(item)
        input_columns = window.diff_columns_from_features(selected_features)
        raw_splits, scaled_splits, scalers = window.build_target_splits(
            data, stations, input_columns, target_feature, INPUT_STEPS, include_current_level=False
        )
        scaled_splits, current_scaler = attach_scaled_current_level(raw_splits, scaled_splits)
        result, arrays_by_split = fit_dual_target_delta_gru(
            torch,
            name,
            target_feature,
            input_columns,
            scaled_splits,
            scalers,
            current_scaler,
            stations,
            device,
        )
        result["input_mode"] = "dual_branch_current_mlp"
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
        "input_steps": INPUT_STEPS,
        "window_hours": steps_to_hours(INPUT_STEPS),
        "include_current_level": False,
        "input_mode": "dual_branch_current_mlp",
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, selected_rows


def _set_result_mode(result: dict, input_mode: str) -> dict:
    result["input_mode"] = input_mode
    for target_result in result.get("targets", {}).values():
        target_result["input_mode"] = input_mode
    return result


def overall_rows(results: dict[str, dict], persistence_by_steps: dict[int, dict]) -> list[dict[str, object]]:
    """[16] 整体 test 指标。"""
    rows = []
    for experiment, result in results.items():
        test = result["best_checkpoint"]["test"]
        rows.append(
            {
                "experiment": experiment,
                "input_steps": result["input_steps"],
                "window_hours": result["window_hours"],
                "input_mode": result.get("input_mode", ""),
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
                "input_mode": "persistence",
                "val_rmse": "",
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_nse": test.get("nse"),
                "valid_points": test.get("valid_points"),
            }
        )
    return rows


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[17] 分指标 test 指标。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]
        for feature in TARGET_FEATURE_COLUMNS:
            rows.append(
                {
                    "experiment": experiment,
                    "input_steps": result["input_steps"],
                    "window_hours": result["window_hours"],
                    "input_mode": result.get("input_mode", ""),
                    "feature": feature,
                    "valid_points": metrics["feature_valid_points"].get(feature, 0),
                    "test_mae": metrics["feature_mae"].get(feature),
                    "test_rmse": metrics["feature_rmse"].get(feature),
                    "test_nse": metrics["feature_nse"].get(feature),
                }
            )
    return rows


def station_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[18] 分站点整体 test 指标。"""
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
                    "input_mode": result.get("input_mode", ""),
                    "station": station,
                    "valid_points": item.get("valid_points", 0),
                    "test_mae": item.get("mae"),
                    "test_rmse": item.get("rmse"),
                    "test_nse": item.get("nse"),
                }
            )
    return rows


def station_feature_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[19] 分站点分指标 test 指标。"""
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
                        "input_mode": result.get("input_mode", ""),
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
    """[20] 保存结果表。"""
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
            "input_mode": result.get("input_mode", ""),
            "best_epoch": result["best_epoch"],
            "best_checkpoint": result["best_checkpoint"],
            "targets": {
                target: {
                    "best_epoch": target_result["best_epoch"],
                    "best_model_path": target_result["best_model_path"],
                    "input_columns": target_result["input_columns"],
                    "input_mode": target_result.get("input_mode", result.get("input_mode", "")),
                }
                for target, target_result in result["targets"].items()
            },
        }
    return safe


def write_report(results: dict[str, dict], persistence_by_steps: dict[int, dict]) -> None:
    """[21] 写简短报告。"""
    overall = pd.DataFrame(overall_rows(results, persistence_by_steps)).sort_values("test_rmse")
    lines = [
        "# 三站点 36h 当前状态融合方式对比",
        "",
        "## 设计",
        "- 站点：下界首 -> 文图 -> 富足山。",
        "- 数据：2023-01-01 起，4 小时粒度。",
        "- 输出：未来 1 步目标 delta，最终预测值 = 当前目标值 + 预测 delta。",
        "- A：corr-top3 diff-only。",
        "- B：corr-top3 diff + 当前目标值重复拼到每个历史步。",
        "- C：每个历史步输入该步 raw level + diff1。",
        "- D：corr-top3 diff 序列走 GRU，当前目标值走小 MLP，最后拼接输出 delta。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR) -> int:
    """[22] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    window.OUTPUT_DIR = output_dir
    random.seed(paper.SEED)
    np.random.seed(paper.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = tuple(station for station, _ in lag.STATION_FILES)
    data = delta.load_diff1_chain_data()
    dataset_summary = {
        "model": "wentu_chain_current_level_fusion_ablation",
        "output_dir": str(OUTPUT_DIR),
        "start_date": paper.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": paper.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "window_hours": steps_to_hours(INPUT_STEPS),
        "output_steps": OUTPUT_STEPS,
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "stations": list(stations),
        "station_files": [{"station": station, "path": str(path)} for station, path in lag.STATION_FILES],
        "corr_top_k": window.CORR_TOP_K,
        "experiments": list(experiment_names()),
        "coverage": lag.coverage_rows(data),
    }
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = lag.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    selected_rows = []

    diff_result, rows = window.run_one_experiment(
        torch, data, stations, INPUT_STEPS, include_current_level=False, device=device
    )
    results[diff_result["experiment"]] = _set_result_mode(diff_result, "diff_only")
    selected_rows.extend({**row, "input_mode": "diff_only"} for row in rows)

    current_result, rows = window.run_one_experiment(
        torch, data, stations, INPUT_STEPS, include_current_level=True, device=device
    )
    results[current_result["experiment"]] = _set_result_mode(current_result, "repeated_current_target")
    selected_rows.extend({**row, "input_mode": "repeated_current_target"} for row in rows)

    step_result, rows = run_step_level_experiment(torch, data, stations, device)
    results[step_result["experiment"]] = step_result
    selected_rows.extend(rows)

    dual_result, rows = run_dual_branch_experiment(torch, data, stations, device)
    results[dual_result["experiment"]] = dual_result
    selected_rows.extend(rows)

    persistence_by_steps = {INPUT_STEPS: window.persistence_for_window(data, stations, INPUT_STEPS)}
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence_by_steps,
        "experiments": _json_safe_results(results),
        "selected_features": selected_rows,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence_by_steps, selected_rows)
    write_report(results, persistence_by_steps)
    console.print(pd.DataFrame(overall_rows(results, persistence_by_steps)).sort_values("test_rmse").to_string(index=False), flush=True)
    console.print(pd.DataFrame(feature_metric_rows(results)).to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
