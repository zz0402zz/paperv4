#!/usr/bin/env python3
"""
GAT-GRU baseline for 4-hour water-quality forecasting.

The script trains two comparable variants on the same 2022+ graph windows:
- self: self-loop only, equivalent to using each station's own history.
- graph: self-loops plus the draft upstream/downstream station graph.
"""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import (
    FEATURE_COLUMNS,
    StandardScaler,
    load_or_build_4h_quantity_data,
    processed_data_provenance,
    target_ok_column,
)


# [01] 实验配置区：2022 起、4 小时、过去 24 小时预测未来 4 小时。
DATA_DIR = Path("data/quantity")
PROCESSED_DATA_PATH = Path("data/processed/v2/quantity_4h_observed.csv")
EDGES_PATH = Path("data/metadata/station_edges_verified_strict.csv")
OUTPUT_DIR = protocol.BASELINE_OUTPUT_ROOT / "model_defaults/gat_gru_baseline_2022_6to1_all9_target5"
SINGLE_STATION_REFERENCE_PATH = Path(
    protocol.BASELINE_OUTPUT_ROOT
    / "reference/gru_forecast_station_separate_target5_6to1_l1/raw_station_run_summary_all8.csv"
)

START_DATE = "2022-01-01"
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
RESAMPLE_RULE = "4h"
DROP_OUTLIERS = True
REBUILD_PROCESSED_DATA = False

INPUT_STEPS = 6
OUTPUT_STEPS = 1
INPUT_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS
TARGET_FEATURE_COLUMNS: tuple[str, ...] = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
APPEND_INPUT_MASK_FEATURES = True

EPOCHS = 30
BATCH_SIZE = 128
GAT_HIDDEN_SIZE = 32
GRU_HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
LOSS_NAME = "l1"
HUBER_BETA = 1.0
GAT_DROPOUT = 0.05
HEAD_DROPOUT = 0.10
SEED = 42
DRY_RUN = False


def require_torch():
    """[02] 延迟导入 PyTorch，缺依赖时给出清楚提示。"""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("训练需要 PyTorch：python -m pip install torch") from exc
    return torch


def save_json(path: Path, payload: dict) -> None:
    """[03] 保存 JSON 结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_edge_frame(path: str | Path) -> pd.DataFrame:
    """[04] 读取站点边文件；缺文件时返回空边表。"""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["source_station", "target_station"])
    edges = pd.read_csv(path)
    required = {"source_station", "target_station"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edge file missing columns: {sorted(missing)}")
    return edges


def build_adjacency_matrix(
    stations: list[str] | tuple[str, ...],
    edges: pd.DataFrame,
    include_self_loops: bool = True,
) -> np.ndarray:
    """[05] 构造 target-by-source 邻接矩阵：adj[下游, 上游] = True。"""
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    adjacency = np.zeros((len(stations), len(stations)), dtype=bool)
    if include_self_loops:
        np.fill_diagonal(adjacency, True)

    for row in edges.itertuples(index=False):
        source = str(getattr(row, "source_station"))
        target = str(getattr(row, "target_station"))
        if source in station_to_idx and target in station_to_idx:
            adjacency[station_to_idx[target], station_to_idx[source]] = True
    return adjacency


def _empty_graph_dataset(
    input_steps: int,
    output_steps: int,
    station_count: int,
    input_dim: int,
    target_dim: int,
) -> dict[str, np.ndarray]:
    return {
        "x": np.empty((0, input_steps, station_count, input_dim), dtype=float),
        "y": np.empty((0, output_steps, station_count, target_dim), dtype=float),
        "x_mask": np.empty((0, input_steps, station_count, input_dim), dtype=bool),
        "y_mask": np.empty((0, output_steps, station_count, target_dim), dtype=bool),
        "target_start": np.asarray([], dtype="datetime64[ns]"),
        "target_end": np.asarray([], dtype="datetime64[ns]"),
    }


def build_graph_dataset(
    data: pd.DataFrame,
    stations: list[str] | tuple[str, ...],
    input_steps: int = INPUT_STEPS,
    output_steps: int = OUTPUT_STEPS,
    input_columns: tuple[str, ...] = INPUT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = TARGET_FEATURE_COLUMNS,
    freq: str = RESAMPLE_RULE,
) -> dict[str, np.ndarray]:
    """[06] 构造全站点图窗口；缺失目标用 y_mask 标记，不直接丢整窗。"""
    if not stations:
        return _empty_graph_dataset(input_steps, output_steps, 0, len(input_columns), len(target_columns))

    target_quality_columns = tuple(
        target_ok_column(column) for column in target_columns if target_ok_column(column) in data.columns
    )
    work = data[
        ["station", "time", *dict.fromkeys([*input_columns, *target_columns, *target_quality_columns])]
    ].copy()
    work["station"] = work["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    for column in dict.fromkeys([*input_columns, *target_columns]):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    start = work["time"].min().ceil(freq)
    end = work["time"].max().floor(freq)
    times = pd.date_range(start, end, freq=freq)
    if len(times) < input_steps + output_steps:
        return _empty_graph_dataset(input_steps, output_steps, len(stations), len(input_columns), len(target_columns))

    grouped = work.groupby(["time", "station"], sort=True).mean(numeric_only=True)
    full_index = pd.MultiIndex.from_product([times, stations], names=["time", "station"])
    panel = grouped.reindex(full_index)
    input_values = panel.loc[:, input_columns].to_numpy(float).reshape(len(times), len(stations), len(input_columns))
    target_values = panel.loc[:, target_columns].to_numpy(float).reshape(len(times), len(stations), len(target_columns))
    if len(target_quality_columns) == len(target_columns):
        target_quality = (
            panel.loc[:, target_quality_columns]
            .fillna(False)
            .to_numpy(bool)
            .reshape(len(times), len(stations), len(target_columns))
        )
    else:
        target_quality = np.ones_like(target_values, dtype=bool)

    total_steps = input_steps + output_steps
    xs, ys, x_masks, y_masks, target_starts, target_ends = [], [], [], [], [], []
    for start_idx in range(len(times) - total_steps + 1):
        x = input_values[start_idx : start_idx + input_steps]
        y = target_values[start_idx + input_steps : start_idx + total_steps]
        x_mask = np.isfinite(x)
        y_quality = target_quality[start_idx + input_steps : start_idx + total_steps]
        y_mask = np.isfinite(y) & y_quality
        if not y_mask.any():
            continue
        xs.append(x)
        ys.append(y)
        x_masks.append(x_mask)
        y_masks.append(y_mask)
        target_starts.append(times[start_idx + input_steps].to_datetime64())
        target_ends.append(times[start_idx + total_steps - 1].to_datetime64())

    if not xs:
        return _empty_graph_dataset(input_steps, output_steps, len(stations), len(input_columns), len(target_columns))

    return {
        "x": np.stack(xs),
        "y": np.stack(ys),
        "x_mask": np.stack(x_masks),
        "y_mask": np.stack(y_masks),
        "target_start": np.asarray(target_starts, dtype="datetime64[ns]"),
        "target_end": np.asarray(target_ends, dtype="datetime64[ns]"),
    }


def split_graph_by_time(dataset: dict[str, np.ndarray], train_end: str, val_end: str) -> dict[str, dict[str, np.ndarray]]:
    """[07] 按目标时间切分 train/val/test，保持时间顺序。"""
    start = pd.to_datetime(dataset["target_start"])
    end = pd.to_datetime(dataset["target_end"])
    masks = {
        "train": end < pd.Timestamp(train_end),
        "val": (start >= pd.Timestamp(train_end)) & (end < pd.Timestamp(val_end)),
        "test": start >= pd.Timestamp(val_end),
    }
    return {
        name: {key: value[np.asarray(mask)] for key, value in dataset.items()}
        for name, mask in masks.items()
    }


class GraphForecastScalers:
    """[08] 管理图模型输入和目标标准化器。"""

    def __init__(self, input_scaler: StandardScaler, target_scaler: StandardScaler) -> None:
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """[08-1] 将目标预测还原到原始量纲。"""
        return self.target_scaler.inverse_transform(values)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """[08-2] 保存标准化参数。"""
        return {
            "input": self.input_scaler.to_dict(),
            "target": self.target_scaler.to_dict(),
        }


def prepare_model_inputs(x_scaled: np.ndarray, x_mask: np.ndarray, append_mask_features: bool = True) -> np.ndarray:
    """[09] 模型输入缺失值用训练均值对应的 0 替代，可选拼接观测 mask。"""
    x_filled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if not append_mask_features:
        return x_filled.astype(np.float32)
    return np.concatenate([x_filled, x_mask.astype(float)], axis=-1).astype(np.float32)


def scale_graph_splits(
    splits: dict[str, dict[str, np.ndarray]],
    append_mask_features: bool = APPEND_INPUT_MASK_FEATURES,
) -> tuple[dict[str, dict[str, np.ndarray]], GraphForecastScalers]:
    """[10] 只用训练集拟合 scaler，再标准化并填补模型输入缺失值。"""
    input_scaler = StandardScaler().fit(splits["train"]["x"])
    target_scaler = StandardScaler().fit(splits["train"]["y"])
    scalers = GraphForecastScalers(input_scaler, target_scaler)

    scaled: dict[str, dict[str, np.ndarray]] = {}
    for name, split in splits.items():
        x_scaled = input_scaler.transform(split["x"])
        y_scaled = target_scaler.transform(split["y"])
        scaled[name] = {
            **split,
            "x": prepare_model_inputs(x_scaled, split["x_mask"], append_mask_features),
            "y": np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            "y_mask": split["y_mask"].astype(bool),
        }
    return scaled, scalers


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_rmse(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values**2)))


def _safe_nse(error_values: np.ndarray, truth_values: np.ndarray | None) -> float | None:
    if truth_values is None:
        return None
    error_values = np.asarray(error_values, dtype=float)
    truth_values = np.asarray(truth_values, dtype=float)
    valid = np.isfinite(error_values) & np.isfinite(truth_values)
    if not valid.any():
        return None
    error_values = error_values[valid]
    truth_values = truth_values[valid]
    denominator = float(np.sum((truth_values - np.mean(truth_values)) ** 2))
    if denominator <= np.finfo(float).eps:
        return None
    return float(1.0 - float(np.sum(error_values**2)) / denominator)


def masked_error_metrics(
    error: np.ndarray,
    mask: np.ndarray,
    feature_names: tuple[str, ...] = TARGET_FEATURE_COLUMNS,
    stations: tuple[str, ...] | list[str] | None = None,
    truth: np.ndarray | None = None,
) -> dict:
    """[11] 只在真实目标存在的位置计算 MAE/RMSE/NSE，并给出逐站点逐指标结果。"""
    error = np.asarray(error, dtype=float)
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != error.shape:
            raise ValueError(f"truth shape {truth.shape} does not match error shape {error.shape}")
    mask = np.asarray(mask, dtype=bool) & np.isfinite(error)
    if truth is not None:
        mask = mask & np.isfinite(truth)
    abs_error = np.abs(error)

    metrics = {
        "windows": int(error.shape[0]) if error.ndim else 0,
        "valid_points": int(mask.sum()),
        "mae": _safe_mean(abs_error[mask]),
        "rmse": _safe_rmse(error[mask]),
        "nse": _safe_nse(error[mask], truth[mask] if truth is not None else None),
        "feature_valid_points": {},
        "feature_mae": {},
        "feature_rmse": {},
        "feature_nse": {},
        "station_metrics": {},
    }

    for feature_idx, feature in enumerate(feature_names):
        feature_mask = mask[..., feature_idx]
        feature_error = error[..., feature_idx]
        feature_truth = truth[..., feature_idx] if truth is not None else None
        metrics["feature_valid_points"][feature] = int(feature_mask.sum())
        metrics["feature_mae"][feature] = _safe_mean(np.abs(feature_error)[feature_mask])
        metrics["feature_rmse"][feature] = _safe_rmse(feature_error[feature_mask])
        metrics["feature_nse"][feature] = _safe_nse(
            feature_error[feature_mask],
            feature_truth[feature_mask] if feature_truth is not None else None,
        )

    if stations is not None:
        for station_idx, station in enumerate(stations):
            station_mask = mask[:, :, station_idx, :]
            station_error = error[:, :, station_idx, :]
            station_truth = truth[:, :, station_idx, :] if truth is not None else None
            station_item = {
                "valid_points": int(station_mask.sum()),
                "mae": _safe_mean(np.abs(station_error)[station_mask]),
                "rmse": _safe_rmse(station_error[station_mask]),
                "nse": _safe_nse(
                    station_error[station_mask],
                    station_truth[station_mask] if station_truth is not None else None,
                ),
                "feature_valid_points": {},
                "feature_mae": {},
                "feature_rmse": {},
                "feature_nse": {},
            }
            for feature_idx, feature in enumerate(feature_names):
                feature_mask = station_mask[..., feature_idx]
                feature_error = station_error[..., feature_idx]
                feature_truth = station_truth[..., feature_idx] if station_truth is not None else None
                station_item["feature_valid_points"][feature] = int(feature_mask.sum())
                station_item["feature_mae"][feature] = _safe_mean(np.abs(feature_error)[feature_mask])
                station_item["feature_rmse"][feature] = _safe_rmse(feature_error[feature_mask])
                station_item["feature_nse"][feature] = _safe_nse(
                    feature_error[feature_mask],
                    feature_truth[feature_mask] if feature_truth is not None else None,
                )
            metrics["station_metrics"][str(station)] = station_item
    return metrics


def make_loss_fn(torch, name: str = LOSS_NAME):
    """[12] 返回逐元素损失函数；最终由 y_mask 做加权平均。"""
    loss_name = name.lower()
    if loss_name == "mse":
        return lambda pred, target: (pred - target) ** 2
    if loss_name in {"l1", "mae"}:
        return lambda pred, target: torch.abs(pred - target)
    if loss_name in {"huber", "smooth_l1", "smoothl1"}:
        return torch.nn.SmoothL1Loss(beta=HUBER_BETA, reduction="none")
    raise ValueError(f"Unsupported LOSS_NAME: {name}")


def masked_loss(torch, pred, target, mask, loss_fn) -> object:
    """[13] 只让存在真实标签的位置参与反向传播。"""
    weights = mask.to(dtype=pred.dtype)
    loss_values = loss_fn(pred, target)
    return (loss_values * weights).sum() / weights.sum().clamp_min(1.0)


def make_model(
    torch,
    input_dim: int,
    target_dim: int,
    output_steps: int = OUTPUT_STEPS,
    gat_hidden_size: int = GAT_HIDDEN_SIZE,
    gru_hidden_size: int = GRU_HIDDEN_SIZE,
):
    """[14] 定义轻量 GAT-GRU：每个时间步做图注意力，再用 GRU 编码时间。"""

    class GraphAttentionLayer(torch.nn.Module):
        """[14-1] 单头 masked GAT，adj[target, source] 控制可见上游。"""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(input_dim, gat_hidden_size, bias=False)
            self.attn_target = torch.nn.Linear(gat_hidden_size, 1, bias=False)
            self.attn_source = torch.nn.Linear(gat_hidden_size, 1, bias=False)
            self.leaky_relu = torch.nn.LeakyReLU(0.2)
            self.dropout = torch.nn.Dropout(GAT_DROPOUT)

        def forward(self, x, adjacency):
            h = self.proj(x)
            scores = self.attn_target(h) + self.attn_source(h).transpose(1, 2)
            scores = self.leaky_relu(scores)
            scores = scores.masked_fill(~adjacency.unsqueeze(0), -1e9)
            attention = torch.softmax(scores, dim=-1)
            return torch.nn.functional.elu(torch.matmul(self.dropout(attention), h))

    class GatGruForecast(torch.nn.Module):
        """[14-2] GAT 空间编码 + GRU 时间编码 + 线性预测头。"""

        def __init__(self) -> None:
            super().__init__()
            self.gat = GraphAttentionLayer()
            self.gru = torch.nn.GRU(
                input_size=gat_hidden_size,
                hidden_size=gru_hidden_size,
                num_layers=NUM_LAYERS,
                batch_first=True,
            )
            self.dropout = torch.nn.Dropout(HEAD_DROPOUT)
            self.head = torch.nn.Linear(gru_hidden_size, output_steps * target_dim)

        def forward(self, x, adjacency):
            batch_size, steps, node_count, _ = x.shape
            spatial = self.gat(x.reshape(batch_size * steps, node_count, -1), adjacency)
            temporal = spatial.reshape(batch_size, steps, node_count, -1).permute(0, 2, 1, 3)
            temporal = temporal.reshape(batch_size * node_count, steps, -1)
            output, _ = self.gru(temporal)
            encoded = self.dropout(output[:, -1, :])
            prediction = self.head(encoded).reshape(batch_size, node_count, output_steps, target_dim)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return GatGruForecast()


def choose_device(torch):
    """[15] 自动选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[16] 将图窗口包装成 DataLoader。"""
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(split["x"], dtype=torch.float32),
        torch.as_tensor(split["y"], dtype=torch.float32),
        torch.as_tensor(split["y_mask"], dtype=torch.bool),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_epoch(torch, model, loader, adjacency, optimizer, loss_fn, device) -> float:
    """[17] 训练一个 epoch。"""
    model.train()
    losses = []
    for x, y, y_mask in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, adjacency)
        loss = masked_loss(torch, prediction, y, y_mask, loss_fn)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(torch, model, loader, adjacency, scalers: GraphForecastScalers, stations, device) -> dict:
    """[18] 评估模型，并还原到原始量纲计算指标。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for x, y, y_mask in loader:
            preds.append(model(x.to(device), adjacency).cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())

    if not preds:
        empty = np.empty((0, OUTPUT_STEPS, len(stations), len(TARGET_FEATURE_COLUMNS)))
        return masked_error_metrics(empty, empty.astype(bool), TARGET_FEATURE_COLUMNS, stations)

    pred = scalers.inverse_transform_target(np.concatenate(preds))
    true = scalers.inverse_transform_target(np.concatenate(trues))
    mask = np.concatenate(masks).astype(bool)
    return masked_error_metrics(pred - true, mask, TARGET_FEATURE_COLUMNS, stations, truth=true)


def target_input_indices(input_columns: tuple[str, ...], target_columns: tuple[str, ...]) -> tuple[int, ...]:
    """[19] 找到目标指标在输入指标中的位置，供持久性 baseline 使用。"""
    missing = [feature for feature in target_columns if feature not in input_columns]
    if missing:
        raise ValueError(f"Persistence baseline needs target features in inputs: {missing}")
    return tuple(input_columns.index(feature) for feature in target_columns)


def evaluate_persistence_baseline(splits: dict[str, dict[str, np.ndarray]], stations) -> dict[str, dict]:
    """[20] 持久性 baseline：未来等于输入窗口最后一个观测。"""
    indices = target_input_indices(INPUT_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS)
    metrics = {}
    for name, split in splits.items():
        if len(split["x"]) == 0:
            empty = np.empty((0, OUTPUT_STEPS, len(stations), len(TARGET_FEATURE_COLUMNS)))
            metrics[name] = masked_error_metrics(empty, empty.astype(bool), TARGET_FEATURE_COLUMNS, stations)
            continue
        pred = np.repeat(split["x"][:, -1:, :, indices], split["y"].shape[1], axis=1)
        valid = split["y_mask"] & np.isfinite(pred)
        metrics[name] = masked_error_metrics(pred - split["y"], valid, TARGET_FEATURE_COLUMNS, stations, truth=split["y"])
    return metrics


def graph_split_summary(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, object]]:
    """[21] 汇总图窗口切分情况。"""
    summary = {}
    for name, split in splits.items():
        starts = split["target_start"]
        ends = split["target_end"]
        summary[name] = {
            "windows": int(len(split["x"])),
            "x_shape": list(split["x"].shape),
            "y_shape": list(split["y"].shape),
            "valid_target_points": int(split["y_mask"].sum()),
            "start": str(starts.min()) if len(starts) else "",
            "end": str(ends.max()) if len(ends) else "",
        }
    return summary


def prepare_data() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, tuple[str, ...], dict]:
    """[22] 读取 2023 起 4 小时数据，构造图窗口和邻接矩阵。"""
    data = load_or_build_4h_quantity_data(
        DATA_DIR,
        PROCESSED_DATA_PATH,
        START_DATE,
        RESAMPLE_RULE,
        DROP_OUTLIERS,
        REBUILD_PROCESSED_DATA,
    )
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    edges = read_edge_frame(EDGES_PATH)
    adjacency = build_adjacency_matrix(stations, edges)
    dataset = build_graph_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=INPUT_FEATURE_COLUMNS,
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=RESAMPLE_RULE,
    )
    splits = split_graph_by_time(dataset, TRAIN_END, VAL_END)
    edge_count_without_self = int(adjacency.sum() - len(stations))
    summary = {
        "model": "GAT-GRU baseline",
        **processed_data_provenance(PROCESSED_DATA_PATH),
        "edges_path": str(EDGES_PATH),
        "start_date": START_DATE,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "input_features": list(INPUT_FEATURE_COLUMNS),
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "append_input_mask_features": APPEND_INPUT_MASK_FEATURES,
        "station_count": len(stations),
        "stations": list(stations),
        "edge_count_without_self_loops": edge_count_without_self,
        "split_summary": graph_split_summary(splits),
    }
    return splits, adjacency, stations, summary


def run_experiment(
    torch,
    name: str,
    adjacency_np: np.ndarray,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: GraphForecastScalers,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[23] 训练一个模型变体并返回最优 checkpoint 指标。"""
    torch.manual_seed(SEED)
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.bool, device=device)
    loaders = {
        split_name: make_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = make_model(
        torch,
        input_dim=scaled_splits["train"]["x"].shape[-1],
        target_dim=len(TARGET_FEATURE_COLUMNS),
        output_steps=OUTPUT_STEPS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = make_loss_fn(torch)
    best_rmse = float("inf")
    best_model_path = OUTPUT_DIR / f"{name}_gat_gru_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(torch, model, loaders["train"], adjacency, optimizer, loss_fn, device)
        val_metrics = evaluate(torch, model, loaders["val"], adjacency, scalers, stations, device)
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
        console.print(f"{name} epoch={epoch:03d} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f}", flush=True)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": name,
                    "input_columns": list(INPUT_FEATURE_COLUMNS),
                    "target_columns": list(TARGET_FEATURE_COLUMNS),
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "stations": list(stations),
                    "adjacency": adjacency_np.astype(int).tolist(),
                    "scalers": scalers.to_dict(),
                },
                best_model_path,
            )

    last_epoch_metrics = {
        split_name: evaluate(torch, model, loaders[split_name], adjacency, scalers, stations, device)
        for split_name in ["train", "val", "test"]
    }
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_metrics = {
        split_name: evaluate(torch, model, loaders[split_name], adjacency, scalers, stations, device)
        for split_name in ["train", "val", "test"]
    }
    return {
        "experiment": name,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["val_rmse"]),
        "best_checkpoint": best_metrics,
        "last_epoch": last_epoch_metrics,
        "best_model_path": str(best_model_path),
    }


def _metric_value(item: dict | None, key: str) -> float | None:
    if not item:
        return None
    value = item.get(key)
    return None if value is None or pd.isna(value) else float(value)


def station_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[24] 生成逐站点整体指标行。"""
    rows = []
    for experiment, result in results.items():
        station_metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            metrics = station_metrics.get(station, {})
            rows.append(
                {
                    "experiment": experiment,
                    "station": station,
                    "valid_points": metrics.get("valid_points", 0),
                    "test_mae": metrics.get("mae"),
                    "test_rmse": metrics.get("rmse"),
                    "test_nse": metrics.get("nse"),
                }
            )
    return rows


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[25] 生成整体逐指标指标行。"""
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


def station_feature_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[26] 生成逐站点逐指标指标行。"""
    rows = []
    for experiment, result in results.items():
        station_metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            metrics = station_metrics.get(station, {})
            for feature in TARGET_FEATURE_COLUMNS:
                rows.append(
                    {
                        "experiment": experiment,
                        "station": station,
                        "feature": feature,
                        "valid_points": metrics.get("feature_valid_points", {}).get(feature, 0),
                        "test_mae": metrics.get("feature_mae", {}).get(feature),
                        "test_rmse": metrics.get("feature_rmse", {}).get(feature),
                        "test_nse": metrics.get("feature_nse", {}).get(feature),
                    }
                )
    return rows


def load_single_station_reference() -> pd.DataFrame:
    """[27] 读取旧的单站点 all9 结果作为参考对比，不参与训练。"""
    if not SINGLE_STATION_REFERENCE_PATH.exists():
        return pd.DataFrame()
    reference = pd.read_csv(SINGLE_STATION_REFERENCE_PATH)
    if "experiment" not in reference.columns:
        return pd.DataFrame()
    return reference[(reference["experiment"] == "all9") & (reference["status"] == "ok")].copy()


def comparison_rows(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> list[dict[str, object]]:
    """[28] 汇总 graph 相比 self、持久性和旧单站点 GRU 的差值。"""
    self_metrics = results["self"]["best_checkpoint"]["test"]["station_metrics"]
    graph_metrics = results["graph"]["best_checkpoint"]["test"]["station_metrics"]
    persistence_metrics = persistence["test"]["station_metrics"]
    reference = load_single_station_reference()
    reference_map = {}
    if not reference.empty:
        reference_map = {
            str(row["station"]): row
            for _, row in reference.iterrows()
        }

    rows = []
    for station in stations:
        self_item = self_metrics.get(station, {})
        graph_item = graph_metrics.get(station, {})
        persistence_item = persistence_metrics.get(station, {})
        ref_item = reference_map.get(station)
        self_rmse = _metric_value(self_item, "rmse")
        graph_rmse = _metric_value(graph_item, "rmse")
        persistence_rmse = _metric_value(persistence_item, "rmse")
        self_mae = _metric_value(self_item, "mae")
        graph_mae = _metric_value(graph_item, "mae")
        persistence_mae = _metric_value(persistence_item, "mae")
        self_nse = _metric_value(self_item, "nse")
        graph_nse = _metric_value(graph_item, "nse")
        persistence_nse = _metric_value(persistence_item, "nse")
        reference_rmse = float(ref_item["test_rmse"]) if ref_item is not None and not pd.isna(ref_item["test_rmse"]) else None
        rows.append(
            {
                "station": station,
                "gat_graph_test_mae": graph_mae,
                "gat_graph_test_rmse": graph_rmse,
                "gat_graph_test_nse": graph_nse,
                "self_loop_test_mae": self_mae,
                "self_loop_test_rmse": self_rmse,
                "self_loop_test_nse": self_nse,
                "persistence_test_mae": persistence_mae,
                "persistence_test_rmse": persistence_rmse,
                "persistence_test_nse": persistence_nse,
                "delta_rmse_graph_minus_self": graph_rmse - self_rmse
                if graph_rmse is not None and self_rmse is not None
                else None,
                "delta_mae_graph_minus_self": graph_mae - self_mae
                if graph_mae is not None and self_mae is not None
                else None,
                "delta_nse_graph_minus_self": graph_nse - self_nse
                if graph_nse is not None and self_nse is not None
                else None,
                "delta_rmse_graph_minus_persistence": graph_rmse - persistence_rmse
                if graph_rmse is not None and persistence_rmse is not None
                else None,
                "delta_nse_graph_minus_persistence": graph_nse - persistence_nse
                if graph_nse is not None and persistence_nse is not None
                else None,
                "old_single_station_all9_rmse_reference": reference_rmse,
                "delta_rmse_graph_minus_old_all9_reference": graph_rmse - reference_rmse
                if graph_rmse is not None and reference_rmse is not None
                else None,
            }
        )
    return rows


def save_tables(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> None:
    """[29] 保存 CSV 表，便于直接查看逐站点提升。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(station_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(feature_metric_rows(results)).to_csv(
        OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(station_feature_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(comparison_rows(results, stations, persistence)).to_csv(
        OUTPUT_DIR / "comparison_vs_baselines.csv", index=False, encoding="utf-8-sig"
    )
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    """[30] 主流程：构造数据、训练 self/graph 两个变体、保存对比结果。"""
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits, graph_adjacency, stations, dataset_summary = prepare_data()
    save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if DRY_RUN:
        console.print("DRY_RUN=True，只构造数据，不训练。", flush=True)
        return 0
    if len(splits["train"]["x"]) == 0 or len(splits["val"]["x"]) == 0:
        raise SystemExit("训练集或验证集为空，请检查时间切分和缺失数据。")

    torch = require_torch()
    device = choose_device(torch)
    console.print(f"device={device}", flush=True)
    scaled_splits, scalers = scale_graph_splits(splits)
    self_adjacency = np.eye(len(stations), dtype=bool)

    results = {
        "self": run_experiment(torch, "self", self_adjacency, scaled_splits, scalers, stations, device),
        "graph": run_experiment(torch, "graph", graph_adjacency, scaled_splits, scalers, stations, device),
    }
    persistence = evaluate_persistence_baseline(splits, stations)
    metrics = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gat_hidden_size": GAT_HIDDEN_SIZE,
            "gru_hidden_size": GRU_HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss_name": LOSS_NAME,
            "gat_dropout": GAT_DROPOUT,
            "head_dropout": HEAD_DROPOUT,
            "seed": SEED,
        },
        "dataset_summary": dataset_summary,
        "persistence_baseline": persistence,
        "experiments": results,
    }
    save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence)
    console.print(pd.DataFrame(comparison_rows(results, stations, persistence)).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
