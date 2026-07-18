#!/usr/bin/env python3
"""Run direct-edge physical-lag GRU on 下界首 -> 文图 -> 富足山."""

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
from scripts.common.wq_gru_data import (
    FEATURE_COLUMNS,
    StandardScaler,
    clean_quantity_frame,
    interpolate_short_gaps,
)


# [01] 实验配置：只测试直接相邻边，物理时延来自距离 + 实测平均流速的粗估。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_physical_lag_2023_6to1_direct_edges"
STATION_FILES: tuple[tuple[str, Path], ...] = (
    ("下界首", Path("data/quantity/下界首2015年01月01日至2025年05月15日最新有效数据表.xls")),
    ("文图", Path("data/省控小时数据/文图2015年01月01日至2025年05月15日最新有效数据表.xls")),
    ("富足山", Path("data/quantity/富足山2015年01月01日至2025年05月15日最新有效数据表.xls")),
)
PHYSICAL_EDGES = pd.DataFrame(
    {
        "source_station": ["下界首", "文图"],
        "target_station": ["文图", "富足山"],
        "lag_steps": [2, 1],
        "chosen_lag_hours": [8, 4],
        "straight_distance_km": [14.759, 4.567],
        "river_distance_factor_range": ["1.2-1.5", "1.2-1.5"],
        "velocity_reference_station": ["常山(三)", "常山(三)"],
    }
)
FLOW_MEASUREMENT_PATH = Path("data/流量数据-水利厅/实测流量表xls.xls")
UPSTREAM_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A_transport_core": ("电导率(μS/cm)", "水温(℃)", "浊度(NTU)"),
    "B_transport_nutrients": ("电导率(μS/cm)", "水温(℃)", "浊度(NTU)", "总磷(mg/L)", "总氮(mg/L)"),
    "C_transport_reactive": (
        "电导率(μS/cm)",
        "水温(℃)",
        "浊度(NTU)",
        "总磷(mg/L)",
        "总氮(mg/L)",
        "高锰酸盐指数(mg/L)",
        "氨氮(mg/L)",
    ),
    "D_all9": FEATURE_COLUMNS,
}

INPUT_STEPS = paper.INPUT_STEPS
OUTPUT_STEPS = paper.OUTPUT_STEPS
INPUT_FEATURE_COLUMNS: tuple[str, ...] = paper.INPUT_FEATURE_COLUMNS
TARGET_FEATURE_COLUMNS: tuple[str, ...] = paper.TARGET_FEATURE_COLUMNS
APPEND_INPUT_MASK_FEATURES = paper.APPEND_INPUT_MASK_FEATURES


def require_torch():
    """[02] 延迟导入 PyTorch，便于单测只检查数据逻辑。"""
    return base.require_torch()


def load_station_frame(station: str, path: Path) -> pd.DataFrame:
    """[03] 读取单站小时表，硬异常置空后重采样到 4 小时。"""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_excel(path, sheet_name=0)
    cleaned = clean_quantity_frame(raw, paper.START_DATE, paper.DROP_OUTLIERS)
    by_time = cleaned.groupby("time")[list(FEATURE_COLUMNS)].mean()
    frame = by_time.resample(paper.RESAMPLE_RULE).mean().reset_index()
    frame["station"] = station
    return frame[["station", "time", *FEATURE_COLUMNS]]


def load_chain_data() -> pd.DataFrame:
    """[04] 合并三站数据，并只做短缺口插值。"""
    frames = [load_station_frame(station, path) for station, path in STATION_FILES]
    data = pd.concat(frames, ignore_index=True).sort_values(["station", "time"])
    return interpolate_short_gaps(data, limit=3)


def _empty_physical_lag_dataset(
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
        "self_mask": np.empty((0, input_steps, station_count, input_dim), dtype=bool),
        "upstream_mask": np.empty((0, station_count, input_dim), dtype=bool),
        "y_mask": np.empty((0, output_steps, station_count, target_dim), dtype=bool),
        "target_start": np.asarray([], dtype="datetime64[ns]"),
        "target_end": np.asarray([], dtype="datetime64[ns]"),
        "upstream_time": np.empty((0, station_count), dtype="datetime64[ns]"),
    }


def build_physical_lag_dataset(
    data: pd.DataFrame,
    stations: tuple[str, ...] | list[str],
    edges: pd.DataFrame,
    input_steps: int = INPUT_STEPS,
    output_steps: int = OUTPUT_STEPS,
    input_columns: tuple[str, ...] = INPUT_FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = TARGET_FEATURE_COLUMNS,
    freq: str = paper.RESAMPLE_RULE,
) -> dict[str, np.ndarray]:
    """[05] 构造窗口：自站用过去 input_steps，上游只取直接父节点的物理时延单点。"""
    stations = tuple(stations)
    if not stations:
        return _empty_physical_lag_dataset(input_steps, output_steps, 0, len(input_columns), len(target_columns))

    required = ["station", "time", *dict.fromkeys([*input_columns, *target_columns])]
    work = data[required].copy()
    work["station"] = work["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    for column in dict.fromkeys([*input_columns, *target_columns]):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    start = work["time"].min().ceil(freq)
    end = work["time"].max().floor(freq)
    times = pd.date_range(start, end, freq=freq)
    if len(times) < input_steps + output_steps:
        return _empty_physical_lag_dataset(
            input_steps,
            output_steps,
            len(stations),
            len(input_columns),
            len(target_columns),
        )

    grouped = work.groupby(["time", "station"], sort=True).mean(numeric_only=True)
    full_index = pd.MultiIndex.from_product([times, stations], names=["time", "station"])
    panel = grouped.reindex(full_index)
    input_values = panel.loc[:, input_columns].to_numpy(float).reshape(len(times), len(stations), len(input_columns))
    target_values = panel.loc[:, target_columns].to_numpy(float).reshape(len(times), len(stations), len(target_columns))

    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    edge_specs = []
    for row in edges.itertuples(index=False):
        source = str(getattr(row, "source_station"))
        target = str(getattr(row, "target_station"))
        lag_steps = int(getattr(row, "lag_steps"))
        if lag_steps < 1:
            raise ValueError("lag_steps must be >= 1 to avoid future leakage.")
        if source in station_to_idx and target in station_to_idx:
            edge_specs.append((station_to_idx[source], station_to_idx[target], lag_steps))

    total_steps = input_steps + output_steps
    self_xs, upstream_xs, ys = [], [], []
    self_masks, upstream_masks, y_masks = [], [], []
    target_starts, target_ends, upstream_times = [], [], []
    for start_idx in range(len(times) - total_steps + 1):
        target_idx = start_idx + input_steps
        self_x = input_values[start_idx : start_idx + input_steps]
        y = target_values[target_idx : start_idx + total_steps]
        self_mask = np.isfinite(self_x)
        y_mask = np.isfinite(y)
        if not y_mask.any():
            continue

        upstream_x = np.full((len(stations), len(input_columns)), np.nan, dtype=float)
        upstream_mask = np.zeros((len(stations), len(input_columns)), dtype=bool)
        upstream_time = np.full((len(stations),), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        for source_idx, target_station_idx, lag_steps in edge_specs:
            lag_idx = target_idx - lag_steps
            if lag_idx < 0:
                continue
            values = input_values[lag_idx, source_idx, :]
            upstream_x[target_station_idx] = values
            upstream_mask[target_station_idx] = np.isfinite(values)
            upstream_time[target_station_idx] = times[lag_idx].to_datetime64()

        self_xs.append(self_x)
        upstream_xs.append(upstream_x)
        ys.append(y)
        self_masks.append(self_mask)
        upstream_masks.append(upstream_mask)
        y_masks.append(y_mask)
        target_starts.append(times[target_idx].to_datetime64())
        target_ends.append(times[start_idx + total_steps - 1].to_datetime64())
        upstream_times.append(upstream_time)

    if not self_xs:
        return _empty_physical_lag_dataset(input_steps, output_steps, len(stations), len(input_columns), len(target_columns))

    return {
        "self_x": np.stack(self_xs),
        "upstream_x": np.stack(upstream_xs),
        "y": np.stack(ys),
        "self_mask": np.stack(self_masks),
        "upstream_mask": np.stack(upstream_masks),
        "y_mask": np.stack(y_masks),
        "target_start": np.asarray(target_starts, dtype="datetime64[ns]"),
        "target_end": np.asarray(target_ends, dtype="datetime64[ns]"),
        "upstream_time": np.stack(upstream_times).astype("datetime64[ns]"),
    }


def split_physical_lag_by_time(
    dataset: dict[str, np.ndarray],
    train_end: str = paper.TRAIN_END,
    val_end: str = paper.VAL_END,
) -> dict[str, dict[str, np.ndarray]]:
    """[06] 按目标时间切分，保持因果顺序。"""
    starts = pd.to_datetime(dataset["target_start"])
    ends = pd.to_datetime(dataset["target_end"])
    masks = {
        "train": ends < pd.Timestamp(train_end),
        "val": (starts >= pd.Timestamp(train_end)) & (ends < pd.Timestamp(val_end)),
        "test": starts >= pd.Timestamp(val_end),
    }
    return {name: {key: value[np.asarray(mask)] for key, value in dataset.items()} for name, mask in masks.items()}


def filter_upstream_features(
    dataset: dict[str, np.ndarray],
    upstream_columns: tuple[str, ...] | list[str],
    input_columns: tuple[str, ...] = INPUT_FEATURE_COLUMNS,
) -> dict[str, np.ndarray]:
    """[06-1] 只保留被允许向下游传递的上游特征，其他上游特征置空。"""
    missing = [column for column in upstream_columns if column not in input_columns]
    if missing:
        raise ValueError(f"Unknown upstream columns: {missing}")
    selected = np.zeros(len(input_columns), dtype=bool)
    selected[[input_columns.index(column) for column in upstream_columns]] = True

    filtered = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in dataset.items()}
    filtered["upstream_x"][..., ~selected] = np.nan
    filtered["upstream_mask"][..., ~selected] = False
    return filtered


def prepare_model_inputs(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[07] 标准化后的缺失值置 0，并拼接观测 mask。"""
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not APPEND_INPUT_MASK_FEATURES:
        return filled.astype(np.float32)
    return np.concatenate([filled, mask.astype(float)], axis=-1).astype(np.float32)


def scale_physical_lag_splits(
    splits: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, dict[str, np.ndarray]], base.GraphForecastScalers]:
    """[08] 只用训练集拟合 scaler，自站历史和上游单点共用输入 scaler。"""
    input_scaler = StandardScaler().fit(splits["train"]["self_x"])
    target_scaler = StandardScaler().fit(splits["train"]["y"])
    scalers = base.GraphForecastScalers(input_scaler, target_scaler)

    scaled = {}
    for split_name, split in splits.items():
        self_scaled = input_scaler.transform(split["self_x"])
        upstream_scaled = input_scaler.transform(split["upstream_x"])
        y_scaled = target_scaler.transform(split["y"])
        scaled[split_name] = {
            **split,
            "self_x": prepare_model_inputs(self_scaled, split["self_mask"]),
            "upstream_x": prepare_model_inputs(upstream_scaled, split["upstream_mask"]),
            "y": np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
            "y_mask": split["y_mask"].astype(bool),
        }
    return scaled, scalers


def make_model(
    torch,
    self_input_dim: int,
    upstream_input_dim: int,
    target_dim: int,
    output_steps: int = OUTPUT_STEPS,
    use_upstream: bool = True,
    hidden_size: int = paper.HIDDEN_SIZE,
):
    """[09] 自站 GRU 编码；physical_lag 变体额外融合上游物理时延单点。"""

    class PhysicalLagGru(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = paper.GRU_DROPOUT if paper.NUM_GRU_LAYERS > 1 else 0.0
            self.use_upstream = use_upstream
            self.encoder = torch.nn.GRU(
                input_size=self_input_dim,
                hidden_size=hidden_size,
                num_layers=paper.NUM_GRU_LAYERS,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            if use_upstream:
                self.upstream_proj = torch.nn.Sequential(
                    torch.nn.Linear(upstream_input_dim, hidden_size, bias=False),
                    torch.nn.ReLU(),
                )
                head_input_dim = hidden_size * 2
            else:
                self.upstream_proj = None
                head_input_dim = hidden_size
            self.dropout = torch.nn.Dropout(paper.HEAD_DROPOUT)
            self.head = torch.nn.Linear(head_input_dim, output_steps * target_dim)

        def forward(self, self_x, upstream_x=None):
            batch_size, steps, node_count, _ = self_x.shape
            encoded_input = self_x.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, -1)
            encoded, _ = self.encoder(encoded_input)
            self_state = encoded[:, -1, :].reshape(batch_size, node_count, hidden_size)
            if self.use_upstream:
                if upstream_x is None:
                    raise ValueError("upstream_x is required when use_upstream=True")
                upstream_state = self.upstream_proj(upstream_x)
                state = torch.cat([self_state, upstream_state], dim=-1)
            else:
                state = self_state
            prediction = self.head(self.dropout(state)).reshape(batch_size, node_count, output_steps, target_dim)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return PhysicalLagGru()


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[10] 包装 DataLoader。"""
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(split["self_x"], dtype=torch.float32),
        torch.as_tensor(split["upstream_x"], dtype=torch.float32),
        torch.as_tensor(split["y"], dtype=torch.float32),
        torch.as_tensor(split["y_mask"], dtype=torch.bool),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=paper.BATCH_SIZE, shuffle=shuffle)


def train_epoch(torch, model, loader, optimizer, loss_fn, device) -> float:
    """[11] 训练一个 epoch。"""
    model.train()
    losses = []
    for self_x, upstream_x, y, y_mask in loader:
        self_x = self_x.to(device)
        upstream_x = upstream_x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(self_x, upstream_x)
        loss = base.masked_loss(torch, prediction, y, y_mask, loss_fn)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(torch, model, loader, scalers: base.GraphForecastScalers, stations: tuple[str, ...], device) -> dict:
    """[12] 还原到原始量纲后计算 MAE/RMSE/NSE。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for self_x, upstream_x, y, y_mask in loader:
            preds.append(model(self_x.to(device), upstream_x.to(device)).cpu().numpy())
            trues.append(y.numpy())
            masks.append(y_mask.numpy())
    if not preds:
        empty = np.empty((0, OUTPUT_STEPS, len(stations), len(TARGET_FEATURE_COLUMNS)))
        return base.masked_error_metrics(empty, empty.astype(bool), TARGET_FEATURE_COLUMNS, stations)
    pred = scalers.inverse_transform_target(np.concatenate(preds))
    true = scalers.inverse_transform_target(np.concatenate(trues))
    mask = np.concatenate(masks).astype(bool)
    return base.masked_error_metrics(pred - true, mask, TARGET_FEATURE_COLUMNS, stations, truth=true)


def run_experiment(
    torch,
    name: str,
    use_upstream: bool,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[13] 训练 self 或 physical_lag 变体，按验证集 RMSE 早停。"""
    torch.manual_seed(paper.SEED)
    loaders = {
        split_name: make_loader(torch, split, shuffle=(split_name == "train"))
        for split_name, split in scaled_splits.items()
    }
    model = make_model(
        torch,
        self_input_dim=scaled_splits["train"]["self_x"].shape[-1],
        upstream_input_dim=scaled_splits["train"]["upstream_x"].shape[-1],
        target_dim=len(TARGET_FEATURE_COLUMNS),
        output_steps=OUTPUT_STEPS,
        use_upstream=use_upstream,
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
    best_model_path = OUTPUT_DIR / f"{name}_gru_physical_lag_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    history = []

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = train_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = evaluate(torch, model, loaders["val"], scalers, stations, device)
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
                    "architecture": "self_gru_plus_direct_physical_lag_upstream_point",
                    "input_columns": list(INPUT_FEATURE_COLUMNS),
                    "target_columns": list(TARGET_FEATURE_COLUMNS),
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "use_upstream": use_upstream,
                    "physical_edges": PHYSICAL_EDGES.to_dict(orient="records"),
                    "stations": list(stations),
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
    best_metrics = {
        split_name: evaluate(torch, model, loaders[split_name], scalers, stations, device)
        for split_name in ["train", "val", "test"]
    }
    return {
        "experiment": name,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["val_rmse"]),
        "best_checkpoint": best_metrics,
        "best_model_path": str(best_model_path),
    }


def target_input_indices(input_columns: tuple[str, ...], target_columns: tuple[str, ...]) -> tuple[int, ...]:
    """[14] 找目标指标在输入指标中的位置。"""
    missing = [feature for feature in target_columns if feature not in input_columns]
    if missing:
        raise ValueError(f"Persistence baseline needs target features in inputs: {missing}")
    return tuple(input_columns.index(feature) for feature in target_columns)


def evaluate_persistence_baseline(splits: dict[str, dict[str, np.ndarray]], stations: tuple[str, ...]) -> dict[str, dict]:
    """[15] 持久性 baseline：未来等于本站输入窗口最后一个观测。"""
    indices = target_input_indices(INPUT_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS)
    metrics = {}
    for split_name, split in splits.items():
        if len(split["self_x"]) == 0:
            empty = np.empty((0, OUTPUT_STEPS, len(stations), len(TARGET_FEATURE_COLUMNS)))
            metrics[split_name] = base.masked_error_metrics(empty, empty.astype(bool), TARGET_FEATURE_COLUMNS, stations)
            continue
        pred = np.repeat(split["self_x"][:, -1:, :, indices], split["y"].shape[1], axis=1)
        valid = split["y_mask"] & np.isfinite(pred)
        metrics[split_name] = base.masked_error_metrics(
            pred - split["y"],
            valid,
            TARGET_FEATURE_COLUMNS,
            stations,
            truth=split["y"],
        )
    return metrics


def coverage_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    """[16] 汇总输入覆盖率。"""
    rows = []
    for station, group in data.groupby("station", sort=True):
        for feature in FEATURE_COLUMNS:
            total = int(len(group))
            valid = int(group[feature].notna().sum())
            rows.append(
                {
                    "station": station,
                    "feature": feature,
                    "total_points": total,
                    "valid_points": valid,
                    "valid_rate": float(valid / total) if total else 0.0,
                }
            )
    return rows


def flow_velocity_summary(path: Path = FLOW_MEASUREMENT_PATH, station: str = "常山(三)") -> dict[str, object]:
    """[17] 从实测流量表提取参考断面平均流速。"""
    if not path.exists():
        return {"path": str(path), "station": station, "available": False}
    frame = pd.read_excel(path, sheet_name=0, header=1)
    frame["测流起时间"] = pd.to_datetime(frame["测流起时间"], errors="coerce")
    frame["断面平均流速"] = pd.to_numeric(frame["断面平均流速"], errors="coerce")
    selected = frame[
        (frame["站名"].astype(str) == station)
        & (frame["测流起时间"] >= pd.Timestamp(paper.START_DATE))
        & frame["断面平均流速"].notna()
    ]
    if selected.empty:
        return {"path": str(path), "station": station, "available": False}
    velocity = selected["断面平均流速"]
    return {
        "path": str(path),
        "station": station,
        "available": True,
        "measurements": int(len(velocity)),
        "median_velocity_mps": float(velocity.median()),
        "p25_velocity_mps": float(velocity.quantile(0.25)),
        "p75_velocity_mps": float(velocity.quantile(0.75)),
        "latest_measurement_time": str(selected["测流起时间"].max()),
    }


def graph_split_summary(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, object]]:
    """[18] 汇总切分窗口数量和目标有效点。"""
    summary = {}
    for split_name, split in splits.items():
        starts = split["target_start"]
        ends = split["target_end"]
        summary[split_name] = {
            "windows": int(len(split["self_x"])),
            "self_x_shape": list(split["self_x"].shape),
            "upstream_x_shape": list(split["upstream_x"].shape),
            "y_shape": list(split["y"].shape),
            "valid_target_points": int(split["y_mask"].sum()),
            "valid_upstream_points": int(split["upstream_mask"].sum()),
            "start": str(starts.min()) if len(starts) else "",
            "end": str(ends.max()) if len(ends) else "",
        }
    return summary


def build_splits(data: pd.DataFrame, stations: tuple[str, ...]) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """[19] 构造物理时延窗口并生成数据摘要。"""
    dataset = build_physical_lag_dataset(
        data,
        stations=stations,
        edges=PHYSICAL_EDGES,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=INPUT_FEATURE_COLUMNS,
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=paper.RESAMPLE_RULE,
    )
    splits = split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    summary = {
        "model": "self_gru_plus_direct_physical_lag_upstream_point",
        "relation_type": "direct_edges_only_physical_delay_single_point",
        "chain": "下界首 -> 文图 -> 富足山",
        "output_dir": str(OUTPUT_DIR),
        "start_date": paper.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": paper.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "input_features": list(INPUT_FEATURE_COLUMNS),
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "stations": list(stations),
        "station_files": [{"station": station, "path": str(path)} for station, path in STATION_FILES],
        "physical_edges": PHYSICAL_EDGES.to_dict(orient="records"),
        "flow_velocity_reference": flow_velocity_summary(),
        "split_summary": graph_split_summary(splits),
        "coverage": coverage_rows(data),
    }
    return splits, summary


def overall_rows(results: dict[str, dict], persistence: dict[str, dict]) -> list[dict[str, object]]:
    """[20] 汇总整体 test 指标。"""
    rows = []
    for experiment, result in results.items():
        test = result["best_checkpoint"]["test"]
        best_epoch = result["best_epoch"]
        rows.append(
            {
                "experiment": experiment,
                "best_epoch": best_epoch.get("epoch"),
                "val_rmse": best_epoch.get("val_rmse"),
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


def station_metric_rows(results: dict[str, dict], stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[21] 逐站点整体指标。"""
    rows = []
    for experiment, result in results.items():
        metrics = result["best_checkpoint"]["test"]["station_metrics"]
        for station in stations:
            item = metrics.get(station, {})
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


def feature_metric_rows(results: dict[str, dict]) -> list[dict[str, object]]:
    """[22] 逐指标整体指标。"""
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
    """[23] 逐站点逐指标指标。"""
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


def _metric_value(item: dict | None, key: str) -> float | None:
    if not item:
        return None
    value = item.get(key)
    return None if value is None or pd.isna(value) else float(value)


def comparison_rows(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> list[dict[str, object]]:
    """[24] physical_lag 相比 self 和持久性 baseline 的逐站点差值。"""
    self_metrics = results["self"]["best_checkpoint"]["test"]["station_metrics"]
    lag_metrics = results["physical_lag"]["best_checkpoint"]["test"]["station_metrics"]
    persistence_metrics = persistence["test"]["station_metrics"]
    rows = []
    for station in stations:
        self_item = self_metrics.get(station, {})
        lag_item = lag_metrics.get(station, {})
        persistence_item = persistence_metrics.get(station, {})
        self_rmse = _metric_value(self_item, "rmse")
        lag_rmse = _metric_value(lag_item, "rmse")
        persistence_rmse = _metric_value(persistence_item, "rmse")
        self_mae = _metric_value(self_item, "mae")
        lag_mae = _metric_value(lag_item, "mae")
        persistence_mae = _metric_value(persistence_item, "mae")
        self_nse = _metric_value(self_item, "nse")
        lag_nse = _metric_value(lag_item, "nse")
        persistence_nse = _metric_value(persistence_item, "nse")
        rows.append(
            {
                "station": station,
                "physical_lag_test_mae": lag_mae,
                "physical_lag_test_rmse": lag_rmse,
                "physical_lag_test_nse": lag_nse,
                "self_test_mae": self_mae,
                "self_test_rmse": self_rmse,
                "self_test_nse": self_nse,
                "persistence_test_mae": persistence_mae,
                "persistence_test_rmse": persistence_rmse,
                "persistence_test_nse": persistence_nse,
                "delta_rmse_physical_lag_minus_self": lag_rmse - self_rmse
                if lag_rmse is not None and self_rmse is not None
                else None,
                "delta_mae_physical_lag_minus_self": lag_mae - self_mae
                if lag_mae is not None and self_mae is not None
                else None,
                "delta_nse_physical_lag_minus_self": lag_nse - self_nse
                if lag_nse is not None and self_nse is not None
                else None,
                "delta_rmse_physical_lag_minus_persistence": lag_rmse - persistence_rmse
                if lag_rmse is not None and persistence_rmse is not None
                else None,
                "delta_nse_physical_lag_minus_persistence": lag_nse - persistence_nse
                if lag_nse is not None and persistence_nse is not None
                else None,
            }
        )
    return rows


def save_tables(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> None:
    """[25] 保存结果表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows(results, persistence)).to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
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


def write_report(results: dict[str, dict], persistence: dict[str, dict], stations: tuple[str, ...]) -> None:
    """[26] 写简短实验报告。"""
    rows = overall_rows(results, persistence)
    comparison = pd.DataFrame(comparison_rows(results, stations, persistence))
    lag_row = next(row for row in rows if row["experiment"] == "physical_lag")
    self_row = next(row for row in rows if row["experiment"] == "self")
    persist = next(row for row in rows if row["experiment"] == "persistence")
    improved_vs_self = int((comparison["delta_rmse_physical_lag_minus_self"] < 0).sum())
    improved_vs_persistence = int((comparison["delta_rmse_physical_lag_minus_persistence"] < 0).sum())
    lines = [
        "# 下界首-文图-富足山物理时延 GRU 试跑",
        "",
        "## 口径",
        "- 数据：2023-01-01 起，4 小时水质序列，短缺口插值。",
        "- 自站输入：过去 6 个 4 小时时间步，即 T-24h 到 T-4h。",
        "- 上游输入：只传直接父节点的一个物理时延点，不传隔站，不传完整 6 步。",
        "- 物理时延：下界首 -> 文图 使用 T-8h；文图 -> 富足山 使用 T-4h。",
        "- 输出：未来 1 个 4 小时时间步，5 个核心指标。",
        "",
        "## 整体结果",
        f"- physical_lag：MAE={lag_row['test_mae']:.6f}，RMSE={lag_row['test_rmse']:.6f}，NSE={lag_row['test_nse']:.6f}。",
        f"- self：MAE={self_row['test_mae']:.6f}，RMSE={self_row['test_rmse']:.6f}，NSE={self_row['test_nse']:.6f}。",
        f"- persistence：MAE={persist['test_mae']:.6f}，RMSE={persist['test_rmse']:.6f}，NSE={persist['test_nse']:.6f}。",
        f"- physical_lag 相比 self 的逐站点 RMSE 改善：{improved_vs_self}/{len(stations)} 个站点。",
        f"- physical_lag 相比 persistence 的逐站点 RMSE 改善：{improved_vs_persistence}/{len(stations)} 个站点。",
        "",
        "## 输出表",
        "- `overall_summary.csv`：整体指标。",
        "- `comparison_vs_baselines.csv`：逐站点 self/physical_lag/persistence 对比。",
        "- `station_feature_metrics.csv`：逐站点逐指标指标。",
        "- `dataset_summary.json`：物理边、时延、流速参考和覆盖率。",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """[27] 主流程。"""
    random.seed(paper.SEED)
    np.random.seed(paper.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = tuple(station for station, _ in STATION_FILES)
    data = load_chain_data()
    splits, dataset_summary = build_splits(data, stations)
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    pd.DataFrame(coverage_rows(data)).to_csv(OUTPUT_DIR / "input_coverage.csv", index=False, encoding="utf-8-sig")
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if len(splits["train"]["self_x"]) == 0 or len(splits["val"]["self_x"]) == 0:
        raise SystemExit("训练集或验证集为空，请检查时间切分和缺失数据。")

    torch = require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    scaled_splits, scalers = scale_physical_lag_splits(splits)
    results = {
        "self": run_experiment(torch, "self", False, scaled_splits, scalers, stations, device),
        "physical_lag": run_experiment(torch, "physical_lag", True, scaled_splits, scalers, stations, device),
    }
    persistence = evaluate_persistence_baseline(splits, stations)
    metrics = {
        "config": {
            "max_epochs": paper.MAX_EPOCHS,
            "batch_size": paper.BATCH_SIZE,
            "hidden_size": paper.HIDDEN_SIZE,
            "num_gru_layers": paper.NUM_GRU_LAYERS,
            "learning_rate": paper.LEARNING_RATE,
            "lr_decay_factor": paper.LR_DECAY_FACTOR,
            "lr_decay_patience": paper.LR_DECAY_PATIENCE,
            "early_stopping_patience": paper.EARLY_STOPPING_PATIENCE,
            "min_delta": paper.MIN_DELTA,
            "loss_name": paper.LOSS_NAME,
            "gru_dropout": paper.GRU_DROPOUT,
            "head_dropout": paper.HEAD_DROPOUT,
            "seed": paper.SEED,
        },
        "dataset_summary": dataset_summary,
        "persistence_baseline": persistence,
        "experiments": results,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence)
    write_report(results, persistence, stations)
    console.print(pd.DataFrame(overall_rows(results, persistence)).to_string(index=False), flush=True)
    console.print(pd.DataFrame(comparison_rows(results, stations, persistence)).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
