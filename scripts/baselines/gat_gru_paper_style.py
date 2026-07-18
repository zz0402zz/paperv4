#!/usr/bin/env python3
"""
Paper-style GAT-GRU baseline.

This script keeps the same 2022+, 6-to-1, all9-to-target5 dataset used by
gat_gru_baseline.py, but replaces the minimal model with an encoder-decoder
GAT-GRU structure closer to the Water 2026 paper:
- 2-layer GRU encoder
- masked GAT spatial attention
- Q-K-V temporal attention
- 2-layer GRU decoder
- fully connected output head
"""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS, load_or_build_4h_quantity_data, processed_data_provenance


# [01] 实验配置区：2022 起、4 小时、过去 24 小时预测未来 4 小时。
DATA_DIR = Path("data/quantity")
PROCESSED_DATA_PATH = Path("data/processed/v2/quantity_4h_observed.csv")
EDGES_PATH = Path("data/metadata/station_edges_verified_strict.csv")
OUTPUT_DIR = protocol.BASELINE_OUTPUT_ROOT / "model_defaults/gat_gru_paper_style_2022_6to1_all9_target5_verified_edges"

START_DATE = "2022-01-01"
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
RESAMPLE_RULE = "4h"
DROP_OUTLIERS = True
REBUILD_PROCESSED_DATA = False

INPUT_STEPS = 6
OUTPUT_STEPS = 1
INPUT_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS
TARGET_FEATURE_COLUMNS: tuple[str, ...] = base.TARGET_FEATURE_COLUMNS
APPEND_INPUT_MASK_FEATURES = True

# [02] 论文式训练参数：hidden=32、2层GRU、batch=64、最多300轮、早停10轮。
MAX_EPOCHS = 300
BATCH_SIZE = 64
HIDDEN_SIZE = 32
NUM_GRU_LAYERS = 2
LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4
LOSS_NAME = "l1"
HUBER_BETA = 1.0
GAT_DROPOUT = 0.05
GRU_DROPOUT = 0.05
HEAD_DROPOUT = 0.10
SEED = 42
DRY_RUN = False


def require_torch():
    """[03] 延迟导入 PyTorch。"""
    return base.require_torch()


def make_model(
    torch,
    input_dim: int,
    target_dim: int,
    output_steps: int = OUTPUT_STEPS,
    hidden_size: int = HIDDEN_SIZE,
):
    """[04] 定义论文式 encoder-decoder GAT-GRU。"""

    class SpatialGAT(torch.nn.Module):
        """[04-1] 单头 masked GAT，adj[target, source] 控制可聚合邻居。"""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.attn_target = torch.nn.Linear(hidden_size, 1, bias=False)
            self.attn_source = torch.nn.Linear(hidden_size, 1, bias=False)
            self.leaky_relu = torch.nn.LeakyReLU(0.2)
            self.dropout = torch.nn.Dropout(GAT_DROPOUT)

        def forward(self, x, adjacency, edge_prior=None):
            h = self.proj(x)
            scores = self.attn_target(h) + self.attn_source(h).transpose(1, 2)
            scores = self.leaky_relu(scores)
            if edge_prior is not None:
                prior = edge_prior.to(device=scores.device, dtype=scores.dtype).clamp_min(1e-6)
                scores = scores + torch.log(prior).unsqueeze(0)
            scores = scores.masked_fill(~adjacency.unsqueeze(0), -1e9)
            attention = torch.softmax(scores, dim=-1)
            return torch.nn.functional.elu(torch.matmul(self.dropout(attention), h))

    class PaperStyleGatGru(torch.nn.Module):
        """[04-2] 多层GRU编码、GAT空间注意力、QKV时间注意力、GRU解码。"""

        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = GRU_DROPOUT if NUM_GRU_LAYERS > 1 else 0.0
            self.encoder = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=NUM_GRU_LAYERS,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.spatial_gat = SpatialGAT()
            self.temporal_query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.temporal_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.temporal_value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.decoder = torch.nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=NUM_GRU_LAYERS,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.dropout = torch.nn.Dropout(HEAD_DROPOUT)
            self.output_head = torch.nn.Linear(hidden_size, target_dim)
            self.last_temporal_attention = None

        def forward(self, x, adjacency, edge_prior=None):
            batch_size, steps, node_count, _ = x.shape
            encoder_input = x.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, -1)
            encoder_output, _ = self.encoder(encoder_input)

            by_time = encoder_output.reshape(batch_size, node_count, steps, hidden_size).permute(0, 2, 1, 3)
            spatial = self.spatial_gat(
                by_time.reshape(batch_size * steps, node_count, hidden_size),
                adjacency,
                edge_prior=edge_prior,
            )
            spatial_seq = spatial.reshape(batch_size, steps, node_count, hidden_size)
            spatial_seq = spatial_seq.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, hidden_size)

            query = self.temporal_query(spatial_seq[:, -1:, :])
            key = self.temporal_key(spatial_seq)
            value = self.temporal_value(spatial_seq)
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(hidden_size)
            attention = torch.softmax(scores, dim=-1)
            self.last_temporal_attention = attention.detach()
            context = torch.matmul(attention, value)

            decoder_input = context.repeat(1, output_steps, 1)
            decoder_output, _ = self.decoder(decoder_input)
            prediction = self.output_head(self.dropout(decoder_output))
            prediction = prediction.reshape(batch_size, node_count, output_steps, target_dim)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return PaperStyleGatGru()


def make_loader(torch, split: dict[str, np.ndarray], shuffle: bool):
    """[05] 使用论文 batch size 构造 DataLoader。"""
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(split["x"], dtype=torch.float32),
        torch.as_tensor(split["y"], dtype=torch.float32),
        torch.as_tensor(split["y_mask"], dtype=torch.bool),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def make_loss_fn(torch):
    """[06] 训练损失沿用当前最佳的 L1，可通过配置切换。"""
    name = LOSS_NAME.lower()
    if name == "mse":
        return lambda pred, target: (pred - target) ** 2
    if name in {"l1", "mae"}:
        return lambda pred, target: torch.abs(pred - target)
    if name in {"huber", "smooth_l1", "smoothl1"}:
        return torch.nn.SmoothL1Loss(beta=HUBER_BETA, reduction="none")
    raise ValueError(f"Unsupported LOSS_NAME: {LOSS_NAME}")


def prepare_data() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, tuple[str, ...], dict]:
    """[07] 读取严格边和 2023 起 4 小时数据，并构造图窗口。"""
    data = load_or_build_4h_quantity_data(
        DATA_DIR,
        PROCESSED_DATA_PATH,
        START_DATE,
        RESAMPLE_RULE,
        DROP_OUTLIERS,
        REBUILD_PROCESSED_DATA,
    )
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    edges = base.read_edge_frame(EDGES_PATH)
    adjacency = base.build_adjacency_matrix(stations, edges)
    dataset = base.build_graph_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=INPUT_FEATURE_COLUMNS,
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=RESAMPLE_RULE,
    )
    splits = base.split_graph_by_time(dataset, TRAIN_END, VAL_END)
    summary = {
        "model": "paper_style_gat_gru",
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
        "edge_count_without_self_loops": int(adjacency.sum() - len(stations)),
        "split_summary": base.graph_split_summary(splits),
    }
    return splits, adjacency, stations, summary


def train_epoch(torch, model, loader, adjacency, optimizer, loss_fn, device) -> float:
    """[08] 训练一个 epoch。"""
    model.train()
    losses = []
    for x, y, y_mask in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, adjacency)
        loss = base.masked_loss(torch, prediction, y, y_mask, loss_fn)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(torch, model, loader, adjacency, scalers: base.GraphForecastScalers, stations, device) -> dict:
    """[09] 评估并还原到原始量纲。"""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for x, y, y_mask in loader:
            preds.append(model(x.to(device), adjacency).cpu().numpy())
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
    adjacency_np: np.ndarray,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    device,
) -> dict:
    """[10] 训练 self 或 graph 变体，使用验证集早停。"""
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_DECAY_FACTOR,
        patience=LR_DECAY_PATIENCE,
    )
    loss_fn = make_loss_fn(torch)
    best_rmse = float("inf")
    bad_epochs = 0
    best_model_path = OUTPUT_DIR / f"{name}_paper_gat_gru_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(torch, model, loaders["train"], adjacency, optimizer, loss_fn, device)
        val_metrics = evaluate(torch, model, loaders["val"], adjacency, scalers, stations, device)
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_rmse < best_rmse - MIN_DELTA
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
                    "architecture": "encoder_decoder_gat_gru_qkv_temporal_attention",
                    "input_columns": list(INPUT_FEATURE_COLUMNS),
                    "target_columns": list(TARGET_FEATURE_COLUMNS),
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "hidden_size": HIDDEN_SIZE,
                    "num_gru_layers": NUM_GRU_LAYERS,
                    "stations": list(stations),
                    "adjacency": adjacency_np.astype(int).tolist(),
                    "scalers": scalers.to_dict(),
                },
                best_model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                console.print(f"{name} early_stop epoch={epoch:03d} best_val_rmse={best_rmse:.6f}", flush=True)
                break

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
        "best_model_path": str(best_model_path),
    }


def save_tables(results: dict[str, dict], stations: tuple[str, ...], persistence: dict[str, dict]) -> None:
    """[11] 保存和基础版同结构的对比表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base.station_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(base.feature_metric_rows(results)).to_csv(
        OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(base.station_feature_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(base.comparison_rows(results, stations, persistence)).to_csv(
        OUTPUT_DIR / "comparison_vs_baselines.csv", index=False, encoding="utf-8-sig"
    )
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    """[12] 主流程：训练 self-loop 与严格图两个论文式变体。"""
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits, graph_adjacency, stations, dataset_summary = prepare_data()
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    if DRY_RUN:
        console.print("DRY_RUN=True，只构造数据，不训练。", flush=True)
        return 0
    if len(splits["train"]["x"]) == 0 or len(splits["val"]["x"]) == 0:
        raise SystemExit("训练集或验证集为空，请检查时间切分和缺失数据。")

    torch = require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    scaled_splits, scalers = base.scale_graph_splits(splits, APPEND_INPUT_MASK_FEATURES)
    self_adjacency = np.eye(len(stations), dtype=bool)
    results = {
        "self": run_experiment(torch, "self", self_adjacency, scaled_splits, scalers, stations, device),
        "graph": run_experiment(torch, "graph", graph_adjacency, scaled_splits, scalers, stations, device),
    }
    persistence = base.evaluate_persistence_baseline(splits, stations)
    metrics = {
        "config": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_gru_layers": NUM_GRU_LAYERS,
            "learning_rate": LEARNING_RATE,
            "lr_decay_factor": LR_DECAY_FACTOR,
            "lr_decay_patience": LR_DECAY_PATIENCE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
            "loss_name": LOSS_NAME,
            "gat_dropout": GAT_DROPOUT,
            "gru_dropout": GRU_DROPOUT,
            "head_dropout": HEAD_DROPOUT,
            "seed": SEED,
        },
        "dataset_summary": dataset_summary,
        "persistence_baseline": persistence,
        "experiments": results,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence)
    console.print(pd.DataFrame(base.comparison_rows(results, stations, persistence)).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
