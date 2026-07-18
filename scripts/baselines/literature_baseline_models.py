#!/usr/bin/env python3
"""Literature-style baselines on the frozen V2 four-hour water-quality task."""

from __future__ import annotations

from scripts.common.terminal_output import console

import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.common import v2_experiment_protocol as protocol
from scripts.common import wq_modeling_common as common
from scripts.common.wq_gru_data import FEATURE_COLUMNS, load_or_build_4h_quantity_data


# [01] V2 单步基线：2022 起、9 步输入、1 步输出、9 入 5 出。
OUTPUT_DIR = protocol.BASELINE_OUTPUT_ROOT / "stage2_baselines_9to1" / "seed_42"

DATA_DIR = paper.DATA_DIR
PROCESSED_DATA_PATH = paper.PROCESSED_DATA_PATH
START_DATE = paper.START_DATE
TRAIN_END = paper.TRAIN_END
VAL_END = paper.VAL_END
RESAMPLE_RULE = paper.RESAMPLE_RULE
DROP_OUTLIERS = paper.DROP_OUTLIERS
REBUILD_PROCESSED_DATA = paper.REBUILD_PROCESSED_DATA

INPUT_STEPS = 9
OUTPUT_STEPS = 1
INPUT_FEATURE_COLUMNS = FEATURE_COLUMNS
TARGET_FEATURE_COLUMNS = paper.TARGET_FEATURE_COLUMNS
APPEND_INPUT_MASK_FEATURES = True

MAX_EPOCHS = 100
BATCH_SIZE = 2048
HIDDEN_SIZE = 64
STATION_EMBED_DIM = 8
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4
RIDGE_ALPHA = 1.0
SEED = 42


def run_config(output_dir: Path, seed: int) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "seed": int(seed),
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "station_embed_dim": STATION_EMBED_DIM,
        "learning_rate": LEARNING_RATE,
        "ridge_alpha": RIDGE_ALPHA,
    }


def prepare_data() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], base.GraphForecastScalers, tuple[str, ...], dict]:
    """[02] 构造全站点图窗口，再把同一份窗口展平成站点级样本。"""
    data = load_or_build_4h_quantity_data(
        DATA_DIR,
        PROCESSED_DATA_PATH,
        START_DATE,
        RESAMPLE_RULE,
        DROP_OUTLIERS,
        REBUILD_PROCESSED_DATA,
    )
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    dataset = base.build_graph_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=INPUT_FEATURE_COLUMNS,
        target_columns=TARGET_FEATURE_COLUMNS,
        freq=RESAMPLE_RULE,
    )
    raw_splits = base.split_graph_by_time(dataset, TRAIN_END, VAL_END)
    scaled_splits, scalers = base.scale_graph_splits(raw_splits, APPEND_INPUT_MASK_FEATURES)
    summary = {
        "model_family": "literature_baselines_water_quality_only",
        "processed_data_path": str(PROCESSED_DATA_PATH),
        "start_date": START_DATE,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "resample_rule": RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "input_features": list(INPUT_FEATURE_COLUMNS),
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "append_input_mask_features": APPEND_INPUT_MASK_FEATURES,
        "station_count": len(stations),
        "stations": list(stations),
        "split_summary": base.graph_split_summary(raw_splits),
    }
    return raw_splits, scaled_splits, scalers, stations, summary


def ridge_fit_predict(
    train_samples: dict[str, np.ndarray],
    eval_samples: dict[str, np.ndarray],
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """[06] 逐目标闭式 Ridge 回归；不依赖 sklearn，作为 LR/线性 baseline。"""
    x_train = common.tabular_features(train_samples)
    x_eval = common.tabular_features(eval_samples)
    y_train = train_samples["y"].astype(np.float64)
    mask_train = train_samples["mask"]
    output_steps, target_dim = y_train.shape[1:]
    y_flat = y_train.reshape(len(y_train), output_steps * target_dim)
    mask_flat = mask_train.reshape(len(mask_train), output_steps * target_dim)
    pred = np.zeros((len(x_eval), output_steps * target_dim), dtype=np.float32)
    reg = np.eye(x_train.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    station_start = 1 + INPUT_STEPS * train_samples["x"].shape[-1]
    reg[station_start:, station_start:] = 0.0

    for target_idx in range(output_steps * target_dim):
        valid = mask_flat[:, target_idx]
        x_valid = x_train[valid]
        y_valid = y_flat[valid, target_idx]
        if len(y_valid) == 0:
            continue
        coef = np.linalg.solve(x_valid.T @ x_valid + reg, x_valid.T @ y_valid)
        pred[:, target_idx] = (x_eval @ coef).astype(np.float32)
    return pred.reshape(len(x_eval), output_steps, target_dim)

def make_loader(torch, samples: dict[str, np.ndarray], shuffle: bool, train_only_valid: bool) -> object:
    """[08] 构造 DataLoader。"""
    return common.make_station_loader(torch, samples, BATCH_SIZE, shuffle=shuffle, train_only_valid=train_only_valid)


def make_torch_model(torch, name: str, input_dim: int, station_count: int, target_dim: int):
    """[09] 定义 GRU、MLP、LSTM、TCN/CNN 四个常见深度 baseline。"""

    class MlpBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, STATION_EMBED_DIM)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(INPUT_STEPS * input_dim + STATION_EMBED_DIM, HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HIDDEN_SIZE, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            flat = x.reshape(x.shape[0], -1)
            output = self.net(torch.cat([flat, self.station_embedding(station_id)], dim=-1))
            return output.reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    class LstmBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, STATION_EMBED_DIM)
            self.rnn = torch.nn.LSTM(input_dim, HIDDEN_SIZE, num_layers=1, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HIDDEN_SIZE + STATION_EMBED_DIM, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            output, _ = self.rnn(x)
            prediction = self.head(torch.cat([output[:, -1], self.station_embedding(station_id)], dim=-1))
            return prediction.reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    class GruBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, STATION_EMBED_DIM)
            self.rnn = torch.nn.GRU(input_dim, HIDDEN_SIZE, num_layers=1, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HIDDEN_SIZE + STATION_EMBED_DIM, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            output, _ = self.rnn(x)
            prediction = self.head(torch.cat([output[:, -1], self.station_embedding(station_id)], dim=-1))
            return prediction.reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    class TcnBaseline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.station_embedding = torch.nn.Embedding(station_count, STATION_EMBED_DIM)
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(input_dim, HIDDEN_SIZE, kernel_size=3, padding=2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=2),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(0.10),
                torch.nn.Linear(HIDDEN_SIZE + STATION_EMBED_DIM, OUTPUT_STEPS * target_dim),
            )

        def forward(self, x, station_id):
            encoded = self.conv(x.transpose(1, 2))[..., -1]
            prediction = self.head(torch.cat([encoded, self.station_embedding(station_id)], dim=-1))
            return prediction.reshape(x.shape[0], OUTPUT_STEPS, target_dim)

    if name == "mlp":
        return MlpBaseline()
    if name == "gru":
        return GruBaseline()
    if name == "lstm":
        return LstmBaseline()
    if name == "tcn":
        return TcnBaseline()
    raise ValueError(f"Unknown torch baseline: {name}")


def evaluate_torch_model(torch, model, loader, device) -> np.ndarray:
    """[11] 输出展平站点样本的标准化预测值。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, station_id, _, _ in loader:
            pred = model(x.to(device), station_id.to(device)).cpu().numpy()
            preds.append(pred)
    return (
        np.concatenate(preds, axis=0)
        if preds
        else np.empty((0, OUTPUT_STEPS, len(TARGET_FEATURE_COLUMNS)), dtype=np.float32)
    )


def run_torch_baseline(
    torch,
    name: str,
    samples: dict[str, dict[str, np.ndarray]],
    station_count: int,
    device,
) -> tuple[dict, dict[str, np.ndarray]]:
    """[12] 训练一个深度 baseline，并返回各 split 预测。"""
    torch.manual_seed(SEED)
    model = make_torch_model(
        torch,
        name,
        input_dim=samples["train"]["x"].shape[-1],
        station_count=station_count,
        target_dim=len(TARGET_FEATURE_COLUMNS),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(torch, samples["train"], shuffle=True, train_only_valid=True)
    eval_loaders = {
        split: make_loader(torch, split_samples, shuffle=False, train_only_valid=False)
        for split, split_samples in samples.items()
    }
    best_state = None
    best_val_rmse = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for x, station_id, y, mask in train_loader:
            x = x.to(device)
            station_id = station_id.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = common.masked_l1_loss(torch, model(x, station_id), y, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = evaluate_torch_model(torch, model, eval_loaders["val"], device)
        val_rmse = common.scaled_flat_rmse(val_pred, samples["val"])
        improved = val_rmse < best_val_rmse - MIN_DELTA
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_scaled_rmse": val_rmse})
        console.print(f"{name} epoch={epoch:03d} train_loss={np.mean(losses):.6f} val_scaled_rmse={val_rmse:.6f}", flush=True)
        if improved:
            best_val_rmse = val_rmse
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                console.print(f"{name} early_stop epoch={epoch:03d} best_val_scaled_rmse={best_val_rmse:.6f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = {
        split: evaluate_torch_model(torch, model, loader, device)
        for split, loader in eval_loaders.items()
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": name,
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "input_features": list(INPUT_FEATURE_COLUMNS),
            "target_features": list(TARGET_FEATURE_COLUMNS),
            "history": history,
        },
        OUTPUT_DIR / f"{name}_best.pt",
    )
    return {"history": history, "best_epoch": min(history, key=lambda item: item["val_scaled_rmse"])}, predictions


def flat_prediction_to_metrics(
    pred_scaled: np.ndarray,
    samples: dict[str, np.ndarray],
    raw_split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
) -> dict:
    """[14] 将展平预测还原成 W,1,N,D，并使用统一指标函数评估。"""
    windows = samples["windows"]
    station_count = samples["station_count"]
    target_dim = len(TARGET_FEATURE_COLUMNS)
    pred_graph_scaled = pred_scaled.reshape(windows, station_count, OUTPUT_STEPS, target_dim).transpose(0, 2, 1, 3)
    pred = scalers.inverse_transform_target(pred_graph_scaled)
    true = raw_split["y"]
    return base.masked_error_metrics(pred - true, raw_split["y_mask"], TARGET_FEATURE_COLUMNS, stations, truth=true)


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    samples: dict[str, dict[str, np.ndarray]],
    raw_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
) -> dict:
    """[15] 对 train/val/test 三个 split 计算原始量纲指标。"""
    return {
        split: flat_prediction_to_metrics(pred, samples[split], raw_splits[split], scalers, stations)
        for split, pred in predictions.items()
    }


def prediction_rows(model_name: str, metrics: dict, best_epoch: dict | None = None) -> dict:
    """[16] 生成整体结果行。"""
    test = metrics["test"]
    return {
        "model": model_name,
        "best_epoch": None if best_epoch is None else best_epoch.get("epoch"),
        "val_rmse": metrics["val"].get("rmse"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_nse": test.get("nse"),
        "valid_points": test.get("valid_points"),
    }


def feature_rows(model_name: str, metrics: dict) -> list[dict[str, object]]:
    """[17] 生成逐指标行。"""
    test = metrics["test"]
    rows = []
    for feature in TARGET_FEATURE_COLUMNS:
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "valid_points": test["feature_valid_points"].get(feature, 0),
                "test_mae": test["feature_mae"].get(feature),
                "test_rmse": test["feature_rmse"].get(feature),
                "test_nse": test["feature_nse"].get(feature),
            }
        )
    return rows


def station_rows(model_name: str, metrics: dict, stations: tuple[str, ...]) -> list[dict[str, object]]:
    """[18] 生成逐站点整体行。"""
    rows = []
    for station in stations:
        item = metrics["test"]["station_metrics"].get(station, {})
        rows.append(
            {
                "model": model_name,
                "station": station,
                "valid_points": item.get("valid_points", 0),
                "test_mae": item.get("mae"),
                "test_rmse": item.get("rmse"),
                "test_nse": item.get("nse"),
            }
        )
    return rows


def run_suite(output_dir: Path = OUTPUT_DIR, seed: int = SEED) -> int:
    """[20] 从同一份 V2 窗口重新训练全部短期基线。"""
    global OUTPUT_DIR, SEED
    OUTPUT_DIR = Path(output_dir)
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_splits, scaled_splits, scalers, stations, dataset_summary = prepare_data()
    manifest = protocol.build_run_manifest(
        experiment="stage2_literature_baselines_9to1",
        output_dir=OUTPUT_DIR,
        seed=SEED,
        code_paths=(Path("scripts/baselines/literature_baseline_models.py"), Path("scripts/common/v2_experiment_protocol.py")),
    )
    dataset_summary["run_manifest"] = manifest
    samples = {split: common.station_samples(scaled_splits[split], len(stations)) for split in ("train", "val", "test")}
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    base.save_json(OUTPUT_DIR / "run_manifest.json", manifest)

    summary_rows: list[dict[str, object]] = []
    feature_metric_rows: list[dict[str, object]] = []
    station_metric_rows: list[dict[str, object]] = []

    persistence = base.evaluate_persistence_baseline(raw_splits, stations)
    summary_rows.append(prediction_rows("persistence", persistence))
    feature_metric_rows.extend(feature_rows("persistence", persistence))
    station_metric_rows.extend(station_rows("persistence", persistence, stations))

    ridge_predictions = {
        split: ridge_fit_predict(samples["train"], samples[split])
        for split in ("train", "val", "test")
    }
    ridge_metrics = evaluate_predictions(ridge_predictions, samples, raw_splits, scalers, stations)
    summary_rows.append(prediction_rows("ridge_lr", ridge_metrics))
    feature_metric_rows.extend(feature_rows("ridge_lr", ridge_metrics))
    station_metric_rows.extend(station_rows("ridge_lr", ridge_metrics, stations))

    torch = paper.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    model_artifacts = {"ridge_lr": {"alpha": RIDGE_ALPHA}}
    for name in ("gru", "mlp", "lstm", "tcn"):
        artifact, predictions = run_torch_baseline(torch, name, samples, len(stations), device)
        metrics = evaluate_predictions(predictions, samples, raw_splits, scalers, stations)
        summary_rows.append(prediction_rows(name, metrics, artifact["best_epoch"]))
        feature_metric_rows.extend(feature_rows(name, metrics))
        station_metric_rows.extend(station_rows(name, metrics, stations))
        model_artifacts[name] = artifact

    summary = pd.DataFrame(summary_rows)
    if "gru" in set(summary["model"]):
        ref_rmse = float(summary.loc[summary["model"] == "gru", "test_rmse"].iloc[0])
        summary["rmse_delta_vs_gru_self"] = summary["test_rmse"] - ref_rmse
        summary["rmse_pct_delta_vs_gru_self"] = summary["rmse_delta_vs_gru_self"] / ref_rmse * 100
    summary = summary.sort_values("test_rmse", na_position="last")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_metric_rows).to_csv(OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_metric_rows).to_csv(OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig")
    base.save_json(
        OUTPUT_DIR / "metrics.json",
        {
            "dataset_summary": dataset_summary,
            "config": run_config(OUTPUT_DIR, SEED),
            "artifacts": model_artifacts,
        },
    )
    console.print(summary.to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
