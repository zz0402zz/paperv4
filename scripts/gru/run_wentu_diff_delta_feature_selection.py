#!/usr/bin/env python3
"""Per-target feature-selection experiments for diff-delta GRU forecasts."""

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
from scripts.gru import run_wentu_diff_delta_gru as delta
from scripts.gru import run_wentu_physical_lag_gru as lag
from scripts.gru import run_wentu_self_feature_ablation as self_ablation
from scripts.common import v2_experiment_protocol as protocol
from scripts.common.wq_gru_data import FEATURE_COLUMNS


# [01] 实验配置：每个目标单独筛 diff 特征，再训练单目标 delta-GRU。
OUTPUT_DIR = protocol.GRU_OUTPUT_ROOT / "helpers/gru_wentu_diff_delta_feature_selection_2023_6to1_chain"
INPUT_STEPS = delta.INPUT_STEPS
OUTPUT_STEPS = delta.OUTPUT_STEPS
TARGET_FEATURE_COLUMNS = delta.TARGET_FEATURE_COLUMNS
CORR_TOP_K = 3
LASSO_FALLBACK_TOP_K = 3
LASSO_ALPHA = 0.002


@dataclass(frozen=True)
class ExperimentSpec:
    """[02] per-target 实验定义。"""

    selector: str
    description: str


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "per_target_self_diff_delta": ExperimentSpec(
        selector="self",
        description="每个目标只用自己的历史 diff1 预测自身未来 delta。",
    ),
    "per_target_corr_top3_delta": ExperimentSpec(
        selector="corr_top3",
        description="每个目标强制保留自身 diff1，并加入训练集相关性 top3 的其他 diff1。",
    ),
    "per_target_lasso_selected_delta": ExperimentSpec(
        selector="lasso",
        description="每个目标强制保留自身 diff1，并加入训练集 Lasso 选出的其他 diff1；无非零项时回退到相关性 top3。",
    ),
}


def safe_filename(value: str) -> str:
    """[03] 将中文实验/指标名转成安全文件名。"""
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def diff_columns_from_features(features: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """[04] 将原始指标名转成对应 diff1 输入列。"""
    return tuple(f"{feature}_diff1" for feature in features)


def self_diff_columns_for_target(target_feature: str) -> tuple[str, ...]:
    """[05] 单目标自回归 diff 输入列。"""
    return (f"{target_feature}_diff1",)


def _feature_order(feature: str) -> int:
    return FEATURE_COLUMNS.index(feature) if feature in FEATURE_COLUMNS else len(FEATURE_COLUMNS)


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values.reshape(values.shape[0], -1)


def _max_abs_corr(values: np.ndarray, target_delta: np.ndarray) -> tuple[float, int]:
    """[06] 返回某指标多个历史步中与目标 delta 最强的绝对相关。"""
    matrix = _as_2d(values)
    target = np.asarray(target_delta, dtype=float).reshape(-1)
    best_score = 0.0
    best_lag = -1
    for lag_idx in range(matrix.shape[1]):
        x = matrix[:, lag_idx]
        valid = np.isfinite(x) & np.isfinite(target)
        if int(valid.sum()) < 3:
            continue
        x_valid = x[valid]
        y_valid = target[valid]
        if np.nanstd(x_valid) <= 0 or np.nanstd(y_valid) <= 0:
            continue
        score = abs(float(np.corrcoef(x_valid, y_valid)[0, 1]))
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lag = lag_idx + 1
    return best_score, best_lag


def select_corr_topk_features(
    lagged_by_feature: dict[str, np.ndarray],
    target_delta: np.ndarray,
    target_feature: str,
    top_k: int = CORR_TOP_K,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """[07] 训练集相关性筛选：自身 diff 强制保留，另取 top-k 个其他指标。"""
    score_rows = []
    for feature, values in lagged_by_feature.items():
        score, best_lag = _max_abs_corr(values, target_delta)
        score_rows.append(
            {
                "feature": feature,
                "score": float(score),
                "best_lag_position": int(best_lag),
                "selection_reason": "not_selected",
            }
        )

    ranked = sorted(score_rows, key=lambda row: (-float(row["score"]), _feature_order(str(row["feature"]))))
    selected = [target_feature]
    for row in ranked:
        feature = str(row["feature"])
        if feature == target_feature:
            continue
        selected.append(feature)
        if len(selected) >= top_k + 1:
            break

    selected_set = set(selected)
    for row in score_rows:
        feature = str(row["feature"])
        if feature == target_feature:
            row["selection_reason"] = "forced_self"
        elif feature in selected_set:
            row["selection_reason"] = "corr_topk"
    return tuple(selected), score_rows


def _standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return (matrix - mean) / scale, mean, scale


def select_lasso_features(
    lagged_by_feature: dict[str, np.ndarray],
    target_delta: np.ndarray,
    target_feature: str,
    fallback_top_k: int = LASSO_FALLBACK_TOP_K,
    alpha: float = LASSO_ALPHA,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """[08] Lasso 筛选；无可用非零项或缺 sklearn 时回退到相关性筛选。"""
    features = tuple(lagged_by_feature)
    matrices = [_as_2d(lagged_by_feature[feature]) for feature in features]
    feature_slices = {}
    start = 0
    for feature, matrix in zip(features, matrices):
        stop = start + matrix.shape[1]
        feature_slices[feature] = slice(start, stop)
        start = stop
    x = np.concatenate(matrices, axis=1)
    y = np.asarray(target_delta, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if int(valid.sum()) < 20:
        selected, rows = select_corr_topk_features(lagged_by_feature, target_delta, target_feature, fallback_top_k)
        for row in rows:
            if row["selection_reason"] in {"forced_self", "corr_topk"}:
                row["selection_reason"] = f"fallback_corr_insufficient_rows_{row['selection_reason']}"
        return selected, rows

    try:
        from sklearn.linear_model import Lasso
    except Exception:
        selected, rows = select_corr_topk_features(lagged_by_feature, target_delta, target_feature, fallback_top_k)
        for row in rows:
            if row["selection_reason"] in {"forced_self", "corr_topk"}:
                row["selection_reason"] = f"fallback_corr_no_sklearn_{row['selection_reason']}"
        return selected, rows

    x_valid = x[valid]
    y_valid = y[valid]
    x_scaled, _, _ = _standardize_matrix(x_valid)
    y_centered = y_valid - np.nanmean(y_valid)
    model = Lasso(alpha=alpha, max_iter=20_000)
    model.fit(x_scaled, y_centered)
    coefficients = np.asarray(model.coef_, dtype=float)

    rows = []
    selected_scores = []
    for feature in features:
        coef = coefficients[feature_slices[feature]]
        score = float(np.nanmax(np.abs(coef))) if coef.size else 0.0
        selected_scores.append((feature, score))
        rows.append(
            {
                "feature": feature,
                "score": score,
                "best_lag_position": int(np.nanargmax(np.abs(coef)) + 1) if coef.size and score > 0 else -1,
                "selection_reason": "lasso_zero",
            }
        )

    nonzero = [feature for feature, score in selected_scores if score > 1e-8 and feature != target_feature]
    nonzero = sorted(nonzero, key=lambda feature: (-dict(selected_scores)[feature], _feature_order(feature)))
    if not nonzero:
        selected, fallback_rows = select_corr_topk_features(lagged_by_feature, target_delta, target_feature, fallback_top_k)
        for row in fallback_rows:
            if row["selection_reason"] in {"forced_self", "corr_topk"}:
                row["selection_reason"] = f"fallback_corr_no_lasso_nonzero_{row['selection_reason']}"
        return selected, fallback_rows

    selected = (target_feature, *nonzero)
    selected_set = set(selected)
    for row in rows:
        feature = str(row["feature"])
        if feature == target_feature:
            row["selection_reason"] = "forced_self"
        elif feature in selected_set:
            row["selection_reason"] = "lasso_nonzero"
    return selected, rows


def build_target_splits(
    data: pd.DataFrame,
    stations: tuple[str, ...],
    input_columns: tuple[str, ...],
    target_feature: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], base.GraphForecastScalers]:
    """[09] 构造单目标 delta 数据集并标准化。"""
    dataset = delta.build_delta_dataset(
        data,
        stations=stations,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        input_columns=input_columns,
        target_columns=(target_feature,),
        freq=paper.RESAMPLE_RULE,
    )
    raw_splits = lag.split_physical_lag_by_time(dataset, paper.TRAIN_END, paper.VAL_END)
    scaled_splits, scalers = lag.scale_physical_lag_splits(raw_splits)
    return raw_splits, scaled_splits, scalers


def flatten_lagged_training_signals(
    train_split: dict[str, np.ndarray],
    input_columns: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """[10] 将训练 split 展成每个指标一个 lag 矩阵，用于筛选。"""
    lagged = {}
    for feature in FEATURE_COLUMNS:
        column = f"{feature}_diff1"
        idx = input_columns.index(column)
        values = train_split["self_x"][..., idx]
        lagged[feature] = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    target_delta = train_split["y"][:, 0, :, 0].reshape(-1)
    return lagged, target_delta


def selected_features_for_experiment(
    experiment: str,
    spec: ExperimentSpec,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    target_feature: str,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """[11] 针对单个目标返回选中特征和记录行。"""
    if spec.selector == "self":
        return (target_feature,), [
            {
                "experiment": experiment,
                "target": target_feature,
                "feature": target_feature,
                "score": 1.0,
                "best_lag_position": 0,
                "selection_reason": "self_only",
                "selected": True,
            }
        ]

    all9_diff_columns = diff_columns_from_features(FEATURE_COLUMNS)
    raw_splits, _, _ = build_target_splits(data, stations, all9_diff_columns, target_feature)
    lagged, target_delta = flatten_lagged_training_signals(raw_splits["train"], all9_diff_columns)
    if spec.selector == "corr_top3":
        selected, rows = select_corr_topk_features(lagged, target_delta, target_feature, CORR_TOP_K)
    elif spec.selector == "lasso":
        selected, rows = select_lasso_features(lagged, target_delta, target_feature, LASSO_FALLBACK_TOP_K, LASSO_ALPHA)
    else:
        raise ValueError(f"Unsupported selector: {spec.selector}")

    selected_set = set(selected)
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "experiment": experiment,
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
    """[12] DataLoader 复用 physical-lag 格式。"""
    return lag.make_loader(torch, split, shuffle)


def collect_target_arrays(
    torch,
    model,
    loader,
    split: dict[str, np.ndarray],
    scalers: base.GraphForecastScalers,
    device,
) -> dict[str, np.ndarray]:
    """[13] 收集单目标绝对值预测、真值和 mask。"""
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
    """[14] 单目标模型按绝对值预测误差评分。"""
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
    device,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """[15] 训练一个单目标 delta-GRU，并返回预测数组。"""
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
    best_model_path = OUTPUT_DIR / f"{safe_filename(name)}_{safe_filename(target_feature)}_{INPUT_STEPS}to{OUTPUT_STEPS}_best.pt"
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
        console.print(
            f"{name}/{target_feature} epoch={epoch:03d} train_loss={train_loss:.6f} "
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
                    "target": target_feature,
                    "architecture": "per_target_self_gru_predict_delta_then_restore_absolute",
                    "input_columns": list(input_columns),
                    "target_columns": [target_feature],
                    "input_steps": INPUT_STEPS,
                    "output_steps": OUTPUT_STEPS,
                    "scalers": scalers.to_dict(),
                },
                best_model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                console.print(f"{name}/{target_feature} early_stop epoch={epoch:03d} best_val_rmse={best_rmse:.6f}", flush=True)
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
    return {
        "experiment": name,
        "target": target_feature,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["val_rmse"]),
        "best_checkpoint": metrics,
        "best_model_path": str(best_model_path),
        "input_columns": list(input_columns),
    }, arrays_by_split


def run_per_target_experiment(
    torch,
    name: str,
    spec: ExperimentSpec,
    data: pd.DataFrame,
    stations: tuple[str, ...],
    device,
) -> tuple[dict, list[dict[str, object]]]:
    """[16] 对五个目标分别筛特征、分别训练，再聚合成同口径指标。"""
    target_results = {}
    selected_feature_rows = []
    arrays_by_split_and_target = {split: {} for split in ["train", "val", "test"]}
    history_rows = []
    for target_feature in TARGET_FEATURE_COLUMNS:
        selected_features, selection_rows = selected_features_for_experiment(name, spec, data, stations, target_feature)
        selected_feature_rows.extend(selection_rows)
        input_columns = diff_columns_from_features(selected_features)
        _, scaled_splits, scalers = build_target_splits(data, stations, input_columns, target_feature)
        result, arrays_by_split = fit_target_delta_gru(
            torch, name, target_feature, input_columns, scaled_splits, scalers, stations, device
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
        "history": history_rows,
        "best_epoch": {
            "epoch": "",
            "val_rmse": metrics["val"].get("rmse"),
            "target_best_epochs": best_epochs,
        },
        "best_checkpoint": metrics,
        "targets": target_results,
    }, selected_feature_rows


def save_tables(
    results: dict[str, dict],
    stations: tuple[str, ...],
    persistence: dict[str, dict],
    selected_feature_rows: list[dict[str, object]],
) -> None:
    """[17] 保存结果表。"""
    pd.DataFrame(self_ablation.overall_rows(results, persistence)).to_csv(
        OUTPUT_DIR / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(self_ablation.feature_metric_rows(results)).to_csv(
        OUTPUT_DIR / "feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(self_ablation.station_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(self_ablation.station_feature_metric_rows(results, stations)).to_csv(
        OUTPUT_DIR / "station_feature_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_feature_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")
    history_rows = []
    for experiment, result in results.items():
        for row in result["history"]:
            history_rows.append({"experiment": experiment, **row})
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "history.csv", index=False, encoding="utf-8-sig")


def _json_safe_results(results: dict[str, dict]) -> dict[str, dict]:
    safe = {}
    for name, result in results.items():
        safe[name] = {
            "experiment": result["experiment"],
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


def write_report(results: dict[str, dict], persistence: dict[str, dict], selected_feature_rows: list[dict[str, object]]) -> None:
    """[18] 写简短报告。"""
    overall = pd.DataFrame(self_ablation.overall_rows(results, persistence))
    selected = pd.DataFrame(selected_feature_rows)
    selected_view = (
        selected[selected["selected"]]
        .groupby(["experiment", "target"], sort=False)["feature"]
        .apply(lambda values: "、".join(values.astype(str)))
        .reset_index(name="selected_features")
    )
    lines = [
        "# per-target diff-delta 特征筛选实验",
        "",
        "## 实验",
        "- per_target_self_diff_delta：每个目标只用自身 diff1。",
        "- per_target_corr_top3_delta：自身 diff1 + 训练集相关性 top3 diff1。",
        "- per_target_lasso_selected_delta：自身 diff1 + 训练集 Lasso 非零 diff1；无非零时回退相关性 top3。",
        "",
        "## 整体结果",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## 选中特征",
        "```text",
        selected_view.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(output_dir: Path = OUTPUT_DIR) -> int:
    """[19] 主流程。"""
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    random.seed(paper.SEED)
    np.random.seed(paper.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = tuple(station for station, _ in lag.STATION_FILES)
    data = delta.load_diff1_chain_data()
    all9_raw_splits, _, _ = delta.build_splits(data, stations, delta.input_columns_for_spec(delta.EXPERIMENT_SPECS["all9_diff_delta"]))
    dataset_summary = {
        "model": "per_target_same_station_diff_delta_feature_selection",
        "output_dir": str(OUTPUT_DIR),
        "start_date": paper.START_DATE,
        "train_end": paper.TRAIN_END,
        "val_end": paper.VAL_END,
        "resample_rule": paper.RESAMPLE_RULE,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "target_features": list(TARGET_FEATURE_COLUMNS),
        "stations": list(stations),
        "corr_top_k": CORR_TOP_K,
        "lasso_alpha": LASSO_ALPHA,
        "experiments": {name: spec.description for name, spec in EXPERIMENT_SPECS.items()},
        "split_summary": lag.graph_split_summary(all9_raw_splits),
    }
    base.save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    console.print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)

    torch = lag.require_torch()
    device = base.choose_device(torch)
    console.print(f"device={device}", flush=True)
    results = {}
    selected_feature_rows = []
    for name, spec in EXPERIMENT_SPECS.items():
        result, rows = run_per_target_experiment(torch, name, spec, data, stations, device)
        results[name] = result
        selected_feature_rows.extend(rows)

    persistence = delta.evaluate_persistence_baseline(all9_raw_splits, stations)
    metrics = {
        "config": dataset_summary,
        "persistence_baseline": persistence,
        "experiments": _json_safe_results(results),
        "selected_features": selected_feature_rows,
    }
    base.save_json(OUTPUT_DIR / "metrics.json", metrics)
    save_tables(results, stations, persistence, selected_feature_rows)
    write_report(results, persistence, selected_feature_rows)
    console.print(pd.DataFrame(self_ablation.overall_rows(results, persistence)).to_string(index=False), flush=True)
    console.print(pd.DataFrame(self_ablation.feature_metric_rows(results)).to_string(index=False), flush=True)
    console.print(pd.DataFrame(selected_feature_rows).query("selected == True").to_string(index=False), flush=True)
    return 0


def main() -> int:
    return run_suite(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
