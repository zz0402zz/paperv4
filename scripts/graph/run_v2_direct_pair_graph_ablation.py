#!/usr/bin/env python3
"""Run the V2 direct-pair graph-message pilot ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.graph import v2_direct_edge_message as message
from scripts.graph import v2_direct_pair_graph_config as cfg
from scripts.common import v2_experiment_protocol as protocol
from scripts.graph import v2_physical_lag
from scripts.graph import v2_upstream_change_forecaster as upstream

REQUIRED_VARIANTS = (
    "self_D_6to9",
    "strict_graph_physical",
    "strict_graph_fast_lag",
    "strict_graph_slow_lag",
    "strict_graph_no_delay",
    "strict_graph_shuffled_time",
    "reverse_graph",
    "wrong_source_graph",
    "oracle_upper_bound",
)
PILOT_VARIANTS = (
    "self_D_6to9",
    "strict_graph_physical",
    "strict_graph_shuffled_time",
    "wrong_source_graph",
)
SELECTION_SPLIT = "val"
MAX_EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


@dataclass
class SimpleScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "SimpleScaler":
        flat = np.asarray(values, dtype=float).reshape(-1, values.shape[-1])
        mean = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        mean[~np.isfinite(mean)] = 0.0
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=float) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) * self.scale + self.mean).astype(np.float32)


def _safe_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_v2_data() -> pd.DataFrame:
    data = pd.read_csv(protocol.OBSERVED_DATA_PATH)
    data["station"] = data["station"].astype(str)
    data["time"] = pd.to_datetime(data["time"])
    data = data[data["time"].ge(pd.Timestamp(protocol.START_DATE))].copy()
    data = data.sort_values(["station", "time"])
    for feature in protocol.INPUT_FEATURE_COLUMNS:
        data[f"{feature}_diff1"] = data.groupby("station", sort=False)[feature].diff()
    return data


def station_panel(data: pd.DataFrame, station: str) -> pd.DataFrame:
    columns = ["time", *protocol.INPUT_FEATURE_COLUMNS, *(f"{f}_diff1" for f in protocol.INPUT_FEATURE_COLUMNS)]
    frame = data.loc[data["station"].eq(station), columns].copy()
    if frame.empty:
        raise ValueError(f"Missing station data: {station}")
    return frame.sort_values("time").reset_index(drop=True)


def _finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(array).all() for array in arrays)


def build_pair_samples(
    data: pd.DataFrame,
    source_station: str,
    target_station: str,
    lag_steps: int,
) -> dict[str, np.ndarray]:
    source = station_panel(data, source_station)
    target = station_panel(data, target_station)
    merged = target.merge(source, on="time", suffixes=("_down", "_up"))

    feature_count = len(protocol.INPUT_FEATURE_COLUMNS)
    target_count = len(protocol.TARGET_FEATURE_COLUMNS)
    source_diff_cols = [f"{feature}_diff1" for feature in protocol.INPUT_FEATURE_COLUMNS]
    target_diff_cols = [f"{feature}_diff1_down" for feature in protocol.INPUT_FEATURE_COLUMNS]
    target_level_cols = [f"{feature}_down" for feature in protocol.TARGET_FEATURE_COLUMNS]
    source_diff = merged[[f"{col}_up" for col in source_diff_cols]].to_numpy(float)
    target_diff = merged[target_diff_cols].to_numpy(float)
    target_level = merged[target_level_cols].to_numpy(float)

    times = pd.to_datetime(merged["time"]).to_numpy(dtype="datetime64[ns]")
    rows: dict[str, list] = {
        "self_x": [],
        "current": [],
        "y": [],
        "anchor": [],
        "source_x": [],
        "source_y": [],
        "aligned_observed": [],
        "aligned_oracle": [],
        "origin_time": [],
        "target_time": [],
    }
    min_origin = max(cfg.INPUT_STEPS, int(lag_steps))
    last_origin = len(merged) - cfg.OUTPUT_STEPS - 1
    for origin in range(min_origin, last_origin + 1):
        self_x = target_diff[origin - cfg.INPUT_STEPS + 1 : origin + 1]
        current = target_level[origin]
        future_level = target_level[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        y = future_level - current[None, :]
        source_x = source_diff[origin - cfg.INPUT_STEPS + 1 : origin + 1]
        source_y = source_diff[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        aligned = []
        aligned_oracle = []
        source_values = source_diff
        for horizon in range(1, cfg.OUTPUT_STEPS + 1):
            source_step = horizon - int(lag_steps)
            index = origin + source_step
            if index < 0 or index >= len(source_values):
                aligned.append(np.full(feature_count, np.nan))
                aligned_oracle.append(np.full(feature_count, np.nan))
            else:
                value = source_values[index]
                aligned_oracle.append(value)
                aligned.append(value if source_step <= 0 else np.full(feature_count, np.nan))
        aligned = np.asarray(aligned, dtype=float)
        aligned_oracle = np.asarray(aligned_oracle, dtype=float)
        if not _finite(self_x, current, y, source_x, source_y):
            continue
        if not _finite(aligned[np.isfinite(aligned).all(axis=1)] if np.isfinite(aligned).any() else np.zeros((0, feature_count))):
            continue
        if not _finite(aligned_oracle):
            continue
        rows["self_x"].append(self_x)
        rows["current"].append(current)
        rows["y"].append(y)
        rows["anchor"].append(current)
        rows["source_x"].append(source_x)
        rows["source_y"].append(source_y)
        rows["aligned_observed"].append(aligned)
        rows["aligned_oracle"].append(aligned_oracle)
        rows["origin_time"].append(times[origin])
        rows["target_time"].append(times[origin + 1 : origin + cfg.OUTPUT_STEPS + 1])

    if not rows["self_x"]:
        raise ValueError(f"No finite samples for {source_station} -> {target_station}")
    return {
        key: np.asarray(value)
        for key, value in rows.items()
    }


def subset_samples_by_origin(samples: dict[str, np.ndarray], origin_times: np.ndarray) -> dict[str, np.ndarray]:
    """Align a source-sample set to the reference target-origin times."""
    lookup = {pd.Timestamp(value).to_datetime64(): idx for idx, value in enumerate(samples["origin_time"])}
    indices = []
    for time in origin_times:
        key = pd.Timestamp(time).to_datetime64()
        if key not in lookup:
            raise ValueError("Wrong-source samples do not cover the same forecast origins.")
        indices.append(lookup[key])
    selected = np.asarray(indices, dtype=int)
    return {
        key: value[selected] if isinstance(value, np.ndarray) and len(value) == len(samples["origin_time"]) else value
        for key, value in samples.items()
    }


def common_origin_times(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> np.ndarray:
    """Sorted forecast origins present in both sample sets."""
    left_times = {pd.Timestamp(value).to_datetime64() for value in left["origin_time"]}
    right_times = {pd.Timestamp(value).to_datetime64() for value in right["origin_time"]}
    common = sorted(left_times & right_times)
    if not common:
        raise ValueError("No common forecast origins between real and wrong-source samples.")
    return np.asarray(common, dtype="datetime64[ns]")


def choose_wrong_source_samples(
    data: pd.DataFrame,
    target: str,
    lag_steps: int,
    stations: tuple[str, ...],
    strict_edges: list[tuple[str, str]],
    reference_samples: dict[str, np.ndarray],
) -> tuple[str, dict[str, np.ndarray], np.ndarray]:
    """Choose a topology-negative source with maximum common target-window coverage."""
    component = message.upstream_component(target, strict_edges)
    best: tuple[int, str, dict[str, np.ndarray], np.ndarray] | None = None
    for candidate in stations:
        if candidate in component or candidate == target:
            continue
        try:
            candidate_samples = build_pair_samples(data, candidate, target, lag_steps)
            common = common_origin_times(reference_samples, candidate_samples)
        except Exception:
            continue
        score = int(len(common))
        if best is None or score > best[0]:
            best = (score, candidate, candidate_samples, common)
    if best is None:
        raise ValueError(f"No wrong-source candidate with common coverage for {target}")
    _, candidate, candidate_samples, common = best
    return candidate, subset_samples_by_origin(candidate_samples, common), common


def split_indices(origin_time: np.ndarray) -> dict[str, np.ndarray]:
    times = pd.to_datetime(origin_time)
    train = times < pd.Timestamp(protocol.TRAIN_END)
    val = (times >= pd.Timestamp(protocol.TRAIN_END)) & (times < pd.Timestamp(protocol.VAL_END))
    test = times >= pd.Timestamp(protocol.VAL_END)
    return {"train": np.flatnonzero(train), "val": np.flatnonzero(val), "test": np.flatnonzero(test)}


def _loader(torch, arrays: tuple[np.ndarray, ...], shuffle: bool):
    tensors = [torch.as_tensor(array, dtype=torch.float32) for array in arrays]
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle)


def fit_scalers(samples: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, SimpleScaler]:
    return {
        "self_x": SimpleScaler.fit(samples["self_x"][train_idx]),
        "current": SimpleScaler.fit(samples["current"][train_idx]),
        "y": SimpleScaler.fit(samples["y"][train_idx]),
        "source_x": SimpleScaler.fit(samples["source_x"][train_idx]),
        "source_y": SimpleScaler.fit(samples["source_y"][train_idx]),
        "aligned": SimpleScaler.fit(samples["source_y"][train_idx]),
    }


def transform_samples(samples: dict[str, np.ndarray], scalers: dict[str, SimpleScaler]) -> dict[str, np.ndarray]:
    return {
        **samples,
        "self_x_scaled": scalers["self_x"].transform(samples["self_x"]),
        "current_scaled": scalers["current"].transform(samples["current"]),
        "y_scaled": scalers["y"].transform(samples["y"]),
        "source_x_scaled": scalers["source_x"].transform(samples["source_x"]),
        "source_y_scaled": scalers["source_y"].transform(samples["source_y"]),
    }


def train_upstream_model(torch, samples: dict[str, np.ndarray], split: dict[str, np.ndarray], device):
    model = upstream.make_upstream_change_forecaster(torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = _loader(torch, (samples["source_x_scaled"][split["train"]], samples["source_y_scaled"][split["train"]]), True)
    val_loader = _loader(torch, (samples["source_x_scaled"][split["val"]], samples["source_y_scaled"][split["val"]]), False)
    best_state = None
    best_val = float("inf")
    bad = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device))
            loss = torch.nn.functional.l1_loss(pred, y.to(device))
            loss.backward()
            optimizer.step()
        val = _evaluate_upstream_loss(torch, model, val_loader, device)
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate_upstream_loss(torch, model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            losses.append(float(torch.nn.functional.l1_loss(model(x.to(device)), y.to(device)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict_upstream(torch, model, samples: dict[str, np.ndarray], device) -> np.ndarray:
    model.eval()
    loader = _loader(torch, (samples["source_x_scaled"],), False)
    preds = []
    with torch.no_grad():
        for (x,) in loader:
            preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_aligned_scaled(
    samples: dict[str, np.ndarray],
    predicted_source_y_scaled: np.ndarray,
    lag_steps: int,
    scaler: SimpleScaler,
    variant: str,
    seed: int,
) -> np.ndarray:
    aligned = []
    observed_scaled = scaler.transform(samples["aligned_observed"])
    oracle_scaled = scaler.transform(samples["aligned_oracle"])
    for row_idx in range(len(samples["self_x"])):
        horizons = []
        for h in range(cfg.OUTPUT_STEPS):
            source_step = (h + 1) - int(lag_steps)
            if variant == "oracle_upper_bound":
                horizons.append(oracle_scaled[row_idx, h])
            elif source_step <= 0:
                horizons.append(observed_scaled[row_idx, h])
            else:
                horizons.append(predicted_source_y_scaled[row_idx, source_step - 1])
        aligned.append(horizons)
    output = np.asarray(aligned, dtype=np.float32)
    if variant == "strict_graph_shuffled_time":
        output = message.block_shuffle(output, block_steps=6, seed=seed)
    return output


def train_self_model(torch, samples: dict[str, np.ndarray], split: dict[str, np.ndarray], device):
    model = message.DirectEdgeMessageModel(
        sequence_input_dim=samples["self_x_scaled"].shape[-1],
        current_input_dim=samples["current_scaled"].shape[-1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: _loader(
            torch,
            (
                samples["self_x_scaled"][idx],
                samples["current_scaled"][idx],
                np.zeros((len(idx), cfg.OUTPUT_STEPS, len(protocol.INPUT_FEATURE_COLUMNS)), dtype=np.float32),
                samples["y_scaled"][idx],
            ),
            name == "train",
        )
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    bad = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for self_x, current, aligned, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            pred = model(self_x.to(device), current.to(device), aligned.to(device), graph_enabled=False)
            loss = torch.nn.functional.l1_loss(pred, y.to(device))
            loss.backward()
            optimizer.step()
        val = _evaluate_model_loss(torch, model, loaders["val"], device, graph_enabled=False)
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_edge_model(torch, self_model, samples: dict[str, np.ndarray], aligned_scaled: np.ndarray, split: dict[str, np.ndarray], device):
    model = message.DirectEdgeMessageModel(
        sequence_input_dim=samples["self_x_scaled"].shape[-1],
        current_input_dim=samples["current_scaled"].shape[-1],
    ).to(device)
    model.load_state_dict(self_model.state_dict())
    for name, parameter in model.named_parameters():
        if not name.startswith("edge_mapper."):
            parameter.requires_grad = False
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    loaders = {
        name: _loader(
            torch,
            (
                samples["self_x_scaled"][idx],
                samples["current_scaled"][idx],
                aligned_scaled[idx],
                samples["y_scaled"][idx],
            ),
            name == "train",
        )
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    bad = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for self_x, current, aligned, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            pred, parts = model(self_x.to(device), current.to(device), aligned.to(device), return_parts=True)
            loss = message.graph_loss(torch, pred, y.to(device), torch.ones_like(y, dtype=torch.bool).to(device), parts["graph_delta"])
            loss.backward()
            optimizer.step()
        val = _evaluate_model_loss(torch, model, loaders["val"], device, graph_enabled=True)
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate_model_loss(torch, model, loader, device, graph_enabled: bool) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for self_x, current, aligned, y in loader:
            pred = model(self_x.to(device), current.to(device), aligned.to(device), graph_enabled=graph_enabled)
            losses.append(float(torch.nn.functional.l1_loss(pred, y.to(device)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def collect_predictions(torch, model, samples: dict[str, np.ndarray], aligned_scaled: np.ndarray, split_idx: np.ndarray, scalers: dict[str, SimpleScaler], device, graph_enabled: bool):
    loader = _loader(
        torch,
        (
            samples["self_x_scaled"][split_idx],
            samples["current_scaled"][split_idx],
            aligned_scaled[split_idx],
        ),
        False,
    )
    preds = []
    graph_parts = []
    model.eval()
    with torch.no_grad():
        for self_x, current, aligned in loader:
            pred, parts = model(
                self_x.to(device),
                current.to(device),
                aligned.to(device),
                graph_enabled=graph_enabled,
                return_parts=True,
            )
            preds.append(pred.cpu().numpy())
            graph_parts.append(parts["graph_delta"].cpu().numpy())
    pred_delta = scalers["y"].inverse(np.concatenate(preds, axis=0))
    graph_delta = scalers["y"].inverse(np.concatenate(graph_parts, axis=0)) - scalers["y"].mean
    truth_delta = samples["y"][split_idx]
    anchor = samples["anchor"][split_idx]
    return {
        "pred_delta": pred_delta,
        "truth_delta": truth_delta,
        "pred_abs": anchor[:, None, :] + pred_delta,
        "truth_abs": anchor[:, None, :] + truth_delta,
        "graph_delta": graph_delta,
    }


def metric_rows(source: str, target: str, variant: str, split_name: str, arrays: dict[str, np.ndarray], seed: int) -> list[dict[str, object]]:
    rows = []
    error = arrays["pred_abs"] - arrays["truth_abs"]
    for h in range(cfg.OUTPUT_STEPS):
        for f_idx, feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
            e = error[:, h, f_idx]
            truth = arrays["truth_abs"][:, h, f_idx]
            denom = float(np.sum((truth - np.mean(truth)) ** 2))
            rows.append(
                {
                    "seed": seed,
                    "source_station": source,
                    "target_station": target,
                    "variant": variant,
                    "split": split_name,
                    "feature": feature,
                    "horizon_step": h + 1,
                    "horizon_hours": (h + 1) * 4,
                    "valid_points": int(len(e)),
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "nse": None if denom <= np.finfo(float).eps else float(1.0 - np.sum(e**2) / denom),
                }
            )
    return rows


def strict_edges() -> list[tuple[str, str]]:
    frame = pd.read_csv(protocol.STRICT_EDGES_PATH)
    return list(frame[["source_station", "target_station"]].itertuples(index=False, name=None))


def run_pair_variant_suite(torch, data: pd.DataFrame, lag_row: pd.Series, seed: int, device) -> list[dict[str, object]]:
    source = str(lag_row["source_station"])
    target = str(lag_row["target_station"])
    lag_steps = int(lag_row["lag_primary_steps"])
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    edge_list = strict_edges()
    samples = build_pair_samples(data, source, target, lag_steps)
    wrong_source, wrong_samples, common_times = choose_wrong_source_samples(
        data,
        target,
        lag_steps,
        stations,
        edge_list,
        samples,
    )
    samples = subset_samples_by_origin(samples, common_times)
    split = split_indices(samples["origin_time"])
    scalers = fit_scalers(samples, split["train"])
    samples = transform_samples(samples, scalers)
    self_model = train_self_model(torch, samples, split, device)
    source_model = train_upstream_model(torch, samples, split, device)
    source_pred = predict_upstream(torch, source_model, samples, device)
    zero_aligned = np.zeros((len(samples["self_x"]), cfg.OUTPUT_STEPS, len(protocol.INPUT_FEATURE_COLUMNS)), dtype=np.float32)
    rows = []

    variant_models = {"self_D_6to9": (self_model, zero_aligned, False, source)}
    for variant in ("strict_graph_physical", "strict_graph_shuffled_time"):
        aligned = build_aligned_scaled(samples, source_pred, lag_steps, scalers["aligned"], variant, seed)
        variant_models[variant] = (train_edge_model(torch, self_model, samples, aligned, split, device), aligned, True, source)

    wrong_samples = transform_samples(wrong_samples, scalers)
    wrong_model = train_upstream_model(torch, wrong_samples, split, device)
    wrong_pred = predict_upstream(torch, wrong_model, wrong_samples, device)
    wrong_aligned = build_aligned_scaled(wrong_samples, wrong_pred, lag_steps, scalers["aligned"], "wrong_source_graph", seed)
    variant_models["wrong_source_graph"] = (
        train_edge_model(torch, self_model, samples, wrong_aligned, split, device),
        wrong_aligned,
        True,
        wrong_source,
    )

    for variant, (model, aligned, graph_enabled, used_source) in variant_models.items():
        for split_name, idx in split.items():
            arrays = collect_predictions(torch, model, samples, aligned, idx, scalers, device, graph_enabled)
            rows.extend(metric_rows(used_source, target, variant, split_name, arrays, seed))
    return rows


def summarize_overall(feature_horizon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in feature_horizon.groupby(["seed", "source_station", "target_station", "variant", "split"], sort=True):
        err_proxy = pd.to_numeric(group["rmse"], errors="coerce")
        rows.append(
            {
                "seed": keys[0],
                "source_station": keys[1],
                "target_station": keys[2],
                "variant": keys[3],
                "split": keys[4],
                "mean_mae": float(pd.to_numeric(group["mae"], errors="coerce").mean()),
                "mean_rmse": float(err_proxy.mean()),
                "macro_nse": float(pd.to_numeric(group["nse"], errors="coerce").mean()),
                "valid_points_total": int(pd.to_numeric(group["valid_points"], errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def run_pilot(seed: int = cfg.PILOT_SEED) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = cfg.PILOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_audit = v2_physical_lag.write_lag_audit()
    data = load_v2_data()
    rows = []
    for lag_row in lag_audit.itertuples(index=False):
        console.print(f"running pair {lag_row.source_station}->{lag_row.target_station} lag={lag_row.lag_primary_steps}", flush=True)
        rows.extend(run_pair_variant_suite(torch, data, pd.Series(lag_row._asdict()), seed, device))
    feature_horizon = pd.DataFrame(rows)
    overall = summarize_overall(feature_horizon)
    feature_horizon.to_csv(output_dir / "pair_feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "pair_overall_metrics.csv", index=False, encoding="utf-8-sig")
    manifest = protocol.build_run_manifest(
        experiment="stage4_direct_pair_graph_pilot",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/graph/run_v2_direct_pair_graph_ablation.py"),
            Path("scripts/graph/v2_direct_edge_message.py"),
            Path("scripts/graph/v2_upstream_change_forecaster.py"),
            Path("scripts/graph/v2_physical_lag.py"),
        ),
    )
    manifest.update(
        {
            "input_steps": cfg.INPUT_STEPS,
            "output_steps": cfg.OUTPUT_STEPS,
            "pilot_variants": list(PILOT_VARIANTS),
            "required_variants": list(REQUIRED_VARIANTS),
            "selection_split": SELECTION_SPLIT,
            "device": str(device),
        }
    )
    _safe_json(output_dir / "run_manifest.json", manifest)
    write_pilot_report(output_dir, overall)
    return output_dir


def write_pilot_report(output_dir: Path, overall: pd.DataFrame) -> None:
    val = overall[overall["split"].eq("val")].sort_values(["target_station", "mean_rmse"])
    test = overall[overall["split"].eq("test")].sort_values(["target_station", "mean_rmse"])
    lines = [
        "# V2 Stage 4 Direct-Pair Graph Pilot",
        "",
        "- Input: 24h self history; output: 4-36h direct changes.",
        "- Graph message: aligned upstream nine-feature changes mapped to five target corrections.",
        "- Selection split: validation only.",
        "",
        "## Validation Overall",
        "```text",
        val.to_string(index=False),
        "```",
        "",
        "## Test Overall",
        "```text",
        test.to_string(index=False),
        "```",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.PILOT_SEED)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.formal:
        raise SystemExit("Formal multiseed mode is gated until the pilot audit is reviewed.")
    run_pilot(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
