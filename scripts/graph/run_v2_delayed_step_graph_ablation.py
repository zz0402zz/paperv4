#!/usr/bin/env python3
"""Run the V2 delayed step-change graph ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.graph import run_v2_direct_pair_graph_ablation as stage4
from scripts.graph import v2_delayed_step_graph as delayed_graph
from scripts.graph import v2_delayed_step_graph_config as cfg
from scripts.common import v2_experiment_protocol as protocol
from scripts.graph import v2_physical_lag


REQUIRED_VARIANTS = (
    "self_D_6to9",
    "point_graph_36h",
    "step_graph_24h",
    "step_graph_36h",
    "step_graph_36h_observed_only",
    "step_graph_36h_shuffled",
    "step_graph_36h_downstream_source",
)
SELECTION_SPLIT = "val"
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TRANSFER_L1 = 1e-4
SOURCE_HIDDEN_SIZE = 64


def make_source_forecaster(torch, feature_count: int, hidden_size: int, output_steps: int):
    """Create a GRU that predicts future one-step changes for all source features."""

    class _SourceForecaster(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_steps = int(output_steps)
            self.feature_count = int(feature_count)
            self.encoder = torch.nn.GRU(self.feature_count, int(hidden_size), batch_first=True)
            self.head = torch.nn.Linear(int(hidden_size), self.output_steps * self.feature_count)

        def forward(self, source_history):
            _, hidden = self.encoder(source_history)
            return self.head(hidden[-1]).reshape(-1, self.output_steps, self.feature_count)

    return _SourceForecaster()


def build_aligned_batch(
    observed_history: np.ndarray,
    predicted_future: np.ndarray,
    support: tuple[int, ...],
    output_steps: int = cfg.OUTPUT_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    histories = np.asarray(observed_history, dtype=float)
    futures = np.asarray(predicted_future, dtype=float)
    if histories.ndim != 3 or futures.ndim != 3 or len(histories) != len(futures):
        raise ValueError("observed_history and predicted_future must be aligned batched arrays")
    aligned_rows = []
    known_rows = []
    for history, future in zip(histories, futures):
        aligned, known = delayed_graph.align_lag_support(history, future, support, output_steps)
        aligned_rows.append(aligned)
        known_rows.append(known)
    return np.asarray(aligned_rows, dtype=np.float32), np.asarray(known_rows, dtype=bool)


def add_diff_features(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["station"] = work["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    work = work.sort_values(["station", "time"]).reset_index(drop=True)
    for feature in protocol.INPUT_FEATURE_COLUMNS:
        work[f"{feature}_diff1"] = work.groupby("station", sort=False)[feature].diff()
    return work


def _station_frame(data: pd.DataFrame, station: str, suffix: str) -> pd.DataFrame:
    diff_columns = [f"{feature}_diff1" for feature in protocol.INPUT_FEATURE_COLUMNS]
    columns = ["time", *protocol.INPUT_FEATURE_COLUMNS, *diff_columns]
    frame = data.loc[data["station"].eq(station), columns].copy()
    if frame.empty:
        raise ValueError(f"Missing station data: {station}")
    rename = {column: f"{column}_{suffix}" for column in columns if column != "time"}
    return frame.rename(columns=rename).sort_values("time")


def build_pair_samples(
    data: pd.DataFrame,
    source_station: str,
    target_station: str,
    control_station: str,
) -> dict[str, np.ndarray]:
    """Build one common set of target origins for strict and direction controls."""
    target = _station_frame(data, target_station, "target")
    source = _station_frame(data, source_station, "source")
    control = _station_frame(data, control_station, "control")
    merged = target.merge(source, on="time", how="inner").merge(control, on="time", how="inner")

    target_diff_columns = [f"{feature}_diff1_target" for feature in protocol.INPUT_FEATURE_COLUMNS]
    source_diff_columns = [f"{feature}_diff1_source" for feature in protocol.INPUT_FEATURE_COLUMNS]
    control_diff_columns = [f"{feature}_diff1_control" for feature in protocol.INPUT_FEATURE_COLUMNS]
    target_level_columns = [f"{feature}_target" for feature in protocol.TARGET_FEATURE_COLUMNS]

    target_diff = merged[target_diff_columns].to_numpy(float)
    source_diff = merged[source_diff_columns].to_numpy(float)
    control_diff = merged[control_diff_columns].to_numpy(float)
    target_level = merged[target_level_columns].to_numpy(float)
    times = pd.to_datetime(merged["time"]).to_numpy(dtype="datetime64[ns]")

    rows: dict[str, list[np.ndarray | np.datetime64]] = {
        "self_x": [],
        "current": [],
        "target_delta": [],
        "anchor": [],
        "source_x_24h": [],
        "source_x_36h": [],
        "source_step_y": [],
        "control_x_36h": [],
        "control_step_y": [],
        "origin_time": [],
    }
    max_history = max(cfg.GRAPH_INPUT_STEPS)
    for origin in range(max_history - 1, len(merged) - cfg.OUTPUT_STEPS):
        self_x = target_diff[origin - cfg.SELF_INPUT_STEPS + 1 : origin + 1]
        source_x_36h = source_diff[origin - max_history + 1 : origin + 1]
        source_x_24h = source_x_36h[-cfg.GRAPH_INPUT_STEPS[0] :]
        control_x_36h = control_diff[origin - max_history + 1 : origin + 1]
        current = target_level[origin]
        future_target = target_level[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        target_delta = future_target - current[None, :]
        source_step_y = source_diff[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        control_step_y = control_diff[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        arrays = (
            self_x,
            source_x_36h,
            source_x_24h,
            control_x_36h,
            current,
            target_delta,
            source_step_y,
            control_step_y,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            continue
        rows["self_x"].append(self_x)
        rows["current"].append(current)
        rows["target_delta"].append(target_delta)
        rows["anchor"].append(current)
        rows["source_x_24h"].append(source_x_24h)
        rows["source_x_36h"].append(source_x_36h)
        rows["source_step_y"].append(source_step_y)
        rows["control_x_36h"].append(control_x_36h)
        rows["control_step_y"].append(control_step_y)
        rows["origin_time"].append(times[origin])
    if not rows["self_x"]:
        raise ValueError(f"No finite common samples for {source_station} -> {target_station}")
    return {key: np.asarray(value) for key, value in rows.items()}


def split_indices(origin_time: np.ndarray) -> dict[str, np.ndarray]:
    times = pd.to_datetime(origin_time)
    return {
        "train": np.flatnonzero(times < pd.Timestamp(protocol.TRAIN_END)),
        "val": np.flatnonzero((times >= pd.Timestamp(protocol.TRAIN_END)) & (times < pd.Timestamp(protocol.VAL_END))),
        "test": np.flatnonzero(times >= pd.Timestamp(protocol.VAL_END)),
    }


def load_v2_data() -> pd.DataFrame:
    data = pd.read_csv(protocol.OBSERVED_DATA_PATH)
    quality = pd.read_csv(protocol.QUALITY_DATA_PATH)
    data["station"] = data["station"].astype(str)
    data["time"] = pd.to_datetime(data["time"])
    data = data[data["time"].ge(pd.Timestamp(protocol.START_DATE))].copy()
    quality["station"] = quality["station"].astype(str)
    quality["time"] = pd.to_datetime(quality["time"])
    data = apply_soft_suspect_mask(data, quality)
    return add_diff_features(data)


def apply_soft_suspect_mask(data: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Exclude quality-sidecar soft suspects from Stage 4b inputs and labels."""
    work = data.copy()
    flags = quality.copy()
    work["station"] = work["station"].astype(str)
    flags["station"] = flags["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    flags["time"] = pd.to_datetime(flags["time"])
    flag_columns = [
        f"{feature}__soft_suspect"
        for feature in protocol.INPUT_FEATURE_COLUMNS
        if f"{feature}__soft_suspect" in flags.columns and feature in work.columns
    ]
    selected = flags[["station", "time", *flag_columns]]
    if selected.duplicated(["station", "time"]).any():
        raise ValueError("quality sidecar has duplicate station-time rows")
    merged = work.merge(selected, on=["station", "time"], how="left", validate="many_to_one", sort=False)
    for flag_column in flag_columns:
        feature = flag_column.removesuffix("__soft_suspect")
        merged[feature] = pd.to_numeric(merged[feature], errors="coerce").mask(
            merged[flag_column].fillna(False).astype(bool)
        )
    return merged.drop(columns=flag_columns)


def build_stage4b_lag_audit() -> pd.DataFrame:
    audit = pd.read_csv(v2_physical_lag.EDGE_AUDIT_PATH)
    rows = []
    for source, target in cfg.PAIRS:
        match = audit[
            audit["source_station"].astype(str).eq(source)
            & audit["target_station"].astype(str).eq(target)
        ]
        if match.empty:
            raise ValueError(f"Missing audited distance for {source} -> {target}")
        row = match.iloc[0]
        rows.append(
            {
                "source_station": source,
                "target_station": target,
                "relation": row.get("draft_relation", ""),
                "distance_km": float(row["straight_distance_km"]),
                "distance_kind": "straight_distance_km",
            }
        )
    lag_audit = v2_physical_lag.build_lag_audit(
        pd.DataFrame(rows),
        v2_physical_lag.load_velocity_observations(),
    )
    v2_physical_lag.validate_lag_audit(lag_audit)
    cfg.ensure_output_dirs()
    lag_audit.to_csv(cfg.LAG_AUDIT_PATH, index=False, encoding="utf-8-sig")
    return lag_audit


def fit_scalers(samples: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, stage4.SimpleScaler]:
    source_train = np.concatenate(
        [samples["source_x_36h"][train_idx], samples["source_step_y"][train_idx]],
        axis=1,
    )
    control_train = np.concatenate(
        [samples["control_x_36h"][train_idx], samples["control_step_y"][train_idx]],
        axis=1,
    )
    return {
        "self_x": stage4.SimpleScaler.fit(samples["self_x"][train_idx]),
        "current": stage4.SimpleScaler.fit(samples["current"][train_idx]),
        "target": stage4.SimpleScaler.fit(samples["target_delta"][train_idx]),
        "source": stage4.SimpleScaler.fit(source_train),
        "control": stage4.SimpleScaler.fit(control_train),
    }


def transform_samples(
    samples: dict[str, np.ndarray],
    scalers: dict[str, stage4.SimpleScaler],
) -> dict[str, np.ndarray]:
    return {
        **samples,
        "self_x_scaled": scalers["self_x"].transform(samples["self_x"]),
        "current_scaled": scalers["current"].transform(samples["current"]),
        "target_scaled": scalers["target"].transform(samples["target_delta"]),
        "source_x_24h_scaled": scalers["source"].transform(samples["source_x_24h"]),
        "source_x_36h_scaled": scalers["source"].transform(samples["source_x_36h"]),
        "source_step_y_scaled": scalers["source"].transform(samples["source_step_y"]),
        "control_x_36h_scaled": scalers["control"].transform(samples["control_x_36h"]),
        "control_step_y_scaled": scalers["control"].transform(samples["control_step_y"]),
    }


def train_source_model(
    torch,
    source_x: np.ndarray,
    source_y: np.ndarray,
    split: dict[str, np.ndarray],
    device,
):
    model = make_source_forecaster(
        torch,
        feature_count=source_x.shape[-1],
        hidden_size=SOURCE_HIDDEN_SIZE,
        output_steps=cfg.OUTPUT_STEPS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: stage4._loader(torch, (source_x[idx], source_y[idx]), name == "train")
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    wait = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        val_loss = evaluate_source_loss(torch, model, loaders["val"], device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_source_loss(torch, model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            losses.append(float(torch.nn.functional.l1_loss(model(x.to(device)), y.to(device)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict_source(torch, model, source_x: np.ndarray, device) -> np.ndarray:
    loader = stage4._loader(torch, (source_x,), False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_self_scaled(torch, self_model, samples: dict[str, np.ndarray], device) -> np.ndarray:
    aligned_zero = np.zeros(
        (len(samples["self_x"]), cfg.OUTPUT_STEPS, len(protocol.INPUT_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    loader = stage4._loader(
        torch,
        (samples["self_x_scaled"], samples["current_scaled"], aligned_zero),
        False,
    )
    predictions = []
    self_model.eval()
    with torch.no_grad():
        for self_x, current, aligned in loader:
            predictions.append(
                self_model(
                    self_x.to(device),
                    current.to(device),
                    aligned.to(device),
                    graph_enabled=False,
                ).cpu().numpy()
            )
    return np.concatenate(predictions, axis=0)


def shuffle_aligned_by_split(
    aligned: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    shuffled = np.asarray(aligned).copy()
    for offset, name in enumerate(("train", "val", "test")):
        idx = split[name]
        shuffled[idx] = stage4.message.block_shuffle(aligned[idx], block_steps=6, seed=seed + offset)
    return shuffled


def train_graph_mapper(
    torch,
    self_prediction: np.ndarray,
    aligned_source: np.ndarray,
    target: np.ndarray,
    split: dict[str, np.ndarray],
    device,
    accumulate: bool,
):
    mapper = delayed_graph.make_delayed_step_mapper(
        torch,
        source_dim=aligned_source.shape[-1],
        target_dim=target.shape[-1],
        lag_count=aligned_source.shape[-2],
    ).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: stage4._loader(
            torch,
            (self_prediction[idx], aligned_source[idx], target[idx]),
            name == "train",
        )
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    wait = 0
    for _epoch in range(MAX_EPOCHS):
        mapper.train()
        for self_pred, aligned, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            _, correction = mapper(aligned.to(device), accumulate=accumulate)
            final = self_pred.to(device) + correction
            loss = torch.nn.functional.l1_loss(final, y.to(device))
            loss = loss + TRANSFER_L1 * mapper.transfer.weight.abs().mean()
            loss.backward()
            optimizer.step()
        val_loss = evaluate_graph_loss(torch, mapper, loaders["val"], device, accumulate)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        mapper.load_state_dict(best_state)
    return mapper


def evaluate_graph_loss(torch, mapper, loader, device, accumulate: bool) -> float:
    mapper.eval()
    losses = []
    with torch.no_grad():
        for self_pred, aligned, y in loader:
            _, correction = mapper(aligned.to(device), accumulate=accumulate)
            losses.append(float(torch.nn.functional.l1_loss(self_pred.to(device) + correction, y.to(device)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def graph_prediction_scaled(
    torch,
    mapper,
    self_prediction: np.ndarray,
    aligned_source: np.ndarray,
    device,
    accumulate: bool,
) -> np.ndarray:
    loader = stage4._loader(torch, (self_prediction, aligned_source), False)
    predictions = []
    mapper.eval()
    with torch.no_grad():
        for self_pred, aligned in loader:
            _, correction = mapper(aligned.to(device), accumulate=accumulate)
            predictions.append((self_pred.to(device) + correction).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def metric_rows(
    source: str,
    target: str,
    variant: str,
    split_name: str,
    prediction_scaled: np.ndarray,
    samples: dict[str, np.ndarray],
    split_idx: np.ndarray,
    target_scaler: stage4.SimpleScaler,
    seed: int,
) -> list[dict[str, object]]:
    prediction_delta = target_scaler.inverse(prediction_scaled[split_idx])
    truth_delta = samples["target_delta"][split_idx]
    anchor = samples["anchor"][split_idx]
    arrays = {
        "pred_abs": anchor[:, None, :] + prediction_delta,
        "truth_abs": anchor[:, None, :] + truth_delta,
    }
    return stage4.metric_rows(source, target, variant, split_name, arrays, seed)


def parameter_rows(
    mapper,
    source: str,
    target: str,
    variant: str,
    support: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    lag_weights = mapper.lag_weights().detach().cpu().numpy()
    transfer = mapper.transfer.weight.detach().cpu().numpy()
    rows = [
        {
            "seed": seed,
            "source_station": source,
            "target_station": target,
            "variant": variant,
            "parameter_type": "lag_weight",
            "lag_steps": lag,
            "lag_hours": lag * 4,
            "source_feature": "",
            "target_feature": "",
            "value": float(weight),
        }
        for lag, weight in zip(support, lag_weights)
    ]
    for target_idx, target_feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
        for source_idx, source_feature in enumerate(protocol.INPUT_FEATURE_COLUMNS):
            rows.append(
                {
                    "seed": seed,
                    "source_station": source,
                    "target_station": target,
                    "variant": variant,
                    "parameter_type": "transfer_weight",
                    "lag_steps": "",
                    "lag_hours": "",
                    "source_feature": source_feature,
                    "target_feature": target_feature,
                    "value": float(transfer[target_idx, source_idx]),
                }
            )
    return rows


def run_pair_suite(torch, data: pd.DataFrame, lag_row: pd.Series, seed: int, device):
    source = str(lag_row["source_station"])
    target = str(lag_row["target_station"])
    control_source = cfg.REVERSE_SOURCES[target]
    primary_lag = int(lag_row["lag_primary_steps"])
    support = delayed_graph.lag_support(primary_lag)
    samples = build_pair_samples(data, source, target, control_source)
    split = split_indices(samples["origin_time"])
    if any(len(indices) == 0 for indices in split.values()):
        raise ValueError(f"Incomplete train/val/test coverage for {source} -> {target}")
    scalers = fit_scalers(samples, split["train"])
    samples = transform_samples(samples, scalers)

    self_samples = {
        **samples,
        "y_scaled": samples["target_scaled"],
    }
    self_model = stage4.train_self_model(torch, self_samples, split, device)
    self_prediction = predict_self_scaled(torch, self_model, samples, device)

    strict24_model = train_source_model(
        torch,
        samples["source_x_24h_scaled"],
        samples["source_step_y_scaled"],
        split,
        device,
    )
    strict36_model = train_source_model(
        torch,
        samples["source_x_36h_scaled"],
        samples["source_step_y_scaled"],
        split,
        device,
    )
    control36_model = train_source_model(
        torch,
        samples["control_x_36h_scaled"],
        samples["control_step_y_scaled"],
        split,
        device,
    )
    strict24_future = predict_source(torch, strict24_model, samples["source_x_24h_scaled"], device)
    strict36_future = predict_source(torch, strict36_model, samples["source_x_36h_scaled"], device)
    control36_future = predict_source(torch, control36_model, samples["control_x_36h_scaled"], device)

    aligned24, known24 = build_aligned_batch(samples["source_x_24h_scaled"], strict24_future, support)
    aligned36, known36 = build_aligned_batch(samples["source_x_36h_scaled"], strict36_future, support)
    aligned_control, _known_control = build_aligned_batch(
        samples["control_x_36h_scaled"],
        control36_future,
        support,
    )
    aligned_observed = delayed_graph.observed_only(aligned36, known36)
    aligned_shuffled = shuffle_aligned_by_split(aligned36, split, seed)

    definitions = {
        "point_graph_36h": (aligned36, False, source),
        "step_graph_24h": (aligned24, True, source),
        "step_graph_36h": (aligned36, True, source),
        "step_graph_36h_observed_only": (aligned_observed, True, source),
        "step_graph_36h_shuffled": (aligned_shuffled, True, source),
        "step_graph_36h_downstream_source": (aligned_control, True, control_source),
    }
    predictions = {"self_D_6to9": (self_prediction, source)}
    parameter_output = []
    for variant, (aligned, accumulate, used_source) in definitions.items():
        mapper = train_graph_mapper(
            torch,
            self_prediction,
            aligned,
            samples["target_scaled"],
            split,
            device,
            accumulate,
        )
        predictions[variant] = (
            graph_prediction_scaled(torch, mapper, self_prediction, aligned, device, accumulate),
            used_source,
        )
        parameter_output.extend(parameter_rows(mapper, used_source, target, variant, support, seed))

    metric_output = []
    for variant, (prediction, used_source) in predictions.items():
        for split_name, split_idx in split.items():
            metric_output.extend(
                metric_rows(
                    used_source,
                    target,
                    variant,
                    split_name,
                    prediction,
                    samples,
                    split_idx,
                    scalers["target"],
                    seed,
                )
            )
    coverage = {
        "source_station": source,
        "target_station": target,
        "control_source": control_source,
        "primary_lag_steps": primary_lag,
        "lag_support_steps": list(support),
        "sample_count": int(len(samples["origin_time"])),
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "observed_message_fraction": float(known36.mean()),
    }
    return metric_output, parameter_output, coverage


def write_report(overall: pd.DataFrame, coverage: list[dict[str, object]], output_path: Path) -> None:
    validation = overall[overall["split"].eq("val")].sort_values(
        ["target_station", "mean_rmse", "variant"]
    )
    test = overall[overall["split"].eq("test")].sort_values(
        ["target_station", "mean_rmse", "variant"]
    )
    lines = [
        "# V2 Stage 4b Delayed Step-Graph Pilot",
        "",
        "- Self branch: 24h history, direct cumulative 4-36h changes.",
        "- Graph branch: physical lag kernel over upstream one-step changes.",
        "- Selection split: validation only.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Validation Overall",
        "",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## Test Overall",
        "",
        "```text",
        test.to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def formal_seed_dir(seed: int) -> Path:
    return cfg.FORMAL_DIR / f"seed_{int(seed)}"


def pilot_supports_formal(overall: pd.DataFrame) -> bool:
    required = {
        "self_D_6to9",
        "step_graph_36h",
        "step_graph_36h_shuffled",
        "step_graph_36h_downstream_source",
    }
    validation = overall[overall["split"].astype(str).eq(SELECTION_SPLIT)].copy()
    targets = {target for _, target in cfg.PAIRS}
    if set(validation["target_station"].astype(str)) != targets:
        return False
    for target in targets:
        rows = validation[validation["target_station"].astype(str).eq(target)]
        values = rows.set_index("variant")["mean_rmse"].to_dict()
        if not required.issubset(values):
            return False
        strict = float(values["step_graph_36h"])
        if not all(
            strict < float(values[variant])
            for variant in (
                "self_D_6to9",
                "step_graph_36h_shuffled",
                "step_graph_36h_downstream_source",
            )
        ):
            return False
    return True


def run_seed(seed: int, output_dir: Path) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_audit = build_stage4b_lag_audit()
    data = load_v2_data()
    metric_output = []
    parameter_output = []
    coverage = []
    for lag_row in lag_audit.itertuples(index=False):
        console.print(
            f"running {lag_row.source_station}->{lag_row.target_station} "
            f"lag={lag_row.lag_primary_steps}",
            flush=True,
        )
        pair_metrics, pair_parameters, pair_coverage = run_pair_suite(
            torch,
            data,
            pd.Series(lag_row._asdict()),
            seed,
            device,
        )
        metric_output.extend(pair_metrics)
        parameter_output.extend(pair_parameters)
        coverage.append(pair_coverage)

    feature_horizon = pd.DataFrame(metric_output)
    overall = stage4.summarize_overall(feature_horizon)
    feature_horizon.to_csv(
        output_dir / "pair_feature_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "pair_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(parameter_output).to_csv(
        output_dir / "transfer_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = protocol.build_run_manifest(
        experiment="stage4b_delayed_step_graph_pilot",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/graph/v2_delayed_step_graph_config.py"),
            Path("scripts/graph/v2_delayed_step_graph.py"),
            Path("scripts/graph/run_v2_delayed_step_graph_ablation.py"),
        ),
    )
    manifest.update(
        {
            "self_input_steps": cfg.SELF_INPUT_STEPS,
            "graph_input_steps": list(cfg.GRAPH_INPUT_STEPS),
            "output_steps": cfg.OUTPUT_STEPS,
            "variants": list(REQUIRED_VARIANTS),
            "selection_split": SELECTION_SPLIT,
            "device": str(device),
            "coverage": coverage,
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(overall, coverage, output_dir / "run_report.md")
    return output_dir


def run_pilot(seed: int = cfg.PILOT_SEED) -> Path:
    cfg.ensure_output_dirs()
    return run_seed(seed, cfg.PILOT_DIR)


def summarize_multiseed(overall: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "target_station", "split"]
    baseline = overall[overall["variant"].eq("self_D_6to9")][keys + ["mean_rmse"]].rename(
        columns={"mean_rmse": "self_rmse"}
    )
    compared = overall.merge(baseline, on=keys, how="left", validate="many_to_one")
    compared["delta_rmse_minus_self"] = compared["mean_rmse"] - compared["self_rmse"]
    rows = []
    for keys_value, group in compared.groupby(["target_station", "variant", "split"], sort=True):
        delta = pd.to_numeric(group["delta_rmse_minus_self"], errors="coerce")
        rows.append(
            {
                "target_station": keys_value[0],
                "variant": keys_value[1],
                "split": keys_value[2],
                "seed_count": int(delta.notna().sum()),
                "mean_rmse": float(pd.to_numeric(group["mean_rmse"], errors="coerce").mean()),
                "std_rmse": float(pd.to_numeric(group["mean_rmse"], errors="coerce").std(ddof=1)),
                "mean_delta_rmse_minus_self": float(delta.mean()),
                "std_delta_rmse_minus_self": float(delta.std(ddof=1)),
                "win_seed_count": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_formal() -> Path:
    pilot_path = cfg.PILOT_DIR / "pair_overall_metrics.csv"
    if not pilot_path.exists():
        raise ValueError("Seed-42 pilot must run before formal multiseed training.")
    pilot_overall = pd.read_csv(pilot_path)
    if not pilot_supports_formal(pilot_overall):
        raise ValueError("Seed-42 validation gate did not support formal multiseed training.")
    cfg.ensure_output_dirs()
    overall_frames = []
    feature_frames = []
    for seed in cfg.FORMAL_SEEDS:
        output_dir = formal_seed_dir(seed)
        console.print(f"formal seed={seed}", flush=True)
        run_seed(seed, output_dir)
        overall_frames.append(pd.read_csv(output_dir / "pair_overall_metrics.csv"))
        feature_frames.append(pd.read_csv(output_dir / "pair_feature_horizon_metrics.csv"))
    overall = pd.concat(overall_frames, ignore_index=True)
    feature_horizon = pd.concat(feature_frames, ignore_index=True)
    summary = summarize_multiseed(overall)
    overall.to_csv(cfg.FORMAL_DIR / "pair_overall_all_seeds.csv", index=False, encoding="utf-8-sig")
    feature_horizon.to_csv(
        cfg.FORMAL_DIR / "pair_feature_horizon_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        cfg.FORMAL_DIR / "multiseed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return cfg.FORMAL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=cfg.PILOT_SEED)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.formal:
        output_dir = run_formal()
    elif args.pilot:
        output_dir = run_pilot(args.seed)
    else:
        raise SystemExit("Choose --pilot or --formal.")
    console.print(f"saved {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
