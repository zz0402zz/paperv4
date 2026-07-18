#!/usr/bin/env python3
"""Validation-only pilot entry point with graph and test-label safety gates."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.graph import continuous_subgraph_controls as controls
from scripts.graph import continuous_subgraph_dataset as dataset
from scripts.graph import continuous_subgraph_evidence as evidence
from scripts.graph import continuous_subgraph_metrics as metric_utils
from scripts.graph.continuous_subgraph_model import ContinuousSubgraphForecaster
from scripts.common import v2_experiment_protocol as protocol

OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage7_continuous_subgraph" / "pilot_seed42"
VALIDATION_DECISION_PATH = OUTPUT_DIR / "validation_decision.json"
CONTROL_VARIANTS = (
    "shared_self",
    "shared_self_param_control",
    "reverse_graph",
    "degree_matched_random_graph",
    "wrong_branch_graph",
    "time_shuffled_graph",
)
VARIANTS = ("strict_graph", *CONTROL_VARIANTS)
STATION_ORDER = ("浮石渡", "半潭", "下童", "洋港", "横山")
INPUT_STEPS = 6
OUTPUT_STEPS = 9
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 7
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 24
PANEL_PATH = Path("data/processed/v2/continuous_subgraph/quantity_4h.csv")
QUALITY_PATH = Path("data/processed/v2/continuous_subgraph/quality_4h.csv")
FLOW_PATH = Path("data/processed/v2/continuous_subgraph/daily_flow.csv")


def evaluate_validation_gate(rmse_by_variant: dict[str, float]) -> dict[str, object]:
    required = {"strict_graph", *CONTROL_VARIANTS}
    missing = sorted(required - set(rmse_by_variant))
    if missing:
        raise ValueError(f"Missing validation RMSE variants: {missing}")
    strict = float(rmse_by_variant["strict_graph"])
    failed = [
        variant
        for variant in CONTROL_VARIANTS
        if strict >= float(rmse_by_variant[variant])
    ]
    return {
        "metric": "validation_rmse",
        "strict_graph_rmse": strict,
        "approved": not failed,
        "failed_comparisons": failed,
        "rmse_by_variant": {key: float(value) for key, value in rmse_by_variant.items()},
    }


def require_approved_validation_decision(path: Path = VALIDATION_DECISION_PATH) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError("Test export requires an existing validation decision file.")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if not bool(decision.get("approved")):
        raise RuntimeError("Test export is not approved by the validation decision.")
    return decision


def graph_preflight() -> evidence.GraphReadiness:
    nodes = pd.read_csv(evidence.NODE_PATH)
    edges = pd.read_csv(evidence.EDGE_PATH)
    return evidence.evaluate_graph_readiness(nodes, edges)


def align_prior_day_flow(
    origin_times: np.ndarray,
    daily_flow: pd.DataFrame,
) -> np.ndarray:
    """Use only the previous complete day's flow for each forecast origin."""
    flow = daily_flow.copy()
    flow["date"] = pd.to_datetime(flow["date"]).dt.normalize()
    flow = flow.set_index("date")
    lookup_dates = pd.to_datetime(origin_times).normalize() - pd.Timedelta(days=1)
    upstream = flow["quzhou"].reindex(lookup_dates).to_numpy(dtype=float)
    downstream = flow["qujiang_lower_est"].reindex(lookup_dates).to_numpy(dtype=float)
    return np.column_stack((upstream, upstream, downstream, downstream))


def normalize_flow_by_train(
    train_flow: np.ndarray,
    split_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_logged = np.log1p(np.maximum(train_flow, 0.0))
    split_logged = np.log1p(np.maximum(split_flow, 0.0))
    scale = np.nanmedian(train_logged, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return train_logged / scale, split_logged / scale


def _strict_edge_array() -> np.ndarray:
    edges = pd.read_csv(evidence.EDGE_PATH)
    node_index = {station: idx for idx, station in enumerate(STATION_ORDER)}
    return np.asarray(
        [[node_index[source], node_index[target]] for source, target in edges[["source_station", "target_station"]].itertuples(index=False, name=None)],
        dtype=int,
    )


def _variant_edges(variant: str, strict: np.ndarray) -> np.ndarray:
    if variant == "reverse_graph":
        return controls.reverse_edges(strict)
    if variant == "degree_matched_random_graph":
        return controls.degree_matched_random_edges(strict, len(STATION_ORDER), protocol.PILOT_SEED)
    if variant == "wrong_branch_graph":
        wrong = np.asarray([[4, 1], [0, 2], [1, 3], [2, 4]], dtype=int)
        return controls.validate_wrong_relation_edges(strict, wrong)
    return strict.copy()


def _graph_history_for_variant(
    variant: str,
    split_name: str,
    history: np.ndarray,
) -> np.ndarray:
    if variant == "shared_self_param_control":
        return np.zeros_like(history)
    if variant == "time_shuffled_graph":
        labels = np.full(len(history), split_name, dtype=object)
        return controls.block_shuffle_by_split(
            history,
            labels,
            block_steps=6,
            seed=protocol.PILOT_SEED + {"train": 0, "val": 1}.get(split_name, 2),
        )
    return history.copy()


def _masked_l1(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_error = torch.where(mask, torch.abs(prediction - truth), torch.zeros_like(prediction))
    return valid_error.sum() / mask.sum().clamp_min(1)


def _make_loader(
    split: dict[str, np.ndarray],
    graph_history: np.ndarray,
    flow_strength: np.ndarray,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    tensors = (
        torch.as_tensor(split["history_diffs"], dtype=torch.float32),
        torch.as_tensor(split["current_targets"], dtype=torch.float32),
        torch.as_tensor(split["target_levels"], dtype=torch.float32),
        torch.as_tensor(split["target_mask"], dtype=torch.bool),
        torch.as_tensor(graph_history, dtype=torch.float32),
        torch.as_tensor(flow_strength, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def _evaluate_model(
    model: ContinuousSubgraphForecaster,
    loader: torch.utils.data.DataLoader,
    edge_tensor: torch.Tensor,
    graph_enabled: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    predictions, truths, masks = [], [], []
    with torch.no_grad():
        for history, current, truth, mask, graph_history, flow in loader:
            output = model(
                history,
                current,
                edge_tensor,
                edge_flow_strength=flow,
                aligned_upstream_diffs=graph_history,
                graph_enabled=graph_enabled,
            )
            loss = _masked_l1(output.level, truth, mask)
            losses.append(float(loss))
            predictions.append(output.level.cpu().numpy())
            truths.append(truth.cpu().numpy())
            masks.append(mask.cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(predictions),
        np.concatenate(truths),
        np.concatenate(masks),
    )


def train_variant(
    variant: str,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    flow_by_split: dict[str, np.ndarray],
    scalers: dict[str, np.ndarray],
    strict_edges: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    random.seed(protocol.PILOT_SEED)
    np.random.seed(protocol.PILOT_SEED)
    torch.manual_seed(protocol.PILOT_SEED)
    model = ContinuousSubgraphForecaster(
        num_nodes=len(STATION_ORDER),
        output_steps=OUTPUT_STEPS,
        hidden_size=HIDDEN_SIZE,
    )
    edge_array = _variant_edges(variant, strict_edges)
    edge_tensor = torch.as_tensor(edge_array, dtype=torch.long)
    graph_enabled = variant != "shared_self"
    loaders = {
        split_name: _make_loader(
            split,
            _graph_history_for_variant(variant, split_name, split["history_diffs"]),
            flow_by_split[split_name],
            shuffle=split_name == "train",
        )
        for split_name, split in scaled_splits.items()
        if split_name in {"train", "val"}
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history_rows = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for history, current, truth, mask, graph_history, flow in loaders["train"]:
            optimizer.zero_grad()
            output = model(
                history,
                current,
                edge_tensor,
                edge_flow_strength=flow,
                aligned_upstream_diffs=graph_history,
                graph_enabled=graph_enabled,
            )
            loss = _masked_l1(output.level, truth, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        val_loss, _, _, _ = _evaluate_model(model, loaders["val"], edge_tensor, graph_enabled)
        history_rows.append(
            {"variant": variant, "epoch": epoch, "train_l1": float(np.mean(train_losses)), "val_l1": val_loss}
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    _, pred_scaled, truth_scaled, mask = _evaluate_model(
        model,
        loaders["val"],
        edge_tensor,
        graph_enabled,
    )
    prediction = dataset.inverse_levels(pred_scaled, scalers)
    truth = dataset.inverse_levels(truth_scaled, scalers)
    overall = metric_utils.masked_regression_metrics(prediction, truth, mask)
    result = {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_l1_scaled": best_val,
        **overall,
        "edge_index": edge_array.tolist(),
    }
    return result, history_rows, prediction, truth, mask


def _detail_metric_rows(
    variant: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for horizon_idx in range(OUTPUT_STEPS):
        for station_idx, station in enumerate(STATION_ORDER):
            for feature_idx, feature in enumerate(dataset.TARGET_FEATURES):
                metrics = metric_utils.masked_regression_metrics(
                    prediction[:, horizon_idx, station_idx, feature_idx],
                    truth[:, horizon_idx, station_idx, feature_idx],
                    mask[:, horizon_idx, station_idx, feature_idx],
                )
                rows.append(
                    {
                        "variant": variant,
                        "split": "val",
                        "horizon_step": horizon_idx + 1,
                        "horizon_hours": (horizon_idx + 1) * 4,
                        "station": station,
                        "feature": feature,
                        **metrics,
                    }
                )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_validation_pilot() -> None:
    panel = pd.read_csv(PANEL_PATH)
    quality = pd.read_csv(QUALITY_PATH)
    raw_splits = dataset.build_joint_samples(panel, quality, STATION_ORDER, INPUT_STEPS, OUTPUT_STEPS)
    scalers = dataset.fit_train_scalers(raw_splits["train"])
    scaled_splits = {
        split_name: dataset.scale_split(split, scalers)
        for split_name, split in raw_splits.items()
    }
    daily_flow = pd.read_csv(FLOW_PATH)
    raw_flow = {
        split_name: align_prior_day_flow(split["origin_time"], daily_flow)
        for split_name, split in raw_splits.items()
    }
    train_flow_scaled, _ = normalize_flow_by_train(raw_flow["train"], raw_flow["train"])
    flow_by_split = {"train": train_flow_scaled}
    for split_name in ("val", "test"):
        _, flow_by_split[split_name] = normalize_flow_by_train(raw_flow["train"], raw_flow[split_name])
    if any(not np.isfinite(flow_by_split[name]).all() for name in ("train", "val")):
        raise ValueError("Prior-day flow coverage is incomplete in train or validation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strict_edges = _strict_edge_array()
    results, history_rows, detail_rows = [], [], []
    for variant in VARIANTS:
        result, history, prediction, truth, mask = train_variant(
            variant,
            scaled_splits,
            flow_by_split,
            scalers,
            strict_edges,
        )
        results.append(result)
        history_rows.extend(history)
        detail_rows.extend(_detail_metric_rows(variant, prediction, truth, mask))
        console.print(f"{variant}: val_rmse={result['rmse']:.6f} best_epoch={result['best_epoch']}")

    result_frame = pd.DataFrame(results).sort_values("rmse")
    result_frame.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(
        OUTPUT_DIR / "validation_station_feature_horizon.csv", index=False, encoding="utf-8-sig"
    )
    rmse_by_variant = {row["variant"]: row["rmse"] for row in results}
    decision = evaluate_validation_gate(rmse_by_variant)
    decision["test_labels_opened"] = False
    _write_json(VALIDATION_DECISION_PATH, decision)
    _write_json(
        OUTPUT_DIR / "dataset_summary.json",
        {
            "station_order": list(STATION_ORDER),
            "input_steps": INPUT_STEPS,
            "output_steps": OUTPUT_STEPS,
            "split_samples": {name: len(split["origin_time"]) for name, split in raw_splits.items()},
            "test_labels_opened": False,
            "flow_policy": "prior complete day; first two edges use Quzhou, last two use Lanxi minus Jinhua",
        },
    )
    _write_json(
        OUTPUT_DIR / "scalers.json",
        {key: value.tolist() for key, value in scalers.items()},
    )
    report = [
        "# Continuous Subgraph Validation-Only Pilot",
        "",
        "- Graph: 浮石渡 -> 半潭 -> 下童 -> 洋港 -> 横山.",
        "- Input: prior 24h nine-feature changes; output: 4-36h five target levels.",
        "- Flow: previous complete day only.",
        "- Test labels opened: False.",
        f"- Validation gate approved: {decision['approved']}.",
        "",
        "## Validation RMSE",
        "",
        "| Variant | RMSE | MAE | NSE | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result_frame.itertuples(index=False):
        report.append(f"| {row.variant} | {row.rmse:.6f} | {row.mae:.6f} | {row.nse:.6f} | {row.best_epoch} |")
    report.extend(["", f"Failed comparisons: {', '.join(decision['failed_comparisons']) or 'none'}", ""])
    (OUTPUT_DIR / "run_report.md").write_text("\n".join(report), encoding="utf-8")


def write_blocked_preflight(result: evidence.GraphReadiness) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_before_training",
        "reason": "graph_evidence_gate_failed",
        "topology_ready": result.topology_ready,
        "flow_ready": result.flow_ready,
        "graph_ready": result.graph_ready,
        "issue_codes": list(result.issue_codes),
        "test_labels_opened": False,
    }
    (OUTPUT_DIR / "preflight_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    result = graph_preflight()
    if not result.graph_ready:
        write_blocked_preflight(result)
        console.print(
            "pilot_not_started graph_evidence_gate_failed "
            f"issues={','.join(result.issue_codes)}"
        )
        return
    run_validation_pilot()


if __name__ == "__main__":
    main()
