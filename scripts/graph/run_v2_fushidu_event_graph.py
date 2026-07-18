#!/usr/bin/env python3
"""Run the focused event-gated graph experiment for Fushidu."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.graph import run_v2_delayed_step_graph_ablation as stage4b
from scripts.graph import run_v2_direct_pair_graph_ablation as stage4
from scripts.graph import run_v2_flow_mass_balance_ablation as stage4c
from scripts.graph import v2_event_gated_graph as gated_graph
from scripts.common import v2_experiment_protocol as protocol

SOURCES = ("双港口", "富足山")
TARGET = "浮石渡"
OUTPUT_DIR = protocol.GRAPH_OUTPUT_ROOT / "stage6_fushidu_event_graph"
PILOT_DIR = OUTPUT_DIR / "pilot_seed42"
VARIANTS = (
    "self_D_6to9",
    "fulltime_shuanggangkou",
    "fulltime_fuzushan",
    "fulltime_dual",
    "fulltime_dual_fuzushan_shuffled",
    "event_shuanggangkou",
    "event_fuzushan",
    "event_dual",
    "event_dual_fuzushan_shuffled",
)
TARGET_INPUT_INDICES = np.asarray(
    [protocol.INPUT_FEATURE_COLUMNS.index(feature) for feature in protocol.TARGET_FEATURE_COLUMNS],
    dtype=int,
)
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 1e-3
TRANSFER_L1 = 1e-4
EVENT_LOSS_WEIGHT = 3.0


def target_feature_view(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[..., TARGET_INPUT_INDICES]


def select_source_gates(gates: np.ndarray, active_sources: tuple[int, ...]) -> np.ndarray:
    output = np.zeros_like(gates, dtype=bool)
    output[:, list(active_sources), :] = np.asarray(gates, dtype=bool)[:, list(active_sources), :]
    return output


def build_event_gates(
    samples: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[gated_graph.EventGateThresholds]]:
    target_current = target_feature_view(samples["self_x"][:, -1])
    rows = []
    thresholds = []
    for source_idx in range(len(SOURCES)):
        source_current = target_feature_view(samples["source_x_36h"][:, source_idx, -1])
        fitted = gated_graph.fit_event_thresholds(source_current, target_current, train_idx)
        thresholds.append(fitted)
        rows.append(gated_graph.event_gate(source_current, target_current, fitted))
    return np.stack(rows, axis=1), thresholds


def build_aligned_sources(torch, samples, split, source_scalers, supports, device):
    aligned = []
    for source_idx, support in enumerate(supports):
        source_model = stage4b.train_source_model(
            torch,
            samples["source_x_scaled"][:, source_idx],
            samples["source_y_scaled"][:, source_idx],
            split,
            device,
        )
        future = stage4b.predict_source(
            torch,
            source_model,
            samples["source_x_scaled"][:, source_idx],
            device,
        )
        source_aligned, _ = stage4b.build_aligned_batch(
            samples["source_x_scaled"][:, source_idx],
            future,
            support,
        )
        aligned.append(source_aligned)
    return tuple(aligned)


def _weighted_l1(torch, prediction, target, feature_gates):
    active = feature_gates.amax(dim=1).cumsum(dim=1).clamp(max=1.0)
    weights = 1.0 + EVENT_LOSS_WEIGHT * active
    return (torch.abs(prediction - target) * weights).mean()


def train_mapper(torch, self_prediction, aligned_sources, feature_gates, target, split, device):
    mapper = gated_graph.make_event_gated_multi_source_mapper(
        torch,
        source_dim=aligned_sources[0].shape[-1],
        target_dim=target.shape[-1],
        lag_counts=tuple(values.shape[-2] for values in aligned_sources),
    ).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: stage4._loader(
            torch,
            (
                self_prediction[idx],
                *(values[idx] for values in aligned_sources),
                feature_gates[idx].astype(np.float32),
                target[idx],
            ),
            name == "train",
        )
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    wait = 0
    for _epoch in range(MAX_EPOCHS):
        mapper.train()
        for batch in loaders["train"]:
            self_pred = batch[0].to(device)
            source_count = len(aligned_sources)
            aligned = tuple(value.to(device) for value in batch[1 : 1 + source_count])
            gates = batch[-2].to(device)
            target_batch = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            _, correction = mapper(aligned, gates)
            loss = _weighted_l1(torch, self_pred + correction, target_batch, gates)
            transfer_l1 = torch.stack(
                [edge.transfer.weight.abs().mean() for edge in mapper.edge_mappers]
            ).mean()
            loss = loss + TRANSFER_L1 * transfer_l1
            loss.backward()
            optimizer.step()
        val_loss = evaluate_mapper(torch, mapper, loaders["val"], len(aligned_sources), device)
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


def evaluate_mapper(torch, mapper, loader, source_count: int, device) -> float:
    mapper.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            aligned = tuple(value.to(device) for value in batch[1 : 1 + source_count])
            gates = batch[-2].to(device)
            _, correction = mapper(aligned, gates)
            losses.append(
                float(_weighted_l1(torch, batch[0].to(device) + correction, batch[-1].to(device), gates).cpu())
            )
    return float(np.mean(losses)) if losses else float("inf")


def predict_mapper(torch, mapper, self_prediction, aligned_sources, feature_gates, device):
    loader = stage4._loader(
        torch,
        (self_prediction, *aligned_sources, feature_gates.astype(np.float32)),
        False,
    )
    predictions = []
    mapper.eval()
    with torch.no_grad():
        for batch in loader:
            aligned = tuple(value.to(device) for value in batch[1:-1])
            _, correction = mapper(aligned, batch[-1].to(device))
            predictions.append((batch[0].to(device) + correction).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def subgroup_metric_rows(variant, prediction, target, gates, split_idx, seed):
    error = prediction[split_idx] - target[split_idx]
    selected_gates = gates[split_idx]
    groups = {
        "all": np.ones((len(split_idx), len(protocol.TARGET_FEATURE_COLUMNS)), dtype=bool),
        "any_source_event": selected_gates.any(axis=1),
        "shuanggangkou_event": selected_gates[:, 0],
        "fuzushan_event": selected_gates[:, 1],
        "fuzushan_only_event": selected_gates[:, 1] & ~selected_gates[:, 0],
        "both_sources_event": selected_gates[:, 0] & selected_gates[:, 1],
    }
    rows = []
    for group_name, feature_mask in groups.items():
        expanded = np.broadcast_to(feature_mask[:, None, :], error.shape)
        values = error[expanded]
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "split": "val",
                "subgroup": group_name,
                "origin_feature_count": int(feature_mask.sum()),
                "scaled_point_count": int(values.size),
                "mae_scaled": float(np.mean(np.abs(values))) if values.size else np.nan,
                "rmse_scaled": float(np.sqrt(np.mean(values**2))) if values.size else np.nan,
            }
        )
    return rows


def subgroup_feature_metric_rows(variant, prediction, target, gates, split_idx, seed):
    error = prediction[split_idx] - target[split_idx]
    selected_gates = gates[split_idx]
    groups = {
        "shuanggangkou_event": selected_gates[:, 0],
        "fuzushan_event": selected_gates[:, 1],
        "fuzushan_only_event": selected_gates[:, 1] & ~selected_gates[:, 0],
        "both_sources_event": selected_gates[:, 0] & selected_gates[:, 1],
    }
    rows = []
    for group_name, feature_mask in groups.items():
        for feature_idx, feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
            origin_mask = feature_mask[:, feature_idx]
            values = error[origin_mask, :, feature_idx]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "split": "val",
                    "subgroup": group_name,
                    "feature": feature,
                    "event_origin_count": int(origin_mask.sum()),
                    "scaled_point_count": int(values.size),
                    "mae_scaled": float(np.mean(np.abs(values))) if values.size else np.nan,
                    "rmse_scaled": float(np.sqrt(np.mean(values**2))) if values.size else np.nan,
                }
            )
    return rows


def parameter_rows(mapper, variant, supports, seed):
    rows = []
    for source, support, edge in zip(SOURCES, supports, mapper.edge_mappers):
        for lag, value in zip(support, edge.lag_weights().detach().cpu().numpy()):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "source_station": source,
                    "target_station": TARGET,
                    "parameter_type": "lag_weight",
                    "lag_steps": lag,
                    "lag_hours": lag * 4,
                    "source_feature": "",
                    "target_feature": "",
                    "value": float(value),
                }
            )
        transfer = edge.transfer.weight.detach().cpu().numpy()
        for target_idx, target_feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
            for source_idx, source_feature in enumerate(protocol.INPUT_FEATURE_COLUMNS):
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "source_station": source,
                        "target_station": TARGET,
                        "parameter_type": "transfer_weight",
                        "lag_steps": "",
                        "lag_hours": "",
                        "source_feature": source_feature,
                        "target_feature": target_feature,
                        "value": float(transfer[target_idx, source_idx]),
                    }
                )
    return rows


def gate_diagnostics(gates, split, thresholds):
    rows = []
    for split_name, idx in split.items():
        for source_idx, source in enumerate(SOURCES):
            for feature_idx, feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
                rows.append(
                    {
                        "split": split_name,
                        "source_station": source,
                        "target_station": TARGET,
                        "feature": feature,
                        "event_count": int(gates[idx, source_idx, feature_idx].sum()),
                        "origin_count": int(len(idx)),
                        "event_rate": float(gates[idx, source_idx, feature_idx].mean()),
                        "source_q95": float(thresholds[source_idx].source_q95[feature_idx]),
                        "target_q50": float(thresholds[source_idx].target_q50[feature_idx]),
                    }
                )
    return rows


def pilot_supports_formal(subgroup: pd.DataFrame) -> bool:
    values = subgroup.set_index(["subgroup", "variant"])["rmse_scaled"].to_dict()
    required = {
        ("all", "self_D_6to9"),
        ("all", "fulltime_shuanggangkou"),
        ("all", "fulltime_dual"),
        ("all", "fulltime_dual_fuzushan_shuffled"),
        ("any_source_event", "self_D_6to9"),
        ("any_source_event", "event_shuanggangkou"),
        ("any_source_event", "event_dual"),
        ("any_source_event", "event_dual_fuzushan_shuffled"),
    }
    if not required.issubset(values):
        return False
    fulltime = values[("all", "fulltime_dual")]
    event = values[("any_source_event", "event_dual")]
    return (
        fulltime < values[("all", "self_D_6to9")]
        and fulltime < values[("all", "fulltime_shuanggangkou")]
        and fulltime < values[("all", "fulltime_dual_fuzushan_shuffled")]
        and event < values[("any_source_event", "self_D_6to9")]
        and event < values[("any_source_event", "event_shuanggangkou")]
        and event < values[("any_source_event", "event_dual_fuzushan_shuffled")]
    )


def write_report(output_dir, subgroup, subgroup_feature, feature_horizon, coverage):
    all_rows = subgroup[subgroup["subgroup"].eq("all")].sort_values("mae_scaled")
    event_rows = subgroup[subgroup["subgroup"].eq("any_source_event")].sort_values("mae_scaled")
    fuzushan_features = subgroup_feature[
        subgroup_feature["subgroup"].eq("fuzushan_event")
        & subgroup_feature["variant"].isin(
            ("self_D_6to9", "fulltime_fuzushan", "event_fuzushan", "event_dual")
        )
    ].sort_values(["feature", "rmse_scaled"])
    gate = pilot_supports_formal(subgroup)
    lines = [
        "# V2 Stage 6 Fushidu Event-Gated Graph Pilot",
        "",
        "- Interpretation: Shuanggangkou may already represent mixed confluence water; sources are candidate information nodes, not two independent tributary weights.",
        "- Self input: 24h downstream changes; graph input: 36h upstream changes; direct output: 4-36h.",
        "- Event gate: train-Q95 upstream shock and train-Q50 quiet current target, computed from origin-visible values only; the message opens only at the edge-specific physical arrival support and accumulates afterward.",
        "- Test labels remain blinded; this report contains validation metrics only.",
        f"- Formal multiseed gate: {gate}.",
        "",
        "## Coverage",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Validation Overall",
        "```text",
        all_rows.to_string(index=False),
        "```",
        "",
        "## Validation Event Subgroup",
        "```text",
        event_rows.to_string(index=False),
        "```",
        "",
        "## Fuzushan Event Features",
        "```text",
        fuzushan_features.to_string(index=False),
        "```",
        "",
        "## Validation Feature-Horizon Preview",
        "```text",
        feature_horizon.sort_values(["variant", "feature", "horizon_step"]).head(30).to_string(index=False),
        "```",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(seed: int = protocol.PILOT_SEED) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    data = stage4b.load_v2_data()
    samples = stage4c.build_confluence_samples(data, SOURCES, TARGET)
    split = stage4b.split_indices(samples["origin_time"])
    target_scalers, source_scalers = stage4c.fit_scalers(samples, split["train"])
    samples = stage4c.transform_samples(samples, target_scalers, source_scalers)
    self_samples = {**samples, "y_scaled": samples["target_scaled"]}
    self_model = stage4.train_self_model(torch, self_samples, split, device)
    self_prediction = stage4b.predict_self_scaled(torch, self_model, samples, device)

    lag_audit = stage4c.build_lag_audit()
    supports = stage4c.edge_lag_supports(lag_audit, SOURCES, TARGET)
    aligned = build_aligned_sources(torch, samples, split, source_scalers, supports, device)
    gates, thresholds = build_event_gates(samples, split["train"])

    event_aligned = tuple(
        gated_graph.current_event_pulse(values, support)
        for values, support in zip(aligned, supports)
    )
    arrival_gates = gated_graph.arrival_feature_gates(
        gates,
        supports,
        output_steps=samples["target_scaled"].shape[1],
    )
    shuffled_fuzushan_aligned = gated_graph.shuffle_by_split(event_aligned[1], split, seed)
    shuffled_fulltime_fuzushan = gated_graph.shuffle_by_split(aligned[1], split, seed)
    shuffled_fuzushan_gate = gated_graph.shuffle_by_split(arrival_gates[:, 1], split, seed)
    shuffled_gates = arrival_gates.copy()
    shuffled_gates[:, 1] = shuffled_fuzushan_gate

    definitions = {
        "fulltime_shuanggangkou": (
            aligned,
            select_source_gates(
                np.ones(
                    (
                        len(gates),
                        len(SOURCES),
                        samples["target_scaled"].shape[1],
                        len(protocol.TARGET_FEATURE_COLUMNS),
                    ),
                    dtype=bool,
                ),
                (0,),
            ),
        ),
        "fulltime_fuzushan": (
            aligned,
            select_source_gates(
                np.ones(
                    (
                        len(gates),
                        len(SOURCES),
                        samples["target_scaled"].shape[1],
                        len(protocol.TARGET_FEATURE_COLUMNS),
                    ),
                    dtype=bool,
                ),
                (1,),
            ),
        ),
        "fulltime_dual": (
            aligned,
            np.ones(
                (
                    len(gates),
                    len(SOURCES),
                    samples["target_scaled"].shape[1],
                    len(protocol.TARGET_FEATURE_COLUMNS),
                ),
                dtype=bool,
            ),
        ),
        "fulltime_dual_fuzushan_shuffled": (
            (aligned[0], shuffled_fulltime_fuzushan),
            np.ones(
                (
                    len(gates),
                    len(SOURCES),
                    samples["target_scaled"].shape[1],
                    len(protocol.TARGET_FEATURE_COLUMNS),
                ),
                dtype=bool,
            ),
        ),
        "event_shuanggangkou": (
            event_aligned,
            select_source_gates(arrival_gates, (0,)),
        ),
        "event_fuzushan": (
            event_aligned,
            select_source_gates(arrival_gates, (1,)),
        ),
        "event_dual": (event_aligned, arrival_gates),
        "event_dual_fuzushan_shuffled": (
            (event_aligned[0], shuffled_fuzushan_aligned),
            shuffled_gates,
        ),
    }
    predictions = {"self_D_6to9": self_prediction}
    parameters = []
    for variant, (variant_aligned, variant_gates) in definitions.items():
        console.print(f"training {variant}", flush=True)
        torch.manual_seed(seed)
        mapper = train_mapper(
            torch,
            self_prediction,
            variant_aligned,
            variant_gates,
            samples["target_scaled"],
            split,
            device,
        )
        predictions[variant] = predict_mapper(
            torch,
            mapper,
            self_prediction,
            variant_aligned,
            variant_gates,
            device,
        )
        parameters.extend(parameter_rows(mapper, variant, supports, seed))

    feature_rows = []
    subgroup_rows = []
    subgroup_feature_rows = []
    val_idx = split["val"]
    for variant, prediction in predictions.items():
        feature_rows.extend(
            stage4b.metric_rows(
                "+".join(SOURCES),
                TARGET,
                variant,
                "val",
                prediction,
                samples,
                val_idx,
                target_scalers["target"],
                seed,
            )
        )
        subgroup_rows.extend(
            subgroup_metric_rows(
                variant,
                prediction,
                samples["target_scaled"],
                gates,
                val_idx,
                seed,
            )
        )
        subgroup_feature_rows.extend(
            subgroup_feature_metric_rows(
                variant,
                prediction,
                samples["target_scaled"],
                gates,
                val_idx,
                seed,
            )
        )

    feature_horizon = pd.DataFrame(feature_rows)
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup_feature = pd.DataFrame(subgroup_feature_rows)
    gate_frame = pd.DataFrame(gate_diagnostics(gates, split, thresholds))
    parameter_frame = pd.DataFrame(parameters)
    feature_horizon.to_csv(PILOT_DIR / "validation_feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(PILOT_DIR / "validation_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    subgroup_feature.to_csv(
        PILOT_DIR / "validation_event_feature_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_frame.to_csv(PILOT_DIR / "event_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    parameter_frame.to_csv(PILOT_DIR / "graph_parameters.csv", index=False, encoding="utf-8-sig")

    coverage = {
        "sources": list(SOURCES),
        "target": TARGET,
        "source_interpretation": "candidate upstream information nodes; Shuanggangkou may already contain confluence mixing",
        "hard_flow_weighting": False,
        "self_input_hours": 24,
        "graph_input_hours": 36,
        "output_hours": list(range(4, 37, 4)),
        "lag_support_steps": {source: list(support) for source, support in zip(SOURCES, supports)},
        "split_counts": {name: int(len(idx)) for name, idx in split.items()},
        "test_labels_evaluated": False,
    }
    manifest = protocol.build_run_manifest(
        experiment="stage6_fushidu_event_gated_graph",
        output_dir=PILOT_DIR,
        seed=seed,
        code_paths=(
            Path("scripts/graph/v2_event_gated_graph.py"),
            Path("scripts/graph/run_v2_fushidu_event_graph.py"),
        ),
    )
    manifest.update({"device": str(device), "variants": list(VARIANTS), "coverage": coverage})
    (PILOT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(PILOT_DIR, subgroup, subgroup_feature, feature_horizon, coverage)
    return PILOT_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=protocol.PILOT_SEED)
    args = parser.parse_args()
    console.print(run(args.seed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
