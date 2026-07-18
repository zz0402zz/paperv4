#!/usr/bin/env python3
"""Run the V2 multi-source flow mass-balance graph ablation."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import hydromet_features
from scripts.graph import run_v2_direct_pair_graph_ablation as stage4
from scripts.graph import run_v2_delayed_step_graph_ablation as stage4b
from scripts.graph import v2_delayed_step_graph as delayed_graph
from scripts.common import v2_experiment_protocol as protocol
from scripts.graph import v2_flow_mass_balance as flow_weight
from scripts.graph import v2_flow_mass_balance_config as cfg
from scripts.graph import v2_physical_lag


MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 1e-3
TRANSFER_L1 = 1e-4


def build_confluence_samples(
    data: pd.DataFrame,
    sources: tuple[str, str],
    target: str,
) -> dict[str, np.ndarray]:
    """Build target origins shared by both incoming source branches."""
    target_frame = stage4b._station_frame(data, target, "target")
    merged = target_frame
    source_suffixes = []
    for source_idx, source in enumerate(sources):
        suffix = f"source{source_idx}"
        source_suffixes.append(suffix)
        merged = merged.merge(stage4b._station_frame(data, source, suffix), on="time", how="inner")

    target_diff_columns = [f"{feature}_diff1_target" for feature in protocol.INPUT_FEATURE_COLUMNS]
    target_level_columns = [f"{feature}_target" for feature in protocol.TARGET_FEATURE_COLUMNS]
    target_diff = merged[target_diff_columns].to_numpy(float)
    target_level = merged[target_level_columns].to_numpy(float)
    source_diff = np.stack(
        [
            merged[[f"{feature}_diff1_{suffix}" for feature in protocol.INPUT_FEATURE_COLUMNS]].to_numpy(float)
            for suffix in source_suffixes
        ],
        axis=1,
    )
    times = pd.to_datetime(merged["time"]).to_numpy(dtype="datetime64[ns]")

    rows = {
        "self_x": [],
        "current": [],
        "target_delta": [],
        "anchor": [],
        "source_x_36h": [],
        "source_step_y": [],
        "origin_time": [],
    }
    history = cfg.GRAPH_INPUT_STEPS
    for origin in range(history - 1, len(merged) - cfg.OUTPUT_STEPS):
        self_x = target_diff[origin - cfg.SELF_INPUT_STEPS + 1 : origin + 1]
        source_x = source_diff[origin - history + 1 : origin + 1].transpose(1, 0, 2)
        source_y = source_diff[origin + 1 : origin + cfg.OUTPUT_STEPS + 1].transpose(1, 0, 2)
        current = target_level[origin]
        future_target = target_level[origin + 1 : origin + cfg.OUTPUT_STEPS + 1]
        target_delta = future_target - current[None, :]
        arrays = (self_x, source_x, source_y, current, target_delta)
        if not all(np.isfinite(value).all() for value in arrays):
            continue
        rows["self_x"].append(self_x)
        rows["current"].append(current)
        rows["target_delta"].append(target_delta)
        rows["anchor"].append(current)
        rows["source_x_36h"].append(source_x)
        rows["source_step_y"].append(source_y)
        rows["origin_time"].append(times[origin])
    if not rows["self_x"]:
        raise ValueError(f"No finite common samples for {sources} -> {target}")
    return {key: np.asarray(value) for key, value in rows.items()}


def build_flow_components(
    gauge_values: dict[str, np.ndarray],
    confluence: cfg.Confluence,
) -> tuple[np.ndarray, np.ndarray]:
    """Build measured or mass-balance-derived branch flows in source order."""
    source_values = []
    for component in confluence.source_components:
        value = np.asarray(gauge_values[component.gauge_code], dtype=float)
        if component.kind == "residual":
            value = np.maximum(value - np.asarray(gauge_values[component.subtract_code], dtype=float), 0.0)
        elif component.kind != "gauge":
            raise ValueError(f"Unsupported flow component kind: {component.kind}")
        source_values.append(value)
    downstream = np.asarray(gauge_values[confluence.target_gauge_code], dtype=float)
    return np.column_stack(source_values), downstream


def edge_lag_supports(
    lag_audit: pd.DataFrame,
    sources: tuple[str, str],
    target: str,
) -> tuple[tuple[int, ...], ...]:
    supports = []
    for source in sources:
        match = lag_audit[
            lag_audit["source_station"].astype(str).eq(source)
            & lag_audit["target_station"].astype(str).eq(target)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one physical-lag row for {source} -> {target}")
        supports.append(delayed_graph.lag_support(int(match.iloc[0]["lag_primary_steps"])))
    return tuple(supports)


def build_lag_audit() -> pd.DataFrame:
    audit = pd.read_csv(v2_physical_lag.EDGE_AUDIT_PATH)
    rows = []
    for confluence in cfg.CONFLUENCES:
        for source in confluence.sources:
            match = audit[
                audit["source_station"].astype(str).eq(source)
                & audit["target_station"].astype(str).eq(confluence.target)
            ]
            if len(match) != 1:
                raise ValueError(f"Missing audited edge distance for {source} -> {confluence.target}")
            row = match.iloc[0]
            rows.append(
                {
                    "source_station": source,
                    "target_station": confluence.target,
                    "relation": row.get("draft_relation", ""),
                    "distance_km": float(row["straight_distance_km"]),
                    "distance_kind": "straight_distance_km",
                }
            )
    output = v2_physical_lag.build_lag_audit(
        pd.DataFrame(rows),
        v2_physical_lag.load_velocity_observations(),
    )
    v2_physical_lag.validate_lag_audit(output)
    cfg.ensure_output_dirs()
    output.to_csv(cfg.LAG_AUDIT_PATH, index=False, encoding="utf-8-sig")
    return output


def prepare_flow_weight_results(
    daily_flow: pd.DataFrame,
    origin_times: np.ndarray,
    confluence: cfg.Confluence,
) -> dict[str, flow_weight.FlowWeightResult]:
    codes = tuple(
        dict.fromkeys(
            [
                confluence.target_gauge_code,
                *(
                    code
                    for component in confluence.source_components
                    for code in (component.gauge_code, component.subtract_code)
                    if code
                ),
            ]
        )
    )
    values = flow_weight.causal_daily_values(
        daily_flow,
        origin_times,
        columns=codes,
        lag_days=1,
        smooth_days=3,
    )
    gauge_values = {code: values[:, idx] for idx, code in enumerate(codes)}
    source, downstream = build_flow_components(gauge_values, confluence)
    return {
        mode: flow_weight.compute_flow_weights(source, downstream, mode)
        for mode in cfg.WEIGHT_MODES
    }


def subset_samples(samples: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    selected = np.asarray(mask, dtype=bool)
    count = len(samples["origin_time"])
    return {
        key: value[selected] if isinstance(value, np.ndarray) and len(value) == count else value
        for key, value in samples.items()
    }


def fit_scalers(
    samples: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[dict[str, stage4.SimpleScaler], tuple[stage4.SimpleScaler, ...]]:
    target_scalers = {
        "self_x": stage4.SimpleScaler.fit(samples["self_x"][train_idx]),
        "current": stage4.SimpleScaler.fit(samples["current"][train_idx]),
        "target": stage4.SimpleScaler.fit(samples["target_delta"][train_idx]),
    }
    source_scalers = []
    for source_idx in range(samples["source_x_36h"].shape[1]):
        values = np.concatenate(
            [
                samples["source_x_36h"][train_idx, source_idx],
                samples["source_step_y"][train_idx, source_idx],
            ],
            axis=1,
        )
        source_scalers.append(stage4.SimpleScaler.fit(values))
    return target_scalers, tuple(source_scalers)


def transform_samples(
    samples: dict[str, np.ndarray],
    target_scalers: dict[str, stage4.SimpleScaler],
    source_scalers: tuple[stage4.SimpleScaler, ...],
) -> dict[str, np.ndarray]:
    source_x = np.stack(
        [
            scaler.transform(samples["source_x_36h"][:, source_idx])
            for source_idx, scaler in enumerate(source_scalers)
        ],
        axis=1,
    )
    source_y = np.stack(
        [
            scaler.transform(samples["source_step_y"][:, source_idx])
            for source_idx, scaler in enumerate(source_scalers)
        ],
        axis=1,
    )
    return {
        **samples,
        "self_x_scaled": target_scalers["self_x"].transform(samples["self_x"]),
        "current_scaled": target_scalers["current"].transform(samples["current"]),
        "target_scaled": target_scalers["target"].transform(samples["target_delta"]),
        "source_x_scaled": source_x,
        "source_y_scaled": source_y,
    }


def train_graph_mapper(
    torch,
    self_prediction: np.ndarray,
    aligned_sources: tuple[np.ndarray, ...],
    edge_weights: np.ndarray,
    target: np.ndarray,
    split: dict[str, np.ndarray],
    device,
):
    mapper = flow_weight.make_multi_source_delayed_mapper(
        torch,
        source_dim=aligned_sources[0].shape[-1],
        target_dim=target.shape[-1],
        lag_counts=tuple(array.shape[-2] for array in aligned_sources),
    ).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: stage4._loader(
            torch,
            (
                self_prediction[idx],
                *(array[idx] for array in aligned_sources),
                edge_weights[idx],
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
            aligned = tuple(value.to(device) for value in batch[1 : 1 + len(aligned_sources)])
            weights = batch[-2].to(device)
            y = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            _, correction = mapper(aligned, weights)
            loss = torch.nn.functional.l1_loss(self_pred + correction, y)
            transfer_l1 = torch.stack(
                [edge_mapper.transfer.weight.abs().mean() for edge_mapper in mapper.edge_mappers]
            ).mean()
            loss = loss + TRANSFER_L1 * transfer_l1
            loss.backward()
            optimizer.step()
        val_loss = evaluate_graph_loss(torch, mapper, loaders["val"], len(aligned_sources), device)
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


def evaluate_graph_loss(torch, mapper, loader, source_count: int, device) -> float:
    mapper.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            self_pred = batch[0].to(device)
            aligned = tuple(value.to(device) for value in batch[1 : 1 + source_count])
            _, correction = mapper(aligned, batch[-2].to(device))
            losses.append(float(torch.nn.functional.l1_loss(self_pred + correction, batch[-1].to(device)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def graph_prediction_scaled(
    torch,
    mapper,
    self_prediction: np.ndarray,
    aligned_sources: tuple[np.ndarray, ...],
    edge_weights: np.ndarray,
    device,
) -> np.ndarray:
    loader = stage4._loader(torch, (self_prediction, *aligned_sources, edge_weights), False)
    predictions = []
    mapper.eval()
    with torch.no_grad():
        for batch in loader:
            self_pred = batch[0].to(device)
            aligned = tuple(value.to(device) for value in batch[1:-1])
            _, correction = mapper(aligned, batch[-1].to(device))
            predictions.append((self_pred + correction).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def parameter_rows(
    mapper,
    confluence: cfg.Confluence,
    variant: str,
    supports: tuple[tuple[int, ...], ...],
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for source, support, edge_mapper in zip(confluence.sources, supports, mapper.edge_mappers):
        for lag, value in zip(support, edge_mapper.lag_weights().detach().cpu().numpy()):
            rows.append(
                {
                    "seed": seed,
                    "source_station": source,
                    "target_station": confluence.target,
                    "variant": variant,
                    "parameter_type": "lag_weight",
                    "lag_steps": lag,
                    "lag_hours": lag * 4,
                    "source_feature": "",
                    "target_feature": "",
                    "value": float(value),
                }
            )
        transfer = edge_mapper.transfer.weight.detach().cpu().numpy()
        for target_idx, target_feature in enumerate(protocol.TARGET_FEATURE_COLUMNS):
            for source_idx, source_feature in enumerate(protocol.INPUT_FEATURE_COLUMNS):
                rows.append(
                    {
                        "seed": seed,
                        "source_station": source,
                        "target_station": confluence.target,
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


def flow_diagnostic_rows(
    confluence: cfg.Confluence,
    weight_results: dict[str, flow_weight.FlowWeightResult],
    split: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for mode, result in weight_results.items():
        for split_name, idx in split.items():
            for source_idx, source in enumerate(confluence.sources):
                values = result.weights[idx, source_idx]
                rows.append(
                    {
                        "target_station": confluence.target,
                        "source_station": source,
                        "mode": mode,
                        "split": split_name,
                        "sample_count": int(len(values)),
                        "mean_weight": float(np.mean(values)),
                        "median_weight": float(np.median(values)),
                        "p10_weight": float(np.quantile(values, 0.1)),
                        "p90_weight": float(np.quantile(values, 0.9)),
                        "median_unobserved_fraction": float(np.median(result.unobserved_fraction[idx])),
                    }
                )
    return rows


def run_confluence_suite(torch, data, daily_flow, lag_audit, confluence, seed, device):
    samples = build_confluence_samples(data, confluence.sources, confluence.target)
    weight_results = prepare_flow_weight_results(daily_flow, samples["origin_time"], confluence)
    valid = weight_results["mass_balance"].valid
    samples = subset_samples(samples, valid)
    weight_results = {
        mode: flow_weight.FlowWeightResult(
            result.weights[valid],
            result.unobserved_fraction[valid],
            result.valid[valid],
        )
        for mode, result in weight_results.items()
    }
    split = stage4b.split_indices(samples["origin_time"])
    if any(len(idx) == 0 for idx in split.values()):
        raise ValueError(f"Incomplete split coverage for {confluence.target}")
    target_scalers, source_scalers = fit_scalers(samples, split["train"])
    samples = transform_samples(samples, target_scalers, source_scalers)

    self_samples = {**samples, "y_scaled": samples["target_scaled"]}
    self_model = stage4.train_self_model(torch, self_samples, split, device)
    self_prediction = stage4b.predict_self_scaled(torch, self_model, samples, device)

    supports = edge_lag_supports(lag_audit, confluence.sources, confluence.target)
    aligned_sources = []
    for source_idx, support in enumerate(supports):
        source_model = stage4b.train_source_model(
            torch,
            samples["source_x_scaled"][:, source_idx],
            samples["source_y_scaled"][:, source_idx],
            split,
            device,
        )
        source_future = stage4b.predict_source(
            torch,
            source_model,
            samples["source_x_scaled"][:, source_idx],
            device,
        )
        aligned, _known = stage4b.build_aligned_batch(
            samples["source_x_scaled"][:, source_idx],
            source_future,
            support,
        )
        aligned_sources.append(aligned)
    aligned_tuple = tuple(aligned_sources)

    predictions = {"self_D_6to9": self_prediction}
    parameters = []
    for mode in cfg.WEIGHT_MODES:
        variant = f"graph_{mode}"
        torch.manual_seed(seed)
        mapper = train_graph_mapper(
            torch,
            self_prediction,
            aligned_tuple,
            weight_results[mode].weights.astype(np.float32),
            samples["target_scaled"],
            split,
            device,
        )
        predictions[variant] = graph_prediction_scaled(
            torch,
            mapper,
            self_prediction,
            aligned_tuple,
            weight_results[mode].weights.astype(np.float32),
            device,
        )
        parameters.extend(parameter_rows(mapper, confluence, variant, supports, seed))

    metrics = []
    source_label = "+".join(confluence.sources)
    for variant, prediction in predictions.items():
        for split_name, idx in split.items():
            metrics.extend(
                stage4b.metric_rows(
                    source_label,
                    confluence.target,
                    variant,
                    split_name,
                    prediction,
                    samples,
                    idx,
                    target_scalers["target"],
                    seed,
                )
            )
    coverage = {
        "target_station": confluence.target,
        "source_stations": list(confluence.sources),
        "sample_count": int(len(samples["origin_time"])),
        "split_counts": {name: int(len(idx)) for name, idx in split.items()},
        "lag_support_steps": {source: list(support) for source, support in zip(confluence.sources, supports)},
    }
    diagnostics = flow_diagnostic_rows(confluence, weight_results, split)
    return metrics, parameters, diagnostics, coverage


def write_report(overall: pd.DataFrame, diagnostics: pd.DataFrame, coverage, output_path: Path) -> None:
    validation = overall[overall["split"].eq("val")].sort_values(["target_station", "mean_rmse"])
    test = overall[overall["split"].eq("test")].sort_values(["target_station", "mean_rmse"])
    lines = [
        "# V2 Stage 4c Flow Mass-Balance Ablation",
        "",
        "- Self input: 24h; direct output: 4-36h.",
        "- Flow lookup: prior complete day, rolling 3-day median.",
        "- Flow enters only as an edge-message weight.",
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
        "",
        "## Flow Diagnostics",
        "",
        "```text",
        diagnostics.to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_seed(seed: int, output_dir: Path, dry_run: bool = False) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_audit = build_lag_audit()
    data = stage4b.load_v2_data()
    daily_flow = hydromet_features.read_daily_flow()
    if dry_run:
        for confluence in cfg.CONFLUENCES:
            samples = build_confluence_samples(data, confluence.sources, confluence.target)
            weights = prepare_flow_weight_results(daily_flow, samples["origin_time"], confluence)
            console.print(
                confluence.target,
                len(samples["origin_time"]),
                int(weights["mass_balance"].valid.sum()),
                edge_lag_supports(lag_audit, confluence.sources, confluence.target),
                flush=True,
            )
        return output_dir

    metric_rows = []
    parameter_output = []
    diagnostic_output = []
    coverage = []
    for confluence in cfg.CONFLUENCES:
        console.print(f"running {'+'.join(confluence.sources)} -> {confluence.target}", flush=True)
        metrics, parameters, diagnostics, item_coverage = run_confluence_suite(
            torch, data, daily_flow, lag_audit, confluence, seed, device
        )
        metric_rows.extend(metrics)
        parameter_output.extend(parameters)
        diagnostic_output.extend(diagnostics)
        coverage.append(item_coverage)

    feature_horizon = pd.DataFrame(metric_rows)
    overall = stage4.summarize_overall(feature_horizon)
    diagnostics = pd.DataFrame(diagnostic_output)
    feature_horizon.to_csv(output_dir / "feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "flow_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(parameter_output).to_csv(
        output_dir / "transfer_parameters.csv", index=False, encoding="utf-8-sig"
    )
    manifest = protocol.build_run_manifest(
        experiment="stage4c_flow_mass_balance",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/graph/v2_flow_mass_balance.py"),
            Path("scripts/graph/v2_flow_mass_balance_config.py"),
            Path("scripts/graph/run_v2_flow_mass_balance_ablation.py"),
        ),
    )
    manifest.update(
        {
            "device": str(device),
            "self_input_steps": cfg.SELF_INPUT_STEPS,
            "graph_input_steps": cfg.GRAPH_INPUT_STEPS,
            "output_steps": cfg.OUTPUT_STEPS,
            "weight_modes": list(cfg.WEIGHT_MODES),
            "flow_lag_days": 1,
            "flow_smooth_days": 3,
            "coverage": coverage,
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(overall, diagnostics, coverage, output_dir / "run_report.md")
    return output_dir


def pilot_supports_formal(overall: pd.DataFrame) -> bool:
    validation = overall[overall["split"].astype(str).eq("val")]
    winning_targets = 0
    for confluence in cfg.CONFLUENCES:
        values = validation[validation["target_station"].astype(str).eq(confluence.target)].set_index("variant")[
            "mean_rmse"
        ].to_dict()
        required = {"self_D_6to9", "graph_branch_normalized", "graph_mass_balance"}
        if required.issubset(values) and float(values["graph_mass_balance"]) < min(
            float(values["self_D_6to9"]), float(values["graph_branch_normalized"])
        ):
            winning_targets += 1
    return winning_targets >= 2


def summarize_multiseed(overall: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "target_station", "split"]
    baseline = overall[overall["variant"].eq("self_D_6to9")][keys + ["mean_rmse"]].rename(
        columns={"mean_rmse": "self_rmse"}
    )
    compared = overall.merge(baseline, on=keys, how="left", validate="many_to_one")
    compared["delta_rmse_minus_self"] = compared["mean_rmse"] - compared["self_rmse"]
    rows = []
    for values, group in compared.groupby(["target_station", "variant", "split"], sort=True):
        delta = pd.to_numeric(group["delta_rmse_minus_self"], errors="coerce")
        rows.append(
            {
                "target_station": values[0],
                "variant": values[1],
                "split": values[2],
                "seed_count": int(delta.notna().sum()),
                "mean_rmse": float(pd.to_numeric(group["mean_rmse"], errors="coerce").mean()),
                "mean_delta_rmse_minus_self": float(delta.mean()),
                "win_seed_count": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_formal() -> Path:
    pilot_path = cfg.PILOT_DIR / "overall_metrics.csv"
    if not pilot_path.exists() or not pilot_supports_formal(pd.read_csv(pilot_path)):
        raise ValueError("Seed-42 validation gate did not support formal multiseed training.")
    cfg.ensure_output_dirs()
    overall_frames = []
    for seed in cfg.FORMAL_SEEDS:
        output_dir = cfg.FORMAL_DIR / f"seed_{seed}"
        run_seed(seed, output_dir)
        overall_frames.append(pd.read_csv(output_dir / "overall_metrics.csv"))
    overall = pd.concat(overall_frames, ignore_index=True)
    summary = summarize_multiseed(overall)
    overall.to_csv(cfg.FORMAL_DIR / "overall_all_seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(cfg.FORMAL_DIR / "multiseed_summary.csv", index=False, encoding="utf-8-sig")
    return cfg.FORMAL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.PILOT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.formal:
        output = run_formal()
    elif args.pilot or args.dry_run:
        cfg.ensure_output_dirs()
        output = run_seed(args.seed, cfg.PILOT_DIR, dry_run=args.dry_run)
    else:
        raise SystemExit("Choose --pilot, --formal, or --dry-run.")
    console.print(f"saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
