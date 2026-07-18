#!/usr/bin/env python3
"""Run the focused constrained flow-transport ablation for Jiangjunyan."""

from __future__ import annotations

from scripts.common.terminal_output import console

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import hydromet_features
from scripts.graph import run_v2_delayed_step_graph_ablation as stage4b
from scripts.graph import run_v2_direct_pair_graph_ablation as stage4
from scripts.graph import run_v2_flow_mass_balance_ablation as stage4c
from scripts.common import v2_experiment_protocol as protocol
from scripts.graph import v2_flow_constrained_transport as transport
from scripts.graph import v2_flow_constrained_transport_config as cfg
from scripts.graph import v2_flow_mass_balance as flow_weight


MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 1e-3
RETENTION_L1 = 1e-4


def focus_confluence():
    return next(
        item
        for item in stage4c.cfg.CONFLUENCES
        if item.target == cfg.TARGET and item.sources == cfg.SOURCES
    )


def prepare_weight_results(
    daily_flow: pd.DataFrame,
    origin_times: np.ndarray,
    smooth_days: int,
) -> dict[str, flow_weight.FlowWeightResult]:
    confluence = focus_confluence()
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
        smooth_days=int(smooth_days),
    )
    gauge_values = {code: values[:, idx] for idx, code in enumerate(codes)}
    source, downstream = stage4c.build_flow_components(gauge_values, confluence)
    return {
        mode: flow_weight.compute_flow_weights(source, downstream, mode)
        for mode in flow_weight.WEIGHT_MODES
    }


def subset_weight_result(
    result: flow_weight.FlowWeightResult,
    mask: np.ndarray,
) -> flow_weight.FlowWeightResult:
    return flow_weight.FlowWeightResult(
        weights=result.weights[mask],
        unobserved_fraction=result.unobserved_fraction[mask],
        valid=result.valid[mask],
    )


def train_transport_model(
    torch,
    self_prediction: np.ndarray,
    aligned_raw: tuple[np.ndarray, ...],
    edge_weights: np.ndarray,
    target_scaled: np.ndarray,
    target_input_indices: tuple[int, ...],
    target_scale: np.ndarray,
    split: dict[str, np.ndarray],
    device,
):
    model = transport.make_constrained_transport(
        torch,
        target_input_indices=target_input_indices,
        target_scale=target_scale,
        lag_counts=tuple(values.shape[-2] for values in aligned_raw),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loaders = {
        name: stage4._loader(
            torch,
            (
                self_prediction[idx],
                *(values[idx] for values in aligned_raw),
                edge_weights[idx],
                target_scaled[idx],
            ),
            name == "train",
        )
        for name, idx in split.items()
    }
    best_state = None
    best_val = float("inf")
    wait = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for batch in loaders["train"]:
            self_pred = batch[0].to(device)
            aligned = tuple(value.to(device) for value in batch[1 : 1 + len(aligned_raw)])
            weights = batch[-2].to(device)
            y = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            _, correction = model(aligned, weights)
            loss = torch.nn.functional.l1_loss(self_pred + correction, y)
            loss = loss + RETENTION_L1 * model.retention().mean()
            loss.backward()
            optimizer.step()
        val_loss = evaluate_transport_loss(torch, model, loaders["val"], len(aligned_raw), device)
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


def evaluate_transport_loss(torch, model, loader, source_count: int, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            aligned = tuple(value.to(device) for value in batch[1 : 1 + source_count])
            _, correction = model(aligned, batch[-2].to(device))
            loss = torch.nn.functional.l1_loss(batch[0].to(device) + correction, batch[-1].to(device))
            losses.append(float(loss.cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict_transport(
    torch,
    model,
    self_prediction: np.ndarray,
    aligned_raw: tuple[np.ndarray, ...],
    edge_weights: np.ndarray,
    device,
) -> np.ndarray:
    loader = stage4._loader(torch, (self_prediction, *aligned_raw, edge_weights), False)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            aligned = tuple(value.to(device) for value in batch[1:-1])
            _, correction = model(aligned, batch[-1].to(device))
            rows.append((batch[0].to(device) + correction).cpu().numpy())
    return np.concatenate(rows, axis=0)


def transport_parameter_rows(model, supports, seed: int, variant: str) -> list[dict[str, object]]:
    rows = []
    retention = model.retention().detach().cpu().numpy()
    for edge_idx, (source, support) in enumerate(zip(cfg.SOURCES, supports)):
        for lag, value in zip(support, model.lag_weights(edge_idx).detach().cpu().numpy()):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "source_station": source,
                    "target_station": cfg.TARGET,
                    "parameter_type": "lag_weight",
                    "feature": "",
                    "lag_steps": lag,
                    "lag_hours": lag * 4,
                    "value": float(value),
                }
            )
        for feature, value in zip(protocol.TARGET_FEATURE_COLUMNS, retention[edge_idx]):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "source_station": source,
                    "target_station": cfg.TARGET,
                    "parameter_type": "retention",
                    "feature": feature,
                    "lag_steps": "",
                    "lag_hours": "",
                    "value": float(value),
                }
            )
    return rows


def weight_diagnostic_rows(weight_variants, split) -> list[dict[str, object]]:
    rows = []
    for variant, weights in weight_variants.items():
        for split_name, idx in split.items():
            for source_idx, source in enumerate(cfg.SOURCES):
                values = weights[idx, source_idx]
                rows.append(
                    {
                        "variant": variant,
                        "split": split_name,
                        "source_station": source,
                        "target_station": cfg.TARGET,
                        "sample_count": int(len(values)),
                        "mean_weight": float(np.mean(values)),
                        "median_weight": float(np.median(values)),
                        "p10_weight": float(np.quantile(values, 0.1)),
                        "p90_weight": float(np.quantile(values, 0.9)),
                    }
                )
    return rows


def run_suite(torch, seed: int, device):
    confluence = focus_confluence()
    data = stage4b.load_v2_data()
    daily_flow = hydromet_features.read_daily_flow()
    lag_audit = stage4c.build_lag_audit()
    samples = stage4c.build_confluence_samples(data, cfg.SOURCES, cfg.TARGET)
    weights_3d = prepare_weight_results(daily_flow, samples["origin_time"], smooth_days=3)
    weights_1d = prepare_weight_results(daily_flow, samples["origin_time"], smooth_days=1)
    valid = weights_3d["mass_balance"].valid & weights_1d["mass_balance"].valid
    samples = stage4c.subset_samples(samples, valid)
    weights_3d = {mode: subset_weight_result(result, valid) for mode, result in weights_3d.items()}
    weights_1d = {mode: subset_weight_result(result, valid) for mode, result in weights_1d.items()}
    split = stage4b.split_indices(samples["origin_time"])
    if any(len(idx) == 0 for idx in split.values()):
        raise ValueError("Incomplete train/validation/test coverage")

    target_scalers, source_scalers = stage4c.fit_scalers(samples, split["train"])
    samples = stage4c.transform_samples(samples, target_scalers, source_scalers)
    self_samples = {**samples, "y_scaled": samples["target_scaled"]}
    self_model = stage4.train_self_model(torch, self_samples, split, device)
    self_prediction = stage4b.predict_self_scaled(torch, self_model, samples, device)

    supports = stage4c.edge_lag_supports(lag_audit, cfg.SOURCES, cfg.TARGET)
    aligned_scaled = []
    aligned_raw = []
    for source_idx, support in enumerate(supports):
        source_model = stage4b.train_source_model(
            torch,
            samples["source_x_scaled"][:, source_idx],
            samples["source_y_scaled"][:, source_idx],
            split,
            device,
        )
        future_scaled = stage4b.predict_source(
            torch,
            source_model,
            samples["source_x_scaled"][:, source_idx],
            device,
        )
        scaled, _ = stage4b.build_aligned_batch(
            samples["source_x_scaled"][:, source_idx], future_scaled, support
        )
        future_raw = source_scalers[source_idx].inverse(future_scaled)
        raw, _ = stage4b.build_aligned_batch(
            samples["source_x_36h"][:, source_idx], future_raw, support
        )
        aligned_scaled.append(scaled)
        aligned_raw.append(raw)
    aligned_scaled = tuple(aligned_scaled)
    aligned_raw = tuple(aligned_raw)

    mass_3d = weights_3d["mass_balance"].weights.astype(np.float32)
    weight_variants = {
        "unweighted": weights_3d["unweighted"].weights.astype(np.float32),
        "branch_normalized": weights_3d["branch_normalized"].weights.astype(np.float32),
        "mass_balance_3d": mass_3d,
        "mass_balance_raw1d": weights_1d["mass_balance"].weights.astype(np.float32),
        "mass_static": transport.static_train_weights(mass_3d, split["train"]),
        "mass_shuffled": transport.shuffle_daily_weights(
            mass_3d, samples["origin_time"], split, seed
        ),
    }

    predictions = {"self_D_6to9": self_prediction}
    parameters = []
    for suffix in ("unweighted", "branch_normalized", "mass_balance_3d"):
        variant = f"full_{suffix}"
        torch.manual_seed(seed)
        model = stage4c.train_graph_mapper(
            torch,
            self_prediction,
            aligned_scaled,
            weight_variants[suffix],
            samples["target_scaled"],
            split,
            device,
        )
        predictions[variant] = stage4c.graph_prediction_scaled(
            torch, model, self_prediction, aligned_scaled, weight_variants[suffix], device
        )
        parameters.extend(stage4c.parameter_rows(model, confluence, variant, supports, seed))

    constrained_variants = {
        "transport_unweighted": "unweighted",
        "transport_branch_normalized": "branch_normalized",
        "transport_mass_balance_3d": "mass_balance_3d",
        "transport_mass_balance_raw1d": "mass_balance_raw1d",
        "transport_mass_static": "mass_static",
        "transport_mass_shuffled": "mass_shuffled",
    }
    target_input_indices = tuple(
        protocol.INPUT_FEATURE_COLUMNS.index(feature)
        for feature in protocol.TARGET_FEATURE_COLUMNS
    )
    for variant, weight_name in constrained_variants.items():
        torch.manual_seed(seed)
        model = train_transport_model(
            torch,
            self_prediction,
            aligned_raw,
            weight_variants[weight_name],
            samples["target_scaled"],
            target_input_indices,
            target_scalers["target"].scale,
            split,
            device,
        )
        predictions[variant] = predict_transport(
            torch, model, self_prediction, aligned_raw, weight_variants[weight_name], device
        )
        parameters.extend(transport_parameter_rows(model, supports, seed, variant))

    metrics = []
    source_label = "+".join(cfg.SOURCES)
    for variant, prediction in predictions.items():
        for split_name, idx in split.items():
            metrics.extend(
                stage4b.metric_rows(
                    source_label,
                    cfg.TARGET,
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
        "source_stations": list(cfg.SOURCES),
        "target_station": cfg.TARGET,
        "sample_count": int(len(samples["origin_time"])),
        "split_counts": {name: int(len(idx)) for name, idx in split.items()},
        "lag_support_steps": {source: list(value) for source, value in zip(cfg.SOURCES, supports)},
    }
    return metrics, parameters, weight_diagnostic_rows(weight_variants, split), coverage


def select_dynamic_variant(overall: pd.DataFrame) -> str:
    validation = overall[
        overall["split"].astype(str).eq("val")
        & overall["target_station"].astype(str).eq(cfg.TARGET)
        & overall["variant"].isin(
            ("transport_mass_balance_3d", "transport_mass_balance_raw1d")
        )
    ]
    if len(validation) != 2:
        raise ValueError("Both dynamic flow variants are required on validation")
    return str(validation.sort_values(["mean_rmse", "variant"]).iloc[0]["variant"])


def pilot_supports_formal(overall: pd.DataFrame) -> bool:
    selected = select_dynamic_variant(overall)
    validation = overall[
        overall["split"].astype(str).eq("val")
        & overall["target_station"].astype(str).eq(cfg.TARGET)
    ].set_index("variant")["mean_rmse"].to_dict()
    controls = (
        "self_D_6to9",
        "full_mass_balance_3d",
        "transport_branch_normalized",
        "transport_mass_static",
        "transport_mass_shuffled",
    )
    return all(name in validation for name in controls) and float(validation[selected]) < min(
        float(validation[name]) for name in controls
    )


def write_report(overall, parameters, diagnostics, coverage, output_path: Path) -> None:
    selected = select_dynamic_variant(overall)
    gate = pilot_supports_formal(overall)
    lines = [
        "# V2 Stage 4d Constrained Flow Transport",
        "",
        f"- Focus: {' + '.join(cfg.SOURCES)} -> {cfg.TARGET}.",
        "- Self input: 24h; direct output: 4-36h.",
        "- Transport: raw-unit same-feature changes with retention in [0, 1].",
        f"- Validation-selected dynamic flow: {selected}.",
        f"- Formal gate: {gate}.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Validation",
        "",
        "```text",
        overall[overall["split"].eq("val")].sort_values("mean_rmse").to_string(index=False),
        "```",
        "",
        "## Test",
        "",
        "```text",
        overall[overall["split"].eq("test")].sort_values("mean_rmse").to_string(index=False),
        "```",
        "",
        "## Retention",
        "",
        "```text",
        parameters[parameters["parameter_type"].eq("retention")].to_string(index=False),
        "```",
        "",
        "## Flow Weights",
        "",
        "```text",
        diagnostics.to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_seed(seed: int, output_dir: Path) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, parameters, diagnostics, coverage = run_suite(torch, seed, device)
    feature_horizon = pd.DataFrame(metrics)
    overall = stage4.summarize_overall(feature_horizon)
    parameters = pd.DataFrame(parameters)
    diagnostics = pd.DataFrame(diagnostics)
    feature_horizon.to_csv(output_dir / "feature_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(output_dir / "transport_parameters.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "flow_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    manifest = protocol.build_run_manifest(
        experiment="stage4d_constrained_flow_transport",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/graph/v2_flow_constrained_transport.py"),
            Path("scripts/graph/v2_flow_constrained_transport_config.py"),
            Path("scripts/graph/run_v2_flow_constrained_transport_ablation.py"),
        ),
    )
    manifest.update(
        {
            "device": str(device),
            "variants": list(cfg.VARIANTS),
            "selected_dynamic_variant": select_dynamic_variant(overall),
            "formal_gate": pilot_supports_formal(overall),
            "flow_lag_days": 1,
            "dynamic_smooth_days": [1, 3],
            "coverage": coverage,
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(overall, parameters, diagnostics, coverage, output_dir / "run_report.md")
    return output_dir


def run_formal() -> Path:
    pilot_path = cfg.PILOT_DIR / "overall_metrics.csv"
    if not pilot_path.exists() or not pilot_supports_formal(pd.read_csv(pilot_path)):
        raise ValueError("Seed-42 validation gate did not support formal multiseed training")
    cfg.ensure_output_dirs()
    frames = []
    for seed in cfg.FORMAL_SEEDS:
        output_dir = cfg.FORMAL_DIR / f"seed_{seed}"
        run_seed(seed, output_dir)
        frames.append(pd.read_csv(output_dir / "overall_metrics.csv"))
    overall = pd.concat(frames, ignore_index=True)
    summary = stage4b.summarize_multiseed(overall)
    overall.to_csv(cfg.FORMAL_DIR / "overall_all_seeds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(cfg.FORMAL_DIR / "multiseed_summary.csv", index=False, encoding="utf-8-sig")
    return cfg.FORMAL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.PILOT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg.ensure_output_dirs()
    if args.formal:
        output = run_formal()
    elif args.pilot:
        output = run_seed(args.seed, cfg.PILOT_DIR)
    else:
        raise SystemExit("Choose --pilot or --formal")
    console.print(f"saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
