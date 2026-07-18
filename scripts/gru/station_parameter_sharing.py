#!/usr/bin/env python3
"""Reusable models, training, metrics, and reports for station parameter sharing."""

from __future__ import annotations

from scripts.common.terminal_output import console

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.baselines import gat_gru_baseline as base
from scripts.baselines import gat_gru_paper_style as paper
from scripts.gru import run_all_station_window_level_ablation as all_window
from scripts.gru import run_wentu_dual_branch_delta_gru as dual
from scripts.gru import run_wentu_self_feature_ablation as metric_helpers
from scripts.gru import run_wentu_window_level_ablation as window
from scripts.common import v2_experiment_protocol as protocol

# [01] Only parameter sharing changes; data, features, scalers, loss and split stay fixed.
OUTPUT_ROOT = protocol.GRU_OUTPUT_ROOT / "stage3b_station_parameter_sharing"
INPUT_STEPS = 9
OUTPUT_STEPS = 1
STATION_EMBED_DIM = 8
VARIANTS = ("shared_D", "shared_D_station_embedding", "local_D")
LOCAL_NODE_AXIS = {
    "self_x": 2,
    "self_mask": 2,
    "y": 2,
    "y_abs": 2,
    "y_mask": 2,
    "upstream_x": 1,
    "upstream_mask": 1,
    "last_target": 1,
    "current_level": 1,
    "upstream_time": 1,
}


def output_dir_for_seed(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed_{int(seed)}"


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def slice_station_split(split: dict[str, np.ndarray], station_idx: int) -> dict[str, np.ndarray]:
    """[02] Keep one node while preserving the global time rows and global scaling."""
    sliced: dict[str, np.ndarray] = {}
    for key, value in split.items():
        if key not in LOCAL_NODE_AXIS:
            sliced[key] = value
            continue
        axis = LOCAL_NODE_AXIS[key]
        index = [slice(None)] * value.ndim
        index[axis] = slice(station_idx, station_idx + 1)
        sliced[key] = value[tuple(index)]
    return sliced


def make_station_embedding_model(
    torch,
    sequence_input_dim: int,
    current_input_dim: int,
    station_count: int,
    output_steps: int = OUTPUT_STEPS,
    station_embed_dim: int = STATION_EMBED_DIM,
):
    """[04] D backbone plus a learned station identity vector; no cross-station mixing."""

    class StationEmbeddingDualBranch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = paper.GRU_DROPOUT if paper.NUM_GRU_LAYERS > 1 else 0.0
            self.sequence_encoder = torch.nn.GRU(
                input_size=sequence_input_dim,
                hidden_size=paper.HIDDEN_SIZE,
                num_layers=paper.NUM_GRU_LAYERS,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(current_input_dim, dual.CURRENT_HIDDEN_SIZE),
                torch.nn.ReLU(),
                torch.nn.Linear(dual.CURRENT_HIDDEN_SIZE, paper.HIDDEN_SIZE),
                torch.nn.ReLU(),
            )
            self.station_embedding = torch.nn.Embedding(station_count, station_embed_dim)
            self.dropout = torch.nn.Dropout(paper.HEAD_DROPOUT)
            self.head = torch.nn.Linear(
                paper.HIDDEN_SIZE * 2 + station_embed_dim,
                output_steps,
            )

        def forward(self, sequence_x, current_level):
            batch_size, steps, node_count, _ = sequence_x.shape
            if node_count != self.station_embedding.num_embeddings:
                raise ValueError(
                    f"Expected {self.station_embedding.num_embeddings} stations, got {node_count}."
                )
            encoded_input = sequence_x.permute(0, 2, 1, 3).reshape(batch_size * node_count, steps, -1)
            encoded, _ = self.sequence_encoder(encoded_input)
            sequence_state = encoded[:, -1, :].reshape(batch_size, node_count, paper.HIDDEN_SIZE)
            current_state = self.current_encoder(current_level)
            station_ids = torch.arange(node_count, device=sequence_x.device)
            station_state = self.station_embedding(station_ids).unsqueeze(0).expand(batch_size, -1, -1)
            state = torch.cat([sequence_state, current_state, station_state], dim=-1)
            prediction = self.head(self.dropout(state)).reshape(batch_size, node_count, output_steps, 1)
            return prediction.permute(0, 2, 1, 3).contiguous()

    return StationEmbeddingDualBranch()


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _cleanup_device(torch, device) -> None:
    gc.collect()
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def fit_one_model(
    torch,
    variant: str,
    model,
    scaled_splits: dict[str, dict[str, np.ndarray]],
    scalers: base.GraphForecastScalers,
    stations: tuple[str, ...],
    target_feature: str,
    checkpoint_path: Path,
    seed: int,
    device,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]], int]:
    """[05] Train one model with the same masked L1 and absolute-RMSE early stopping as D."""
    seed_everything(torch, seed)
    model = model.to(device)
    loaders = {
        "train": dual.make_dual_loader(torch, scaled_splits["train"], shuffle=True),
        "val": dual.make_dual_loader(torch, scaled_splits["val"], shuffle=False),
        "test": dual.make_dual_loader(torch, scaled_splits["test"], shuffle=False),
    }
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
    history: list[dict[str, object]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, paper.MAX_EPOCHS + 1):
        train_loss = dual.train_dual_epoch(torch, model, loaders["train"], optimizer, loss_fn, device)
        val_metrics = dual.evaluate_dual_target_model(
            torch,
            model,
            loaders["val"],
            scaled_splits["val"],
            scalers,
            stations,
            target_feature,
            device,
        )
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        scheduler.step(val_rmse)
        improved = val_rmse < best_rmse - paper.MIN_DELTA
        history.append(
            {
                "variant": variant,
                "target": target_feature,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
            }
        )
        if improved:
            best_rmse = val_rmse
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= paper.EARLY_STOPPING_PATIENCE:
                break

    if not checkpoint_path.exists():
        raise RuntimeError(f"No valid checkpoint was produced for {variant}/{target_feature}.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    arrays = {
        split: dual.collect_dual_target_arrays(
            torch,
            model,
            loaders[split],
            scaled_splits[split],
            scalers,
            device,
        )
        for split in ("val", "test")
    }
    count = parameter_count(model)
    del model, optimizer, scheduler, loaders
    _cleanup_device(torch, device)
    return arrays, history, count


def empty_prediction_arrays(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    shape = split["y_abs"].shape
    return {
        "pred": np.full(shape, np.nan, dtype=float),
        "true": np.asarray(split["y_abs"], dtype=float).copy(),
        "mask": np.zeros(shape, dtype=bool),
    }


def assign_local_prediction(
    combined: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    station_idx: int,
) -> None:
    combined["pred"][:, :, station_idx : station_idx + 1, :] = local["pred"]
    combined["true"][:, :, station_idx : station_idx + 1, :] = local["true"]
    combined["mask"][:, :, station_idx : station_idx + 1, :] = local["mask"]


def metric_tables(
    arrays_by_variant: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    stations: tuple[str, ...],
    targets: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """[06] Pooled, equal-station macro, feature, and station-feature metrics."""
    summary_rows = []
    station_rows = []
    feature_rows = []
    station_feature_rows = []
    for variant, by_split in arrays_by_variant.items():
        for split, arrays_by_target in by_split.items():
            metrics = metric_helpers.aggregate_single_target_arrays(arrays_by_target, stations, targets)
            station_rmse = [
                item["rmse"] for item in metrics["station_metrics"].values() if item.get("rmse") is not None
            ]
            station_nse = [
                item["nse"] for item in metrics["station_metrics"].values() if item.get("nse") is not None
            ]
            summary_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "valid_points": metrics["valid_points"],
                    "pooled_mae": metrics["mae"],
                    "pooled_rmse": metrics["rmse"],
                    "pooled_nse": metrics["nse"],
                    "macro_station_rmse": float(np.mean(station_rmse)) if station_rmse else None,
                    "macro_station_nse": float(np.mean(station_nse)) if station_nse else None,
                    "evaluable_stations": len(station_rmse),
                }
            )
            for feature in targets:
                feature_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "feature": feature,
                        "valid_points": metrics["feature_valid_points"][feature],
                        "mae": metrics["feature_mae"][feature],
                        "rmse": metrics["feature_rmse"][feature],
                        "nse": metrics["feature_nse"][feature],
                    }
                )
            for station, item in metrics["station_metrics"].items():
                station_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "station": station,
                        "valid_points": item["valid_points"],
                        "mae": item["mae"],
                        "rmse": item["rmse"],
                        "nse": item["nse"],
                    }
                )
                for feature in targets:
                    station_feature_rows.append(
                        {
                            "variant": variant,
                            "split": split,
                            "station": station,
                            "feature": feature,
                            "valid_points": item["feature_valid_points"][feature],
                            "mae": item["feature_mae"][feature],
                            "rmse": item["feature_rmse"][feature],
                            "nse": item["feature_nse"][feature],
                        }
                    )
    return tuple(
        pd.DataFrame(rows)
        for rows in (summary_rows, station_rows, feature_rows, station_feature_rows)
    )


def comparison_table(station_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        frame = station_metrics[station_metrics["split"].eq(split)].pivot(
            index="station", columns="variant", values="rmse"
        )
        if "shared_D" not in frame:
            continue
        for variant in sorted(set(frame.columns) - {"shared_D"}):
            pair = frame[["shared_D", variant]].dropna()
            delta = pair[variant] - pair["shared_D"]
            pct = delta / pair["shared_D"] * 100
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "evaluable_stations": len(pair),
                    "winning_stations": int(delta.lt(0).sum()),
                    "losing_stations": int(delta.gt(0).sum()),
                    "mean_rmse_delta_vs_shared": float(delta.mean()),
                    "median_rmse_pct_delta_vs_shared": float(pct.median()),
                }
            )
    return pd.DataFrame(rows)


def validation_decision(summary: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, object]:
    """[07] Predeclared pilot gate; test rows are not used."""
    val = summary[summary["split"].eq("val")].set_index("variant")
    station = comparison[comparison["split"].eq("val")].set_index("variant")
    shared = val.loc["shared_D"]
    decisions = {}
    for variant in sorted(set(val.index) - {"shared_D"}):
        candidate = val.loc[variant]
        wins = int(station.loc[variant, "winning_stations"])
        evaluable = int(station.loc[variant, "evaluable_stations"])
        approved = bool(
            candidate["macro_station_rmse"] < shared["macro_station_rmse"]
            and candidate["pooled_rmse"] <= shared["pooled_rmse"] * 1.005
            and wins >= (evaluable // 2 + 1)
        )
        decisions[variant] = {
            "approved_for_multiseed": approved,
            "macro_station_rmse_delta": float(
                candidate["macro_station_rmse"] - shared["macro_station_rmse"]
            ),
            "pooled_rmse_delta": float(candidate["pooled_rmse"] - shared["pooled_rmse"]),
            "winning_stations": wins,
            "evaluable_stations": evaluable,
        }
    return {
        "uses_test": False,
        "gate": (
            "validation macro-station RMSE improves, pooled RMSE is no more than 0.5% worse, "
            "and the candidate wins a majority of stations"
        ),
        "variants": decisions,
    }


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    feature: pd.DataFrame,
    comparison: pd.DataFrame,
    decision: dict[str, object],
    parameter_rows: list[dict[str, object]],
) -> None:
    val = summary[summary["split"].eq("val")].sort_values("macro_station_rmse")
    test = summary[summary["split"].eq("test")].sort_values("macro_station_rmse")
    val_features = feature[feature["split"].eq("val")].sort_values(["feature", "rmse"])
    lines = [
        "# V2 Station Parameter-Sharing Ablation",
        "",
        "## Protocol",
        "",
        "- Same V2 data, global train-only feature selection and global train-only scalers.",
        "- Past 9 four-hour steps predict the next four-hour change.",
        "- Five targets are trained separately with masked L1; validation absolute RMSE selects epochs.",
        "- shared_D: current D without station identity.",
        "- shared_D_station_embedding: shared D plus an 8-dimensional station identity.",
        "- local_D: one independent D parameter set for every station and target.",
        "",
        "## Validation summary",
        "",
        "```text",
        val.to_string(index=False),
        "```",
        "",
        "## Validation station wins versus shared D",
        "",
        "```text",
        comparison[comparison["split"].eq("val")].to_string(index=False),
        "```",
        "",
        "## Validation feature metrics",
        "",
        "```text",
        val_features.to_string(index=False),
        "```",
        "",
        "## Validation-only decision",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Test readout",
        "",
        "```text",
        test.to_string(index=False),
        "```",
        "",
        "## Trainable parameters",
        "",
        "```text",
        pd.DataFrame(parameter_rows).to_string(index=False),
        "```",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_seed(
    seed: int,
    variants: tuple[str, ...] = VARIANTS,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """[08] Run all three variants for one seed and persist auditable tables."""
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or "shared_D" not in variants:
        raise ValueError(f"Invalid variants: {unknown}; shared_D is required.")
    output_dir = Path(output_dir) if output_dir is not None else output_dir_for_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    paper.SEED = int(seed)
    window.OUTPUT_STEPS = OUTPUT_STEPS
    dual.INPUT_STEPS = INPUT_STEPS
    dual.OUTPUT_STEPS = OUTPUT_STEPS

    data = all_window.load_all_station_diff1_data()
    stations = tuple(sorted(data["station"].dropna().astype(str).unique()))
    targets = tuple(window.TARGET_FEATURE_COLUMNS)
    torch = paper.require_torch()
    device = base.choose_device(torch)
    seed_everything(torch, seed)
    console.print(f"seed={seed} device={device} stations={len(stations)}", flush=True)

    arrays_by_variant = {
        variant: {"val": {}, "test": {}}
        for variant in variants
    }
    selected_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for target_idx, target in enumerate(targets):
        selected, rows = window.select_corr_features_for_target(
            data,
            stations,
            target,
            INPUT_STEPS,
            "station_parameter_sharing_ablation",
            include_current_level=False,
        )
        selected_rows.extend(rows)
        input_columns = window.diff_columns_from_features(selected)
        raw_splits, scaled_splits, scalers = window.build_target_splits(
            data,
            stations,
            input_columns,
            target,
            INPUT_STEPS,
            include_current_level=False,
        )
        scaled_splits, _ = dual.attach_scaled_current_level(raw_splits, scaled_splits)
        shared_seed = int(seed + target_idx * 1000)

        seed_everything(torch, shared_seed)
        shared_model = dual.make_dual_branch_model(
            torch,
            sequence_input_dim=scaled_splits["train"]["self_x"].shape[-1],
            current_input_dim=scaled_splits["train"]["current_level"].shape[-1],
            target_dim=1,
            output_steps=OUTPUT_STEPS,
        )
        shared_arrays, history, count = fit_one_model(
            torch,
            "shared_D",
            shared_model,
            scaled_splits,
            scalers,
            stations,
            target,
            checkpoint_root / "shared_D" / f"{target_idx}.pt",
            shared_seed,
            device,
        )
        history_rows.extend(history)
        parameter_rows.append({"variant": "shared_D", "target": target, "parameters": count})
        for split in ("val", "test"):
            arrays_by_variant["shared_D"][split][target] = shared_arrays[split]

        if "shared_D_station_embedding" in variants:
            seed_everything(torch, shared_seed)
            embedding_model = make_station_embedding_model(
                torch,
                sequence_input_dim=scaled_splits["train"]["self_x"].shape[-1],
                current_input_dim=scaled_splits["train"]["current_level"].shape[-1],
                station_count=len(stations),
                output_steps=OUTPUT_STEPS,
            )
            embedding_arrays, history, count = fit_one_model(
                torch,
                "shared_D_station_embedding",
                embedding_model,
                scaled_splits,
                scalers,
                stations,
                target,
                checkpoint_root / "shared_D_station_embedding" / f"{target_idx}.pt",
                shared_seed,
                device,
            )
            history_rows.extend(history)
            parameter_rows.append(
                {"variant": "shared_D_station_embedding", "target": target, "parameters": count}
            )
            for split in ("val", "test"):
                arrays_by_variant["shared_D_station_embedding"][split][target] = embedding_arrays[split]

        if "local_D" in variants:
            local_combined = {
                split: empty_prediction_arrays(scaled_splits[split])
                for split in ("val", "test")
            }
            local_parameter_count = 0
            for station_idx, station in enumerate(stations):
                local_splits = {
                    split: slice_station_split(values, station_idx)
                    for split, values in scaled_splits.items()
                }
                local_seed = int(seed + target_idx * 1000 + station_idx + 1)
                seed_everything(torch, local_seed)
                local_model = dual.make_dual_branch_model(
                    torch,
                    sequence_input_dim=local_splits["train"]["self_x"].shape[-1],
                    current_input_dim=local_splits["train"]["current_level"].shape[-1],
                    target_dim=1,
                    output_steps=OUTPUT_STEPS,
                )
                local_arrays, history, count = fit_one_model(
                    torch,
                    "local_D",
                    local_model,
                    local_splits,
                    scalers,
                    (station,),
                    target,
                    checkpoint_root / "local_D" / str(station_idx) / f"{target_idx}.pt",
                    local_seed,
                    device,
                )
                for row in history:
                    row["station"] = station
                history_rows.extend(history)
                local_parameter_count += count
                for split in ("val", "test"):
                    assign_local_prediction(local_combined[split], local_arrays[split], station_idx)
                console.print(f"local_D target={target} station={station} done", flush=True)
            parameter_rows.append(
                {
                    "variant": "local_D_total_25_models",
                    "target": target,
                    "parameters": local_parameter_count,
                }
            )
            for split in ("val", "test"):
                arrays_by_variant["local_D"][split][target] = local_combined[split]

    summary, station, feature, station_feature = metric_tables(arrays_by_variant, stations, targets)
    comparison = comparison_table(station)
    decision = validation_decision(summary, comparison)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    station.to_csv(output_dir / "station_metrics.csv", index=False, encoding="utf-8-sig")
    feature.to_csv(output_dir / "feature_metrics.csv", index=False, encoding="utf-8-sig")
    station_feature.to_csv(output_dir / "station_feature_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "station_comparison_vs_shared.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameter_counts.csv", index=False, encoding="utf-8-sig")
    base.save_json(output_dir / "validation_decision.json", decision)
    manifest = protocol.build_run_manifest(
        experiment="station_parameter_sharing_ablation",
        output_dir=output_dir,
        seed=seed,
        code_paths=(
            Path("scripts/gru/station_parameter_sharing.py"),
            Path("scripts/gru/run_v2_station_parameter_sharing_ablation.py"),
            Path("scripts/gru/run_wentu_dual_branch_delta_gru.py"),
            Path("scripts/gru/run_wentu_window_level_ablation.py"),
        ),
    )
    manifest["design"] = {
        "variants": list(variants),
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "station_embed_dim": STATION_EMBED_DIM,
        "feature_selection": "global_train_only_corr_top3_plus_forced_self",
        "scaling": "global_train_only_shared_across_variants",
        "loss": paper.LOSS_NAME,
        "selection_uses_test": False,
    }
    base.save_json(output_dir / "run_manifest.json", manifest)
    write_report(output_dir, summary, feature, comparison, decision, parameter_rows)
    console.print(summary.to_string(index=False), flush=True)
    console.print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return decision


def summarize_formal_multiseed(root: Path, seeds: tuple[int, ...]) -> None:
    """[09] Aggregate paired shared-versus-embedding results across formal seeds."""
    summary_frames = []
    comparison_frames = []
    feature_frames = []
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        summary = pd.read_csv(seed_dir / "summary.csv")
        summary.insert(0, "seed", seed)
        summary_frames.append(summary)
        comparison = pd.read_csv(seed_dir / "station_comparison_vs_shared.csv")
        comparison.insert(0, "seed", seed)
        comparison_frames.append(comparison)
        feature = pd.read_csv(seed_dir / "feature_metrics.csv")
        feature.insert(0, "seed", seed)
        feature_frames.append(feature)
    summary = pd.concat(summary_frames, ignore_index=True)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    feature = pd.concat(feature_frames, ignore_index=True)
    paired = summary.pivot(index=["seed", "split"], columns="variant", values=["pooled_rmse", "macro_station_rmse"])
    paired.columns = [f"{metric}_{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    for metric in ("pooled_rmse", "macro_station_rmse"):
        paired[f"{metric}_delta_embedding_vs_shared"] = (
            paired[f"{metric}_shared_D_station_embedding"] - paired[f"{metric}_shared_D"]
        )
    val = paired[paired["split"].eq("val")]
    paired_feature = feature.pivot(
        index=["seed", "split", "feature"],
        columns="variant",
        values="rmse",
    ).reset_index()
    paired_feature["rmse_delta_embedding_vs_shared"] = (
        paired_feature["shared_D_station_embedding"] - paired_feature["shared_D"]
    )
    feature_summary = (
        paired_feature.groupby(["split", "feature"])["rmse_delta_embedding_vs_shared"]
        .agg(mean_delta="mean", std_delta="std", winning_seeds=lambda values: int(values.lt(0).sum()))
        .reset_index()
    )
    formal_decision = {
        "uses_test": False,
        "seeds": list(seeds),
        "approved": bool(
            val["macro_station_rmse_delta_embedding_vs_shared"].lt(0).sum() >= 4
            and val["pooled_rmse_delta_embedding_vs_shared"].mean() < 0
            and val["macro_station_rmse_delta_embedding_vs_shared"].mean() < 0
        ),
        "gate": "embedding improves validation macro station RMSE in at least 4/5 seeds and has negative mean pooled and macro RMSE deltas",
        "validation_seed_wins_macro": int(val["macro_station_rmse_delta_embedding_vs_shared"].lt(0).sum()),
        "validation_mean_pooled_rmse_delta": float(val["pooled_rmse_delta_embedding_vs_shared"].mean()),
        "validation_mean_macro_station_rmse_delta": float(val["macro_station_rmse_delta_embedding_vs_shared"].mean()),
    }
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "all_seed_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(root / "all_seed_station_comparison.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(root / "paired_seed_metrics.csv", index=False, encoding="utf-8-sig")
    paired_feature.to_csv(root / "paired_feature_metrics.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(root / "feature_delta_summary.csv", index=False, encoding="utf-8-sig")
    base.save_json(root / "formal_decision.json", formal_decision)
    lines = [
        "# Station Embedding Formal Multiseed Report",
        "",
        "## Paired seed metrics",
        "",
        "```text",
        paired.to_string(index=False),
        "```",
        "",
        "## Feature RMSE deltas",
        "",
        "```text",
        feature_summary.to_string(index=False),
        "```",
        "",
        "## Validation-only decision",
        "",
        "```json",
        json.dumps(formal_decision, ensure_ascii=False, indent=2),
        "```",
    ]
    (root / "formal_report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(paired.to_string(index=False), flush=True)
    console.print(json.dumps(formal_decision, ensure_ascii=False, indent=2), flush=True)


def run_formal() -> None:
    """[10] Refit only the validation-approved embedding variant over formal seeds."""
    root = OUTPUT_ROOT / "formal_embedding_multiseed"
    variants = ("shared_D", "shared_D_station_embedding")
    for seed in protocol.FORMAL_SEEDS:
        run_seed(seed, variants=variants, output_dir=root / f"seed_{seed}")
    summarize_formal_multiseed(root, protocol.FORMAL_SEEDS)
