#!/usr/bin/env python3
"""Aggregate five-model 2024 results without rerunning the frozen GRU models."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import forecasting, v2_experiment_protocol as protocol
from scripts.tabpfn_comparison import (
    config,
    data as comparison_data,
    io,
    models as model_helpers,
    run as runner,
)


def _flat_metrics(pred, true, mask) -> dict[str, object]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return {"valid_points": 0, "mae": None, "rmse": None, "nse": None}
    error = np.asarray(pred, dtype=float)[valid] - np.asarray(true, dtype=float)[valid]
    truth = np.asarray(true, dtype=float)[valid]
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "valid_points": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "nse": None
        if denominator <= np.finfo(float).eps
        else float(1.0 - np.sum(np.square(error)) / denominator),
    }


def tabpfn_cells(model: str, seed: int) -> pd.DataFrame:
    rows = []
    for target in config.TARGETS:
        for station in config.STATIONS:
            path = io.prediction_path(model, seed, target, station)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run the requested TabPFN model first."
                )
            arrays, _ = io.load_prediction(path)
            for horizon_index in range(config.OUTPUT_STEPS):
                rows.append(
                    {
                        "variant": model,
                        "station": station,
                        "target": target,
                        "horizon_step": horizon_index + 1,
                        "horizon_hours": (horizon_index + 1) * config.STEP_HOURS,
                        **_flat_metrics(
                            arrays["pred"][:, horizon_index],
                            arrays["true"][:, horizon_index],
                            arrays["mask"][:, horizon_index],
                        ),
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def mainline_cells() -> pd.DataFrame:
    """Recompute both GRU and persistence metrics from audited origin files."""
    frames = []
    for seed in config.SEEDS:
        frames.extend(
            tabpfn_cells(model, seed)
            for model in (config.DELTA_GRU_KEY, config.MATCHED_GRU_KEY)
        )
        rows = []
        for target in config.TARGETS:
            for station in config.STATIONS:
                arrays, _ = io.load_prediction(
                    io.prediction_path(
                        config.DELTA_GRU_KEY,
                        seed,
                        target,
                        station,
                    )
                )
                persistence = np.repeat(
                    arrays["current"], config.OUTPUT_STEPS, axis=1
                )
                for horizon_index in range(config.OUTPUT_STEPS):
                    rows.append(
                        {
                            "variant": config.PERSISTENCE_KEY,
                            "station": station,
                            "target": target,
                            "horizon_step": horizon_index + 1,
                            "horizon_hours": (horizon_index + 1)
                            * config.STEP_HOURS,
                            **_flat_metrics(
                                persistence[:, horizon_index],
                                arrays["true"][:, horizon_index],
                                arrays["mask"][:, horizon_index],
                            ),
                            "seed": seed,
                        }
                    )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _expected_metadata(
    model: str,
    seed: int,
    target: str,
    station: str,
    features_by_target: dict[str, tuple[str, ...]],
) -> dict:
    if model in {config.DELTA_GRU_KEY, config.MATCHED_GRU_KEY}:
        # Frozen checkpoints are loaded in the main paper environment. Preserve
        # that audited torch version rather than substituting TabPFN-env torch.
        manifest_path = config.MAINLINE_VALIDATION_DIR / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_payload = runner.task_metadata(
            model=model,
            seed=seed,
            target=target,
            station=station,
            features=features_by_target[target],
        )
        metadata_payload["torch_version"] = manifest["runtime_versions"]["torch"]
        return metadata_payload
    return runner.task_metadata(
        model=model,
        seed=seed,
        target=target,
        station=station,
        features=features_by_target[target],
    )


def completed_models(
    features_by_target: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    models = []
    for model in config.TABPFN_KEYS:
        if all(
            io.is_complete(
                io.prediction_path(model, seed, target, station),
                _expected_metadata(
                    model,
                    seed,
                    target,
                    station,
                    features_by_target,
                ),
            )
            for seed in config.model_seeds(model)
            for target in config.TARGETS
            for station in config.STATIONS
        ):
            models.append(model)
    return tuple(models)


def require_frozen_gru_exports(
    features_by_target: dict[str, tuple[str, ...]],
) -> None:
    missing = [
        io.prediction_path(model, seed, target, station)
        for model in (config.DELTA_GRU_KEY, config.MATCHED_GRU_KEY)
        for seed in config.SEEDS
        for target in config.TARGETS
        for station in config.STATIONS
        if (
            not io.prediction_path(model, seed, target, station).exists()
            or not io.is_complete(
                io.prediction_path(model, seed, target, station),
                _expected_metadata(
                    model,
                    seed,
                    target,
                    station,
                    features_by_target,
                ),
            )
        )
    ]
    if missing:
        raise SystemExit(
            "Frozen GRU prediction exports are incomplete. Run "
            "`python -m scripts.tabpfn_comparison.run --model frozen_gru` "
            "from the TabPFN environment first. "
            f"First missing file: {missing[0]}"
        )


def attach_persistence(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "station", "target", "horizon_hours"]
    persistence = (
        cells[cells["variant"].eq(config.PERSISTENCE_KEY)][keys + ["rmse"]]
        .rename(columns={"rmse": "persistence_rmse"})
        .drop_duplicates(keys)
    )
    output = cells.merge(persistence, on=keys, how="left", validate="many_to_one")
    output["rmse_ratio_to_persistence"] = output["rmse"] / output["persistence_rmse"]
    return output


def summarize(cells: pd.DataFrame, models: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for name, group in (
        ("overall", None),
        ("by_horizon", "horizon_hours"),
        ("by_target", "target"),
        ("by_station", "station"),
    ):
        groups = ["variant"] if group is None else [group, "variant"]
        table = (
            cells[cells["variant"].isin(models)]
            .groupby(groups, as_index=False)["rmse_ratio_to_persistence"]
            .mean()
        )
        if group is None:
            table = table.set_index("variant")[["rmse_ratio_to_persistence"]].T
            table.index = [0]
        else:
            table = table.pivot(
                index=group,
                columns="variant",
                values="rmse_ratio_to_persistence",
            ).reset_index()
        for model in models:
            if model == config.DELTA_GRU_KEY:
                continue
            table[f"{model}_vs_delta_gru_pct"] = (
                table[model] / table[config.DELTA_GRU_KEY] - 1.0
            ) * 100.0
        output[name] = table
    by_seed = (
        cells[cells["variant"].isin(models)]
        .groupby(["seed", "variant"], as_index=False)["rmse_ratio_to_persistence"]
        .mean()
        .pivot(index="seed", columns="variant", values="rmse_ratio_to_persistence")
        .reset_index()
    )
    output["by_seed"] = by_seed
    return output


def weekly_bootstrap(
    model: str,
    comparator: str,
    *,
    repeats: int = 2000,
) -> dict[str, object]:
    weekly_values: dict[str, list[float]] = {}
    # A native zero-shot prediction is deterministic here.  Pair that one
    # prediction with every frozen GRU seed and average within each week; this
    # propagates comparator-seed variability without pretending the repeated
    # native prediction is five independent model runs.
    shared_seeds = config.SEEDS
    for seed in shared_seeds:
        model_seed = config.model_seed(model, seed)
        for target in config.TARGETS:
            for station in config.STATIONS:
                arrays, _ = io.load_prediction(
                    io.prediction_path(model, model_seed, target, station)
                )
                control, _ = io.load_prediction(
                    io.prediction_path(comparator, seed, target, station)
                )
                for key in ("target_start", "true", "mask", "current"):
                    if not np.array_equal(
                        arrays[key], control[key], equal_nan=True
                    ):
                        raise RuntimeError(
                            f"Models do not share identical {key}: "
                            f"{model}, seed={seed}, {target}, {station}"
                        )
                weeks = pd.to_datetime(arrays["target_start"]).to_period("W").astype(str)
                persistence = np.repeat(arrays["current"], config.OUTPUT_STEPS, axis=1)
                for horizon_index in range(config.OUTPUT_STEPS):
                    mask = (
                        arrays["mask"][:, horizon_index].astype(bool)
                        & control["mask"][:, horizon_index].astype(bool)
                    )
                    truth = arrays["true"][:, horizon_index]
                    persistence_error = persistence[:, horizon_index] - truth
                    scale = (
                        float(np.sqrt(np.mean(np.square(persistence_error[mask]))))
                        if mask.any()
                        else np.nan
                    )
                    if not np.isfinite(scale) or scale <= 0:
                        continue
                    difference = (
                        np.square(arrays["pred"][:, horizon_index] - truth)
                        - np.square(control["pred"][:, horizon_index] - truth)
                    ) / np.square(scale)
                    for week, value, valid in zip(weeks, difference, mask):
                        if valid and np.isfinite(value):
                            weekly_values.setdefault(str(week), []).append(float(value))
    values = np.asarray(
        [np.mean(items) for items in weekly_values.values()], dtype=float
    )
    if values.size < 2:
        raise RuntimeError(
            f"Insufficient natural-week blocks for bootstrap: {model} has "
            f"{values.size}."
        )
    rng = np.random.default_rng(20260813)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    )
    return {
        "comparison": f"{model} minus {comparator} normalized squared error",
        "weekly_blocks": int(len(values)),
        "point_estimate": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def save_report(
    tables: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    bootstrap: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# TabPFN 与变化量 GRU：2024 严格比较",
        "",
        "## 协议",
        "",
        "- 固定7站、5指标、4小时时间粒度及4–72小时全部18个时距。",
        "- Delta-GRU结果直接复用冻结主线，不重新训练。",
        "- TabPFN-TS-v2对照和TabPFN-TS-3为逐站逐指标零样本滚动预测。",
        "- v2对照由tabpfn==8.1.0的ModelVersion.V2解析；未把它冒充为已核验相同的论文2noar4o2权重。",
        "- Delta-TabPFN-v2使用与匹配GRU相同的24小时原值、diff1、mask及当前值。",
        "- 原生零样本模型固定seed=0；重复其相同预测不能冒充多种子实验。",
        "- 主指标为站点×指标×时距等权RMSE/持久性RMSE，越低越好。",
        "",
        "## 已完成模型",
        "",
        *[f"- {model}" for model in models],
    ]
    for title, key in (
        ("总体", "overall"),
        ("分时距", "by_horizon"),
        ("分指标", "by_target"),
        ("分站点", "by_station"),
        ("分种子", "by_seed"),
    ):
        lines.extend(
            ["", f"## {title}", "", "```text", tables[key].to_string(index=False), "```"]
        )
    lines.extend(
        ["", "## 按周聚类Bootstrap（相对原变化量Delta-GRU）", "", "```json", json.dumps(bootstrap, ensure_ascii=False, indent=2), "```", ""]
    )
    (config.OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_manifest(models: tuple[str, ...]) -> None:
    manifest = protocol.build_run_manifest(
        experiment="tabpfn_2024_strict_comparison",
        output_dir=config.OUTPUT_DIR,
        seed=config.SEEDS[0],
        code_paths=tuple(
            Path(f"scripts/tabpfn_comparison/{name}")
            for name in (
                "config.py",
                "data.py",
                "io.py",
                "models.py",
                "run.py",
                "report.py",
            )
        ),
    )
    package_versions = {}
    for package in ("tabpfn", "tabpfn-time-series"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest.update(
        {
            "models": list(models),
            "stations": list(config.STATIONS),
            "seeds": list(config.SEEDS),
            "native_zero_shot_seed": 0,
            "package_versions": package_versions,
            "model_identities": {
                model: model_helpers.model_identity(model)
                for model in config.TABPFN_KEYS
            },
            "frozen_gru_result_dir": str(config.MAINLINE_VALIDATION_DIR),
        }
    )
    forecasting.save_json(config.OUTPUT_DIR / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    features_by_target = comparison_data.selected_features()
    done = completed_models(features_by_target)
    if args.allow_partial and not done:
        raise SystemExit(
            "No complete TabPFN model is available; partial report was not generated."
        )
    if not args.allow_partial and set(done) != set(config.TABPFN_KEYS):
        missing = sorted(set(config.TABPFN_KEYS).difference(done))
        raise SystemExit(f"TabPFN predictions incomplete: {missing}")
    require_frozen_gru_exports(features_by_target)
    frames = [mainline_cells()]
    for model in done:
        frames.extend(tabpfn_cells(model, seed) for seed in config.model_seeds(model))
    cells = pd.concat(frames, ignore_index=True)
    # Native zero-shot predictions are copied across the five reporting seeds only
    # for equal-weight comparison with stochastic GRUs; uncertainty is not inferred
    # from these duplicates, and the report explicitly labels seed=0.
    native = cells[cells["variant"].isin(config.NATIVE_SPECS)].copy()
    copies = []
    for seed in config.SEEDS:
        item = native.copy()
        item["seed"] = seed
        copies.append(item)
    cells = pd.concat(
        [cells[~cells["variant"].isin(config.NATIVE_SPECS)], *copies],
        ignore_index=True,
    )
    cells = attach_persistence(cells)
    models = (
        config.DELTA_GRU_KEY,
        config.MATCHED_GRU_KEY,
        *done,
    )
    tables = summarize(cells, models)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(
        config.OUTPUT_DIR / "station_target_horizon_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for key, table in tables.items():
        table.to_csv(
            config.OUTPUT_DIR / f"comparison_{key}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    bootstrap = {
        model: weekly_bootstrap(model, config.DELTA_GRU_KEY)
        for model in done
    }
    forecasting.save_json(config.OUTPUT_DIR / "bootstrap_summary.json", bootstrap)
    save_report(tables, models, bootstrap)
    save_manifest(models)


if __name__ == "__main__":
    main()
