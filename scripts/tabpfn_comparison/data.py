"""Shared, causal data construction for TabPFN variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.attention import training as mainline_training
from scripts.tabpfn_comparison import config


def selected_features() -> dict[str, tuple[str, ...]]:
    frame = pd.read_csv(config.FEATURE_SELECTION_PATH)
    selected: dict[str, tuple[str, ...]] = {}
    for target in config.TARGETS:
        rows = frame[
            frame["target"].eq(target) & frame["selected"].astype(bool)
        ].copy()
        rows["_self"] = rows["feature"].eq(target)
        rows = rows.sort_values(
            ["_self", "mean_horizon_score"],
            ascending=[False, False],
        )
        selected[target] = tuple(rows["feature"].astype(str))
    return selected


def validation_splits(
    panel: pd.DataFrame,
    target: str,
    features: tuple[str, ...],
):
    """Use the exact existing mainline window builder and split boundaries."""
    return mainline_training.build_variant_splits(
        panel,
        config.STATIONS,
        target,
        features,
        "state_change",
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
    )


def station_arrays(split: dict[str, np.ndarray], station_index: int) -> dict:
    """Extract one station while retaining the mainline forecast origins."""
    return {
        "true": np.asarray(split["y_abs"][:, :, station_index, 0], dtype=float),
        "mask": np.asarray(split["y_mask"][:, :, station_index, 0], dtype=bool),
        "current": np.asarray(
            split["last_target"][:, station_index, :],
            dtype=float,
        ),
        "target_start": np.asarray(split["target_start"], dtype="datetime64[ns]"),
    }


def approved_target_series(
    panel: pd.DataFrame,
    station: str,
    target: str,
) -> pd.DataFrame:
    """Return causal target history; unapproved labels are absent, not imputed."""
    target_ok = f"{target}__target_ok"
    columns = ["time", target]
    if target_ok in panel.columns:
        columns.append(target_ok)
    series = panel.loc[panel["station"].astype(str).eq(str(station)), columns].copy()
    series["time"] = pd.to_datetime(series["time"])
    series[target] = pd.to_numeric(series[target], errors="coerce")
    valid = np.isfinite(series[target].to_numpy(float))
    if target_ok in series:
        valid &= series[target_ok].fillna(False).to_numpy(bool)
    return (
        series.loc[valid, ["time", target]]
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .rename(columns={target: "target"})
        .reset_index(drop=True)
    )


def native_origin_batch(
    series: pd.DataFrame,
    origins: np.ndarray,
    *,
    max_context_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch independent contexts; TabPFN-TS processes item_ids independently."""
    contexts = []
    futures = []
    for item_index, origin_value in enumerate(origins):
        origin = pd.Timestamp(origin_value)
        history = series.loc[series["time"].le(origin)].tail(max_context_length)
        if len(history) < 2:
            raise ValueError(f"Only {len(history)} valid values before {origin}.")
        item_id = str(item_index)
        # Restore the regular 4 h axis before handing data to the official
        # pipeline. It then drops NaN targets exactly as documented, while the
        # time-series frequency remains explicit and no value is imputed.
        grid = pd.date_range(
            history["time"].min(),
            origin,
            freq=f"{config.STEP_HOURS}h",
        )
        context = (
            history.set_index("time")
            .reindex(grid)
            .rename_axis("timestamp")
            .reset_index()
        )
        context["item_id"] = item_id
        future = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": pd.date_range(
                    origin + pd.Timedelta(hours=config.STEP_HOURS),
                    periods=config.OUTPUT_STEPS,
                    freq=f"{config.STEP_HOURS}h",
                ),
            }
        )
        contexts.append(context[["item_id", "timestamp", "target"]])
        futures.append(future)
    return pd.concat(contexts, ignore_index=True), pd.concat(futures, ignore_index=True)


def reshape_native_prediction(frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    work = frame.reset_index()
    if "target" not in work:
        raise ValueError("TabPFN-TS output does not contain target predictions.")
    work["_item_order"] = pd.to_numeric(work["item_id"], errors="raise")
    work = work.sort_values(["_item_order", "timestamp"])
    values = pd.to_numeric(work["target"], errors="coerce").to_numpy(float)
    expected = batch_size * config.OUTPUT_STEPS
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} predictions, received {len(values)}."
        )
    return values.reshape(batch_size, config.OUTPUT_STEPS)


def delta_tabular_xy(
    raw_train: dict[str, np.ndarray],
    raw_val: dict[str, np.ndarray],
    station_index: int,
    horizon_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same local state+diff+mask/current information as the matched GRU."""

    def features(split: dict[str, np.ndarray]) -> np.ndarray:
        values = np.asarray(split["self_x"][:, :, station_index, :], dtype=float)
        valid = np.asarray(split["self_mask"][:, :, station_index, :], dtype=bool)
        current = np.asarray(split["last_target"][:, station_index, :], dtype=float)
        current_valid = np.isfinite(current)
        return np.concatenate(
            (
                values.reshape(len(values), -1),
                valid.reshape(len(values), -1).astype(float),
                current,
                current_valid.astype(float),
            ),
            axis=1,
        )

    train_x = features(raw_train)
    val_x = features(raw_val)
    train_y = np.asarray(
        raw_train["y"][:, horizon_index, station_index, 0],
        dtype=float,
    )
    train_ok = np.asarray(
        raw_train["y_mask"][:, horizon_index, station_index, 0],
        dtype=bool,
    ) & np.isfinite(train_y)
    return train_x[train_ok], train_y[train_ok], val_x, train_ok
