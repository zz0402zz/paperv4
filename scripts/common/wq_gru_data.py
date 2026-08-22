"""Canonical water-quality columns and the V2 processed-data loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = (
    "水温(℃)",
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "浊度(NTU)",
    "电导率(μS/cm)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
    "总氮(mg/L)",
)


def target_ok_column(feature: str) -> str:
    """Return the quality-sidecar column approving a target observation."""
    return f"{feature}__target_ok"


def processed_data_provenance(path: str | Path) -> dict[str, str]:
    """Return a stable identity for a processed data file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"processed_data_path": str(path), "processed_data_sha256": digest.hexdigest()}


def load_processed_4h_data(path: str | Path) -> pd.DataFrame:
    """Load V2 observations and attach the target-quality sidecar when present."""
    path = Path(path)
    data = pd.read_csv(path)
    data["time"] = pd.to_datetime(data["time"])
    for feature in FEATURE_COLUMNS:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    quality_path = path.with_name("quantity_4h_quality.csv")
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        quality["time"] = pd.to_datetime(quality["time"])
        quality_columns = [
            column for column in quality.columns if column not in {"station", "time"}
        ]
        data = data.merge(
            quality[["station", "time", *quality_columns]],
            on=["station", "time"],
            how="left",
            validate="one_to_one",
        )
        for feature in FEATURE_COLUMNS:
            column = target_ok_column(feature)
            if column in data:
                data[column] = data[column].fillna(False).astype(bool)

    return data.sort_values(["station", "time"]).reset_index(drop=True)
