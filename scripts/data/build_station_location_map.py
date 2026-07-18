#!/usr/bin/env python3
"""Build the auditable water-quality and hydrometeorological station map."""

from __future__ import annotations

from scripts.common.terminal_output import console

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NATIONAL_METADATA_PATH = ROOT / "data/metadata/station_metadata.csv"
CURATED_WQ_PATH = ROOT / "data/metadata/water_quality_station_locations_curated.csv"
HYDROMET_COORDINATE_PATH = ROOT / "data/metadata/hydromet_station_locations_curated.csv"
PROVINCIAL_DATA_DIR = ROOT / "data/省控小时数据"
HYDRO_DATA_DIRS = (
    ROOT / "data/流量数据-水利厅/按站点拆分",
    ROOT / "data/降水和水位和补充流量/按站点拆分",
)
RAIN_SUMMARY_PATH = ROOT / "data/降水和水位和补充流量/按站点拆分/降水极值_转换汇总.csv"
OUTPUT_DIR = ROOT / "outputs/maps"

MAP_COLUMNS = [
    "record_id",
    "station",
    "station_codes",
    "layer_keys",
    "network",
    "station_type",
    "river",
    "city",
    "county",
    "longitude",
    "latitude",
    "coordinate_confidence",
    "coordinate_source",
    "coordinate_source_url",
    "coordinate_source_file",
    "data_types",
    "data_start",
    "data_end",
    "data_status",
    "source_file",
    "note",
]


def clean_text(value: object) -> str:
    """Convert a spreadsheet value into stable display text."""
    if pd.isna(value):
        return ""
    return str(value).strip().strip(",，")


def display_path(path: Path) -> str:
    """Display project files relatively and external files absolutely."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def station_name_from_path(path: Path) -> str:
    """Extract the station name from a provincial hourly filename."""
    return path.name.split("2015年")[0]


def load_provincial_station_metadata(data_dir: Path = PROVINCIAL_DATA_DIR) -> pd.DataFrame:
    """Read one station metadata row from every provincial hourly workbook."""
    rows = []
    for path in sorted(data_dir.glob("*.xls")):
        data = pd.read_excel(path, nrows=8)
        if "站点名称" not in data.columns:
            continue
        valid = data[pd.notna(data["站点名称"])].head(1)
        if valid.empty:
            continue
        row = valid.iloc[0]
        rows.append(
            {
                "station": clean_text(row.get("站点名称")) or station_name_from_path(path),
                "section": clean_text(row.get("断面名称")),
                "network": clean_text(row.get("管理级别")) or "省控/市控",
                "city": clean_text(row.get("所在设区市")),
                "county": clean_text(row.get("所在县(市、区)")),
                "basin": clean_text(row.get("流域")),
                "system": clean_text(row.get("水系")),
                "river": clean_text(row.get("水体")),
                "first_time_in_file": clean_text(row.get("监测时间")),
                "source_file": display_path(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["city", "county", "river", "station"]).reset_index(drop=True)


def load_national_locations(path: Path = NATIONAL_METADATA_PATH) -> pd.DataFrame:
    """Load the 25 national-control stations with official section coordinates."""
    source = pd.read_csv(path)
    rows = []
    for index, row in source.iterrows():
        rows.append(
            {
                "record_id": f"national-wq-{index}",
                "station": clean_text(row["station"]),
                "station_codes": "",
                "layer_keys": "national_wq",
                "network": "国控",
                "station_type": "国控水质断面",
                "river": clean_text(row.get("river")),
                "city": "",
                "county": "",
                "longitude": row.get("longitude"),
                "latitude": row.get("latitude"),
                "coordinate_confidence": "official_section_coordinate",
                "coordinate_source": clean_text(row.get("coordinate_source")),
                "coordinate_source_url": clean_text(row.get("coordinate_source_url")),
                "coordinate_source_file": display_path(path),
                "data_types": "4小时水质",
                "data_start": "",
                "data_end": "",
                "data_status": "本地国控水质数据",
                "source_file": "data/processed/v2/quantity_4h_observed.csv",
                "note": clean_text(row.get("coordinate_note")),
            }
        )
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def load_provincial_locations(
    data_dir: Path = PROVINCIAL_DATA_DIR,
    curated_path: Path = CURATED_WQ_PATH,
) -> pd.DataFrame:
    """Attach the user-reviewed station coordinates to provincial hourly stations."""
    metadata = load_provincial_station_metadata(data_dir)
    curated = pd.read_csv(curated_path)
    merged = metadata.merge(curated, on="station", how="left", suffixes=("", "_curated"))
    rows = []
    for index, row in merged.iterrows():
        river = clean_text(row.get("river_curated")) or clean_text(row.get("river"))
        business_type = clean_text(row.get("business_type"))
        source_file = clean_text(row.get("coordinate_source_file"))
        if business_type == "饮用水水源" and "饮用水" not in source_file:
            source_file = "literature/sources/maps/省控饮用水站点位置图.jpg"
        has_coordinate = pd.notna(row.get("longitude")) and pd.notna(row.get("latitude"))
        rows.append(
            {
                "record_id": f"provincial-wq-{index}",
                "station": clean_text(row["station"]),
                "station_codes": "",
                "layer_keys": "provincial_wq",
                "network": clean_text(row.get("network")) or "省控/市控",
                "station_type": business_type or "省控/市控水质断面",
                "river": river,
                "city": clean_text(row.get("city")),
                "county": clean_text(row.get("county")),
                "longitude": row.get("longitude"),
                "latitude": row.get("latitude"),
                "coordinate_confidence": clean_text(row.get("coordinate_confidence")) if has_coordinate else "missing",
                "coordinate_source": clean_text(row.get("coordinate_source")) if has_coordinate else "待补官方站点坐标",
                "coordinate_source_url": "",
                "coordinate_source_file": source_file,
                "data_types": "小时水质",
                "data_start": clean_text(row.get("first_time_in_file")),
                "data_end": "",
                "data_status": "本地省控/市控小时数据",
                "source_file": clean_text(row.get("source_file")),
                "note": "" if has_coordinate else "当前不绘制，避免用区县中心冒充站点位置。",
            }
        )
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def parse_hydro_filename(path: Path) -> tuple[str, str, str] | None:
    """Parse station code, name and variable from a split daily hydrology file."""
    match = re.match(r"(?P<code>\d+)_(?P<station>.+?)_日平均(?P<kind>流量|水位)表\.csv$", path.name)
    if not match:
        return None
    return match.group("code"), match.group("station"), f"日平均{match.group('kind')}"


def data_date_range(path: Path) -> tuple[str, str]:
    """Read the minimum and maximum date from one split station CSV."""
    data = pd.read_csv(path)
    date_column = next((column for column in data.columns if "日期" in str(column) or "时间" in str(column)), None)
    if date_column is None:
        return "", ""
    dates = pd.to_datetime(data[date_column], errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def collect_daily_hydromet_rows(data_dirs: tuple[Path, ...] = HYDRO_DATA_DIRS) -> pd.DataFrame:
    """Inventory the local split daily discharge and water-level files."""
    rows = []
    seen_paths: set[Path] = set()
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("*_日平均*表.csv")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            parsed = parse_hydro_filename(path)
            if parsed is None:
                continue
            code, station, data_type = parsed
            start, end = data_date_range(path)
            rows.append(
                {
                    "station": station,
                    "station_code": code,
                    "data_type": data_type,
                    "data_start": start,
                    "data_end": end,
                    "source_file": display_path(path),
                }
            )
    return pd.DataFrame(rows)


def collect_rain_rows(summary_path: Path = RAIN_SUMMARY_PATH) -> pd.DataFrame:
    """Inventory local station-split rainfall-extreme data."""
    summary = pd.read_csv(summary_path)
    rows = []
    for (station, code), group in summary.groupby(["站名", "站码"], dropna=False):
        source_files = sorted({clean_text(value) for value in group["长表输出文件"] if clean_text(value)})
        rows.append(
            {
                "station": clean_text(station),
                "station_code": str(int(code)) if pd.notna(code) else "",
                "data_type": "降水极值（分钟/小时历时）",
                "data_start": f"{int(group['起始年'].min())}-01-01",
                "data_end": f"{int(group['结束年'].max())}-12-31",
                "source_file": " | ".join(source_files),
            }
        )
    return pd.DataFrame(rows)


def aggregate_hydromet_inventory(rows: pd.DataFrame, category: str) -> pd.DataFrame:
    """Collapse duplicate station files into one marker record per station/category."""
    if rows.empty:
        return rows
    records = []
    for station, group in rows.groupby("station", sort=True):
        types = sorted(set(group["data_type"].astype(str)))
        if category == "hydrology":
            layer_keys = "|".join(
                key for key, token in (("flow", "流量"), ("level", "水位")) if any(token in value for value in types)
            )
        else:
            layer_keys = "rainfall"
        records.append(
            {
                "station": station,
                "station_codes": " | ".join(sorted(set(group["station_code"].astype(str)))),
                "layer_keys": layer_keys,
                "data_types": " | ".join(types),
                "data_start": min(value for value in group["data_start"].astype(str) if value),
                "data_end": max(value for value in group["data_end"].astype(str) if value),
                "source_file": " | ".join(sorted(set(group["source_file"].astype(str)))),
            }
        )
    return pd.DataFrame(records)


def combine_hydromet_categories(inventory: pd.DataFrame) -> pd.DataFrame:
    """Merge same-name flow/level/rain records so overlapping markers remain clickable."""
    records = []
    layer_order = ("flow", "level", "rainfall")
    for station, group in inventory.groupby("station", sort=True):
        layers = {item for value in group["layer_keys"].astype(str) for item in value.split("|") if item}
        records.append(
            {
                "station": station,
                "station_codes": " | ".join(
                    sorted({item.strip() for value in group["station_codes"].astype(str) for item in value.split("|") if item.strip()})
                ),
                "layer_keys": "|".join(key for key in layer_order if key in layers),
                "data_types": " | ".join(
                    sorted({item.strip() for value in group["data_types"].astype(str) for item in value.split("|") if item.strip()})
                ),
                "data_start": min(value for value in group["data_start"].astype(str) if value),
                "data_end": max(value for value in group["data_end"].astype(str) if value),
                "source_file": " | ".join(
                    sorted({item.strip() for value in group["source_file"].astype(str) for item in value.split("|") if item.strip()})
                ),
            }
        )
    return pd.DataFrame(records)


def select_coordinate_row(group: pd.Series, coordinates: pd.DataFrame) -> pd.Series | None:
    """Prefer a station-code coordinate match, then a same-name coordinate."""
    codes = {value.strip() for value in str(group["station_codes"]).split("|")}
    exact = coordinates[coordinates["station_code"].astype(str).isin(codes)]
    if not exact.empty:
        return exact.iloc[0]
    named = coordinates[coordinates["station"].astype(str) == str(group["station"])]
    return None if named.empty else named.iloc[0]


def load_hydromet_locations(coordinate_path: Path = HYDROMET_COORDINATE_PATH) -> pd.DataFrame:
    """Build map records for local discharge, water-level and rainfall datasets."""
    coordinates = pd.read_csv(coordinate_path, dtype={"station_code": str})
    daily = aggregate_hydromet_inventory(collect_daily_hydromet_rows(), "hydrology")
    rainfall = aggregate_hydromet_inventory(collect_rain_rows(), "rainfall")
    inventory = combine_hydromet_categories(pd.concat([daily, rainfall], ignore_index=True))
    rows = []
    for index, record in inventory.iterrows():
        coordinate = select_coordinate_row(record, coordinates)
        values = coordinate.to_dict() if coordinate is not None else {}
        layers = set(str(record["layer_keys"]).split("|"))
        station_type = "流量/水位站" if {"flow", "level"}.issubset(layers) else (
            "流量站" if "flow" in layers else "水位站" if "level" in layers else "降水站"
        )
        rows.append(
            {
                "record_id": f"hydromet-{index}",
                "station": clean_text(record["station"]),
                "station_codes": clean_text(record["station_codes"]),
                "layer_keys": clean_text(record["layer_keys"]),
                "network": "水文气象",
                "station_type": station_type,
                "river": clean_text(values.get("river_or_scope")),
                "city": "",
                "county": "",
                "longitude": values.get("longitude"),
                "latitude": values.get("latitude"),
                "coordinate_confidence": clean_text(values.get("coordinate_confidence")) or "missing",
                "coordinate_source": clean_text(values.get("coordinate_source")) or "待补官方站点坐标",
                "coordinate_source_url": clean_text(values.get("coordinate_source_url")),
                "coordinate_source_file": display_path(coordinate_path),
                "data_types": clean_text(record["data_types"]),
                "data_start": clean_text(record["data_start"]),
                "data_end": clean_text(record["data_end"]),
                "data_status": "本地数据可用",
                "source_file": clean_text(record["source_file"]),
                "note": clean_text(values.get("note")),
            }
        )
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def is_reliable_coordinate(confidence: str) -> bool:
    """Return whether a coordinate is station/address level rather than regional."""
    return str(confidence) in {
        "official_section_coordinate",
        "official_station_coordinate",
        "official_open_data_coordinate",
        "prior_curated_station_coordinate",
        "named_feature_coordinate",
    }


def distance_quality(left: str, right: str) -> str:
    """Describe whether a computed distance is suitable for quantitative use."""
    return "usable_station_level" if is_reliable_coordinate(left) and is_reliable_coordinate(right) else "rough_contains_approximate"


def build_distance_tables(locations: pd.DataFrame, output_dir: Path) -> None:
    """Write water-quality nearest-neighbor and same-river distance tables."""
    valid = locations.dropna(subset=["longitude", "latitude"]).copy()
    rows = []
    for _, source in valid.iterrows():
        candidates = valid[valid["record_id"] != source["record_id"]]
        distances = []
        for _, candidate in candidates.iterrows():
            distances.append(
                (
                    haversine_km(source["latitude"], source["longitude"], candidate["latitude"], candidate["longitude"]),
                    candidate,
                )
            )
        nearest = min(distances, key=lambda item: item[0]) if distances else (None, None)
        nearest_same_river = min(
            (item for item in distances if str(item[1]["river"]) == str(source["river"]) and str(source["river"])),
            key=lambda item: item[0],
            default=(None, None),
        )
        rows.append(
            {
                "station": source["station"],
                "network": source["network"],
                "river": source["river"],
                "longitude": source["longitude"],
                "latitude": source["latitude"],
                "coordinate_confidence": source["coordinate_confidence"],
                "nearest_station": "" if nearest[1] is None else nearest[1]["station"],
                "nearest_distance_km": nearest[0],
                "nearest_distance_quality": "" if nearest[1] is None else distance_quality(source["coordinate_confidence"], nearest[1]["coordinate_confidence"]),
                "nearest_same_river_station": "" if nearest_same_river[1] is None else nearest_same_river[1]["station"],
                "nearest_same_river_distance_km": nearest_same_river[0],
                "nearest_same_river_distance_quality": "" if nearest_same_river[1] is None else distance_quality(source["coordinate_confidence"], nearest_same_river[1]["coordinate_confidence"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "station_nearest_distances.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    records = valid.to_dict("records")
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if not left["river"] or str(left["river"]) != str(right["river"]):
                continue
            pair_rows.append(
                {
                    "left_station": left["station"],
                    "left_network": left["network"],
                    "right_station": right["station"],
                    "right_network": right["network"],
                    "river": left["river"],
                    "distance_km": haversine_km(left["latitude"], left["longitude"], right["latitude"], right["longitude"]),
                    "distance_quality": distance_quality(left["coordinate_confidence"], right["coordinate_confidence"]),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["river", "distance_km"])
    pairs.to_csv(output_dir / "station_same_river_distance_pairs.csv", index=False, encoding="utf-8-sig")


def json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to browser-safe records without NaN values."""
    clean = frame.astype(object).where(pd.notna(frame), "")
    return clean.to_dict("records")


def escape_json_for_html(value: object) -> str:
    """Serialize JSON safely inside an HTML script element."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_map_html(locations: pd.DataFrame, output_path: Path) -> None:
    """Write the interactive Leaflet map."""
    valid = locations.dropna(subset=["longitude", "latitude"]).copy()
    missing = locations[locations["longitude"].isna() | locations["latitude"].isna()].copy()
    layer_counts = {
        key: int(valid["layer_keys"].astype(str).str.split("|").apply(lambda values: key in values).sum())
        for key in ("national_wq", "provincial_wq", "flow", "level", "rainfall")
    }
    approximate_count = int((~valid["coordinate_confidence"].map(is_reliable_coordinate)).sum())
    center_lat = float(valid["latitude"].mean()) if len(valid) else 29.5
    center_lon = float(valid["longitude"].mean()) if len(valid) else 120.0
    records_json = escape_json_for_html(json_records(valid))
    missing_json = escape_json_for_html(json_records(missing[["station", "network", "station_type", "note"]]))
    counts_json = escape_json_for_html(layer_counts)
    html_text = MAP_HTML_TEMPLATE.replace("__STATIONS__", records_json)
    html_text = html_text.replace("__MISSING__", missing_json).replace("__COUNTS__", counts_json)
    html_text = html_text.replace("__APPROXIMATE_COUNT__", str(approximate_count))
    html_text = html_text.replace("__CENTER_LAT__", f"{center_lat:.6f}").replace("__CENTER_LON__", f"{center_lon:.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


MAP_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>钱塘江水质与水文气象站点图</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
  <style>
    :root { --ink:#172033; --muted:#64748b; --line:#d8dee8; --surface:rgba(255,255,255,.97); --accent:#0f766e; }
    * { box-sizing:border-box; }
    html, body, #map { height:100%; width:100%; margin:0; }
    body { color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }
    button, input { font:inherit; letter-spacing:0; }
    #map { background:#e8edf3; }
    .panel { position:absolute; z-index:700; top:14px; left:14px; width:340px; max-height:calc(100% - 28px); overflow:auto; background:var(--surface); border:1px solid var(--line); border-radius:8px; box-shadow:0 12px 30px rgba(15,23,42,.16); }
    .panel-head { position:sticky; top:0; z-index:2; display:flex; align-items:center; justify-content:space-between; min-height:48px; padding:10px 12px; background:var(--surface); border-bottom:1px solid var(--line); }
    .title { font-size:16px; font-weight:700; }
    .icon-button { width:32px; height:32px; display:grid; place-items:center; border:1px solid var(--line); border-radius:6px; color:#334155; background:#fff; cursor:pointer; }
    .icon-button:hover, .tool-button:hover { border-color:#94a3b8; background:#f8fafc; }
    .panel-body { padding:12px; }
    .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:12px; }
    .stat { padding:7px 6px; border:1px solid var(--line); border-radius:6px; background:#f8fafc; text-align:center; }
    .stat b { display:block; font-size:15px; }
    .stat span { color:var(--muted); font-size:11px; }
    .section { margin-top:12px; }
    .section-title { margin-bottom:7px; color:#475569; font-size:12px; font-weight:700; }
    .search-wrap { position:relative; }
    .search { width:100%; height:36px; padding:0 34px 0 10px; border:1px solid var(--line); border-radius:6px; outline:none; }
    .search:focus { border-color:var(--accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
    .search-icon { position:absolute; right:9px; top:9px; width:18px; height:18px; color:#64748b; pointer-events:none; }
    .search-results { display:none; position:absolute; z-index:8; top:40px; left:0; right:0; max-height:220px; overflow:auto; background:#fff; border:1px solid var(--line); border-radius:6px; box-shadow:0 8px 20px rgba(15,23,42,.15); }
    .search-result { width:100%; padding:8px 10px; border:0; border-bottom:1px solid #eef2f7; background:#fff; text-align:left; cursor:pointer; }
    .search-result:hover { background:#f1f5f9; }
    .search-result small { display:block; color:var(--muted); margin-top:2px; }
    .filters { display:grid; grid-template-columns:1fr 1fr; gap:7px 10px; }
    .filters label { min-height:28px; display:flex; align-items:center; gap:7px; font-size:13px; }
    .filters input { width:15px; height:15px; accent-color:var(--accent); }
    .legend-mark { width:11px; height:11px; flex:0 0 11px; border:2px solid currentColor; border-radius:50%; }
    .national { color:#2563eb; } .provincial { color:#ea580c; } .flow { color:#15803d; } .level { color:#7c3aed; } .rain { color:#0891b2; }
    .tool-row { display:flex; gap:7px; }
    .tool-button { min-width:0; flex:1; height:34px; display:flex; align-items:center; justify-content:center; gap:6px; border:1px solid var(--line); border-radius:6px; background:#fff; color:#334155; cursor:pointer; }
    .tool-button svg { width:16px; height:16px; }
    .tool-button.active { color:#fff; border-color:var(--accent); background:var(--accent); }
    .measure-status { min-height:19px; margin-top:7px; color:#475569; font-size:12px; }
    details { font-size:12px; color:#475569; }
    details summary { cursor:pointer; }
    .missing-list { margin:7px 0 0; padding-left:18px; line-height:1.65; }
    .confidence { margin-top:9px; padding-top:9px; border-top:1px solid var(--line); color:var(--muted); font-size:11px; line-height:1.45; }
    .panel.collapsed { width:48px; height:48px; overflow:hidden; }
    .panel.collapsed .panel-head { border-bottom:0; padding:7px; }
    .panel.collapsed .title, .panel.collapsed .panel-body { display:none; }
    .leaflet-popup-content { min-width:250px; max-width:320px; margin:12px 14px; line-height:1.42; }
    .popup-title { margin-bottom:7px; font-size:15px; font-weight:700; }
    .popup-badge { display:inline-block; margin-left:5px; padding:1px 5px; border-radius:4px; color:#475569; background:#eef2f7; font-size:11px; font-weight:600; }
    .popup-grid { display:grid; grid-template-columns:68px 1fr; gap:4px 8px; font-size:12px; }
    .popup-grid dt { color:#64748b; } .popup-grid dd { margin:0; overflow-wrap:anywhere; }
    .source-link { color:#0369a1; text-decoration:none; }
    .low-note { color:#9a3412; font-weight:600; }
    .station-label { padding:2px 5px; border:1px solid #cbd5e1; border-radius:4px; box-shadow:none; color:#253047; background:rgba(255,255,255,.9); font-size:11px; }
    .station-label::before { display:none; }
    .measure-label { padding:3px 6px; border:0; border-radius:4px; color:#fff; background:#0f172a; font-size:12px; }
    .measure-label::before { display:none; }
    @media (max-width:640px) { .panel { top:8px; left:8px; width:calc(100% - 16px); max-height:52%; } .stats { grid-template-columns:repeat(5,1fr); } .stat { padding:5px 2px; } .stat span { font-size:10px; } }
  </style>
</head>
<body>
  <div id="map"></div>
  <aside class="panel" id="panel">
    <div class="panel-head">
      <div class="title">钱塘江站点分布</div>
      <button class="icon-button" id="collapsePanel" title="收起面板" aria-label="收起面板"><i data-lucide="panel-left-close"></i></button>
    </div>
    <div class="panel-body">
      <div class="stats" id="stats"></div>
      <div class="search-wrap">
        <input class="search" id="stationSearch" type="search" placeholder="搜索站名、河流或站码" autocomplete="off" aria-label="搜索站点">
        <i class="search-icon" data-lucide="search"></i>
        <div class="search-results" id="searchResults"></div>
      </div>
      <div class="section">
        <div class="section-title">图层</div>
        <div class="filters">
          <label><input type="checkbox" data-layer="national_wq" checked><span class="legend-mark national"></span>国控水质</label>
          <label><input type="checkbox" data-layer="provincial_wq" checked><span class="legend-mark provincial"></span>省控水质</label>
          <label><input type="checkbox" data-layer="flow" checked><span class="legend-mark flow"></span>流量站</label>
          <label><input type="checkbox" data-layer="level" checked><span class="legend-mark level"></span>水位站</label>
          <label><input type="checkbox" data-layer="rainfall" checked><span class="legend-mark rain"></span>降水站</label>
          <label><input type="checkbox" id="showApproximate" checked>近似坐标</label>
          <label><input type="checkbox" id="showLabels">站名标签</label>
        </div>
      </div>
      <div class="section">
        <div class="section-title">工具</div>
        <div class="tool-row">
          <button class="tool-button" id="fitVisible"><i data-lucide="maximize"></i><span>全部</span></button>
          <button class="tool-button" id="measure"><i data-lucide="ruler"></i><span>测距</span></button>
          <button class="tool-button" id="clearMeasure"><i data-lucide="trash-2"></i><span>清除</span></button>
        </div>
        <div class="measure-status" id="measureStatus"></div>
      </div>
      <div class="section" id="missingSection"></div>
      <div class="confidence">虚线标记表示地名或扫描图近似位置，只用于查看空间分布；定量距离以站点级坐标为准。当前近似坐标 __APPROXIMATE_COUNT__ 个。</div>
    </div>
  </aside>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script src="https://unpkg.com/lucide@0.468.0/dist/umd/lucide.min.js"></script>
  <script>
    const stations = __STATIONS__;
    const missingStations = __MISSING__;
    const layerCounts = __COUNTS__;
    const styleByLayer = {
      national_wq: { color:'#1d4ed8', fillColor:'#2563eb', radius:5.5, weight:1.2, fillOpacity:.88 },
      provincial_wq: { color:'#c2410c', fillColor:'#f97316', radius:5.0, weight:1.2, fillOpacity:.82 },
      flow: { color:'#15803d', fillColor:'#22c55e', radius:7.0, weight:2.2, fillOpacity:.26 },
      level: { color:'#7c3aed', fillColor:'#a78bfa', radius:6.5, weight:2.0, fillOpacity:.24 },
      rainfall: { color:'#0e7490', fillColor:'#06b6d4', radius:6.0, weight:1.8, fillOpacity:.70 }
    };
    const reliable = new Set(['official_section_coordinate','official_station_coordinate','official_open_data_coordinate','prior_curated_station_coordinate','named_feature_coordinate']);
    const map = L.map('map', { preferCanvas:true, zoomControl:false }).setView([__CENTER_LAT__, __CENTER_LON__], 8);
    L.control.zoom({ position:'bottomright' }).addTo(map);
    const street = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom:19, attribution:'&copy; OpenStreetMap contributors' }).addTo(map);
    const imagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom:18, attribution:'Tiles &copy; Esri' });
    L.control.layers({ '街道图':street, '卫星影像':imagery }, null, { position:'bottomright', collapsed:true }).addTo(map);

    function esc(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
    function layersOf(record) { return String(record.layer_keys || '').split('|').filter(Boolean); }
    function primaryLayer(record) { const layers=layersOf(record); return ['flow','level','rainfall','national_wq','provincial_wq'].find(key => layers.includes(key)) || 'provincial_wq'; }
    function isApproximate(record) { return !reliable.has(String(record.coordinate_confidence)); }
    function sourceHtml(record) {
      const label=esc(record.coordinate_source || record.coordinate_source_file || '');
      return record.coordinate_source_url ? `<a class="source-link" href="${esc(record.coordinate_source_url)}" target="_blank" rel="noreferrer">${label}</a>` : label;
    }
    function popupHtml(record) {
      const approx=isApproximate(record);
      return `<div class="popup-title">${esc(record.station)}<span class="popup-badge">${esc(record.station_type)}</span></div>
        <dl class="popup-grid">
          <dt>站码</dt><dd>${esc(record.station_codes || '无')}</dd>
          <dt>河流/区域</dt><dd>${esc(record.river || '未记录')}</dd>
          <dt>城市区县</dt><dd>${esc([record.city,record.county].filter(Boolean).join(' ') || '未记录')}</dd>
          <dt>本地数据</dt><dd>${esc(record.data_types || record.data_status)}</dd>
          <dt>数据时间</dt><dd>${esc([record.data_start,record.data_end].filter(Boolean).join(' 至 ') || '未汇总')}</dd>
          <dt>坐标</dt><dd>${Number(record.latitude).toFixed(5)}, ${Number(record.longitude).toFixed(5)}</dd>
          <dt>坐标等级</dt><dd class="${approx ? 'low-note' : ''}">${approx ? '近似位置' : '站点级位置'}</dd>
          <dt>坐标来源</dt><dd>${sourceHtml(record)}</dd>
          ${record.note ? `<dt>备注</dt><dd>${esc(record.note)}</dd>` : ''}
        </dl>`;
    }

    const markerEntries=[];
    const entryById=new Map();
    for (const record of stations) {
      const key=primaryLayer(record); const style={...styleByLayer[key]};
      if (isApproximate(record)) style.dashArray='4 3';
      const marker=L.circleMarker([record.latitude,record.longitude],style).bindPopup(popupHtml(record));
      const entry={record,marker}; markerEntries.push(entry); entryById.set(record.record_id,entry);
      marker.on('click', event => handleMeasureClick(entry,event));
    }

    const filters={};
    document.querySelectorAll('[data-layer]').forEach(input => { filters[input.dataset.layer]=input; input.addEventListener('change',redraw); });
    document.getElementById('showApproximate').addEventListener('change',redraw);
    document.getElementById('showLabels').addEventListener('change',refreshLabels);
    function isVisible(entry) {
      const layerVisible=layersOf(entry.record).some(key => filters[key]?.checked);
      return layerVisible && (document.getElementById('showApproximate').checked || !isApproximate(entry.record));
    }
    function bindLabel(entry) {
      entry.marker.unbindTooltip();
      entry.marker.bindTooltip(esc(entry.record.station), { permanent:document.getElementById('showLabels').checked, direction:'top', className:'station-label', offset:[0,-6] });
    }
    function refreshLabels() { markerEntries.forEach(bindLabel); }
    function redraw() {
      markerEntries.forEach(entry => { if (isVisible(entry)) entry.marker.addTo(map); else entry.marker.remove(); });
      refreshLabels();
    }
    function fitVisible() {
      const points=markerEntries.filter(isVisible).map(entry => [entry.record.latitude,entry.record.longitude]);
      if (points.length) map.fitBounds(points,{padding:[34,34],maxZoom:12});
    }

    const countItems=[['national_wq','国控'],['provincial_wq','省控'],['flow','流量'],['level','水位'],['rainfall','降水']];
    document.getElementById('stats').innerHTML=countItems.map(([key,label]) => `<div class="stat"><b>${layerCounts[key] || 0}</b><span>${label}</span></div>`).join('');
    if (missingStations.length) {
      document.getElementById('missingSection').innerHTML=`<details><summary>未定位站点 ${missingStations.length} 个</summary><ul class="missing-list">${missingStations.map(item => `<li>${esc(item.station)}（${esc(item.network)}）</li>`).join('')}</ul></details>`;
    }

    const searchInput=document.getElementById('stationSearch'); const resultsBox=document.getElementById('searchResults');
    function searchable(record) { return [record.station,record.station_codes,record.river,record.city,record.county,record.station_type].join(' ').toLowerCase(); }
    function openEntry(entry) { if (!map.hasLayer(entry.marker)) entry.marker.addTo(map); map.setView(entry.marker.getLatLng(),13); entry.marker.openPopup(); resultsBox.style.display='none'; }
    function search() {
      const query=searchInput.value.trim().toLowerCase();
      if (!query) { resultsBox.style.display='none'; return; }
      const matches=markerEntries.filter(entry => searchable(entry.record).includes(query)).slice(0,10);
      resultsBox.innerHTML=matches.length ? matches.map(entry => `<button class="search-result" data-id="${esc(entry.record.record_id)}">${esc(entry.record.station)}<small>${esc(entry.record.station_type)} · ${esc(entry.record.river || '河流未记录')}</small></button>`).join('') : '<div class="search-result">没有匹配站点</div>';
      resultsBox.style.display='block';
      resultsBox.querySelectorAll('[data-id]').forEach(button => button.addEventListener('click',() => openEntry(entryById.get(button.dataset.id))));
    }
    searchInput.addEventListener('input',search);
    searchInput.addEventListener('keydown',event => { if (event.key==='Enter') { const first=resultsBox.querySelector('[data-id]'); if (first) first.click(); } });
    document.addEventListener('click',event => { if (!event.target.closest('.search-wrap')) resultsBox.style.display='none'; });

    let measuring=false; let measurePoints=[]; let measureLayer=L.layerGroup().addTo(map);
    const measureButton=document.getElementById('measure'); const measureStatus=document.getElementById('measureStatus');
    function setMeasureStatus(text) { measureStatus.textContent=text; }
    function clearMeasure() { measurePoints=[]; measureLayer.clearLayers(); setMeasureStatus(measuring ? '依次点击两个站点' : ''); }
    function handleMeasureClick(entry,event) {
      if (!measuring) return;
      if (event.originalEvent) L.DomEvent.stopPropagation(event.originalEvent);
      measurePoints.push({name:entry.record.station,latlng:entry.marker.getLatLng()});
      L.circleMarker(entry.marker.getLatLng(),{radius:10,color:'#0f172a',weight:2,fillOpacity:0}).addTo(measureLayer);
      if (measurePoints.length===1) { setMeasureStatus(`起点：${entry.record.station}，请选择终点`); return; }
      const [a,b]=measurePoints; const km=map.distance(a.latlng,b.latlng)/1000;
      L.polyline([a.latlng,b.latlng],{color:'#0f172a',weight:2,dashArray:'6 5'}).addTo(measureLayer);
      L.marker([(a.latlng.lat+b.latlng.lat)/2,(a.latlng.lng+b.latlng.lng)/2],{icon:L.divIcon({className:'measure-label',html:`${km.toFixed(2)} km`})}).addTo(measureLayer);
      setMeasureStatus(`${a.name} 至 ${b.name}：${km.toFixed(2)} km（直线距离）`); measuring=false; measureButton.classList.remove('active');
    }
    measureButton.addEventListener('click',() => { measuring=!measuring; measureButton.classList.toggle('active',measuring); clearMeasure(); if (measuring) setMeasureStatus('依次点击两个站点'); });
    document.getElementById('clearMeasure').addEventListener('click',() => { measuring=false; measureButton.classList.remove('active'); clearMeasure(); });
    document.getElementById('fitVisible').addEventListener('click',fitVisible);

    const panel=document.getElementById('panel'); const collapse=document.getElementById('collapsePanel');
    collapse.addEventListener('click',() => { const collapsed=panel.classList.toggle('collapsed'); collapse.title=collapsed ? '展开面板' : '收起面板'; collapse.innerHTML=`<i data-lucide="${collapsed ? 'panel-left-open' : 'panel-left-close'}"></i>`; lucide.createIcons(); });
    redraw(); fitVisible(); lucide.createIcons();
  </script>
</body>
</html>'''


def main() -> int:
    """Build map tables and the standalone interactive HTML file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    national = load_national_locations()
    provincial = load_provincial_locations()
    hydromet = load_hydromet_locations()
    water_quality = pd.concat([national, provincial], ignore_index=True)
    combined = pd.concat([water_quality, hydromet], ignore_index=True)

    national.to_csv(OUTPUT_DIR / "national_station_locations.csv", index=False, encoding="utf-8-sig")
    provincial.to_csv(OUTPUT_DIR / "provincial_station_locations.csv", index=False, encoding="utf-8-sig")
    water_quality.to_csv(OUTPUT_DIR / "station_locations_combined.csv", index=False, encoding="utf-8-sig")
    hydromet.to_csv(OUTPUT_DIR / "hydromet_station_locations.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "all_station_locations.csv", index=False, encoding="utf-8-sig")
    build_distance_tables(water_quality, OUTPUT_DIR)
    build_map_html(combined, OUTPUT_DIR / "station_locations_map.html")

    summary_rows = []
    for layer in ("national_wq", "provincial_wq", "flow", "level", "rainfall"):
        selected = combined["layer_keys"].astype(str).str.split("|").apply(lambda values: layer in values)
        group = combined[selected]
        summary_rows.append(
            {
                "layer": layer,
                "stations": int(len(group)),
                "with_coordinates": int(group["longitude"].notna().sum()),
                "station_level_coordinates": int(group["coordinate_confidence"].map(is_reliable_coordinate).sum()),
                "coordinate_confidence_counts": json.dumps(group["coordinate_confidence"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "station_location_summary.csv", index=False, encoding="utf-8-sig")
    console.print(summary.to_string(index=False))
    console.print(f"wrote {OUTPUT_DIR / 'station_locations_map.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
