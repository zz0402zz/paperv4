from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# [01] 日尺度水文气象特征：每个 4 小时时刻只使用前 1/2/3 天，避免当天日均值泄漏。
DEFAULT_LAGS = (1, 2, 3)
HYDROMET_FEATURE_COLUMNS = tuple(
    column
    for lag in DEFAULT_LAGS
    for column in (
        f"hydro_flow_log1p_lag{lag}d",
        f"hydro_level_lag{lag}d",
        f"meteo_rain60_log1p_lag{lag}d",
        f"meteo_rain1440_log1p_lag{lag}d",
    )
)

FLOW_DIRS = (
    Path("data/流量数据-水利厅/按站点拆分/日平均流量表_按站点"),
    Path("data/降水和水位和补充流量/按站点拆分/日平均流量表_按站点"),
)
LEVEL_DIR = Path("data/降水和水位和补充流量/按站点拆分/日平均水位表_按站点")
RAIN_DAILY_EVENT_DIR = Path("data/降水和水位和补充流量/按站点拆分/降水极值日事件宽表_按站点")
FLOW_VALID_FROM = pd.Timestamp("2021-01-01")
RAIN_FEATURE_SEMANTICS = "annual_duration_maximum_event_only"

FLOW_STATION_CODES = {
    "开化": "70100160",
    "常山(三)": "70100350",
    "衢州": "70100500",
    "兰溪": "70100900",
    "富春江电站": "70101500",
    "江山(二)": "70104660",
    "金华": "70108400",
    "武义": "70110050",
    "新安江电站": "70112000",
    "分水江": "70115250",
    "诸暨(二)": "70117700",
}
LEVEL_STATION_CODES = {
    "开化": "70100160",
    "龙游": "70100600",
    "富春江电站": "70101500",
    "双塔底": "70104700",
    "新安江电站": "70112000",
    "梅城": "70112400",
    "分水江": "70115250",
    "浦江": "70116900",
    "安华(二)": "70117400",
    "枫桥": "70119500",
    "萧山义桥": "70211510",
}
RAIN_STATION_CODES = {
    "前河": "70123900",
    "泽随": "70127600",
    "横锦水库": "70128800",
    "金华": "70132200",
    "大市": "70146800",
    "瑶琳": "70149360",
    "富阳大源": "70151800",
    "苏溪": "70153400",
    "杨佳山": "70155000",
    "应店街": "70155900",
}


# [02] 第一版人工映射：代表性水文/雨量站，不等同于严格水动力站点。
STATION_HYDROMET_MAP: dict[str, dict[str, str | None]] = {
    "上仙屋": {"flow_station": "诸暨(二)", "level_station": "浦江", "rain_station": "应店街"},
    "下界首": {"flow_station": "开化", "level_station": "开化", "rain_station": "前河"},
    "下童": {"flow_station": "衢州", "level_station": "龙游", "rain_station": "泽随"},
    "东关桥": {"flow_station": "金华", "level_station": None, "rain_station": "横锦水库"},
    "东迹渡": {"flow_station": "衢州", "level_station": "龙游", "rain_station": "泽随"},
    "义东桥": {"flow_station": "金华", "level_station": None, "rain_station": "横锦水库"},
    "南江桥": {"flow_station": "金华", "level_station": None, "rain_station": "横锦水库"},
    "双塔底": {"flow_station": "江山(二)", "level_station": "双塔底", "rain_station": "泽随"},
    "双港口": {"flow_station": "江山(二)", "level_station": "双塔底", "rain_station": "泽随"},
    "台口": {"flow_station": "金华", "level_station": None, "rain_station": "横锦水库"},
    "塔下洲": {"flow_station": "金华", "level_station": None, "rain_station": "苏溪"},
    "富足山": {"flow_station": "常山(三)", "level_station": "开化", "rain_station": "前河"},
    "将军岩": {"flow_station": "兰溪", "level_station": "梅城", "rain_station": "金华"},
    "桐君山": {"flow_station": "分水江", "level_station": "分水江", "rain_station": "瑶琳"},
    "桐庐": {"flow_station": "富春江电站", "level_station": "富春江电站", "rain_station": "瑶琳"},
    "横山": {"flow_station": "衢州", "level_station": "龙游", "rain_station": "金华"},
    "洋溪渡": {"flow_station": "新安江电站", "level_station": "新安江电站", "rain_station": "大市"},
    "洪坞桥": {"flow_station": "武义", "level_station": None, "rain_station": "金华"},
    "浦阳江出口": {"flow_station": "诸暨(二)", "level_station": "枫桥", "rain_station": "应店街"},
    "浮石渡": {"flow_station": "衢州", "level_station": "龙游", "rain_station": "泽随"},
    "湄池": {"flow_station": "诸暨(二)", "level_station": "安华(二)", "rain_station": "应店街"},
    "章店": {"flow_station": "武义", "level_station": None, "rain_station": "金华"},
    "费垅": {"flow_station": "金华", "level_station": None, "rain_station": "金华"},
    "郑家": {"flow_station": "衢州", "level_station": "龙游", "rain_station": "泽随"},
    "闸口": {"flow_station": "富春江电站", "level_station": "萧山义桥", "rain_station": "富阳大源"},
}

for _mapping in STATION_HYDROMET_MAP.values():
    _mapping["flow_station_code"] = FLOW_STATION_CODES.get(str(_mapping.get("flow_station")))
    _mapping["level_station_code"] = LEVEL_STATION_CODES.get(str(_mapping.get("level_station")))
    _mapping["rain_station_code"] = RAIN_STATION_CODES.get(str(_mapping.get("rain_station")))


def _feature_columns_for_lags(lags: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        column
        for lag in lags
        for column in (
            f"hydro_flow_log1p_lag{lag}d",
            f"hydro_level_lag{lag}d",
            f"meteo_rain60_log1p_lag{lag}d",
            f"meteo_rain1440_log1p_lag{lag}d",
        )
    )


def _normalize_daily_index(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work.index = pd.to_datetime(work.index).normalize()
    return work.sort_index()


def read_daily_value_tables(dirs: tuple[Path, ...], value_column: str) -> pd.DataFrame:
    """[03] 读取日均流量/水位，并以站码为列，避免同名站被错误平均。"""
    frames = []
    for source_priority, directory in enumerate(dirs):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            raw = pd.read_csv(path, dtype={"站码": str})
            if "日期" not in raw.columns or "站名" not in raw.columns or value_column not in raw.columns:
                continue
            raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
            raw[value_column] = pd.to_numeric(raw[value_column], errors="coerce")
            raw["_station_key"] = raw["站码"].astype(str) if "站码" in raw else raw["站名"].astype(str)
            raw["_source_priority"] = source_priority
            frames.append(raw[["日期", "_station_key", value_column, "_source_priority"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["日期", "_station_key", "_source_priority"], kind="stable")
    combined = combined.drop_duplicates(["日期", "_station_key"], keep="first")
    wide = combined.pivot(index="日期", columns="_station_key", values=value_column)
    return _normalize_daily_index(wide)


def read_daily_flow() -> pd.DataFrame:
    """[04] 读取日均流量宽表。"""
    flow = read_daily_value_tables(FLOW_DIRS, "平均流量")
    return flow.loc[flow.index >= FLOW_VALID_FROM]


def read_daily_level() -> pd.DataFrame:
    """[05] 读取日均水位宽表。"""
    return read_daily_value_tables((LEVEL_DIR,), "平均水位")


def read_daily_rain_extremes(directory: Path = RAIN_DAILY_EVENT_DIR) -> dict[str, pd.DataFrame]:
    """[06] 读取降水极值日事件，生成 60min/1440min 两类日事件宽表。"""
    frames = []
    if not directory.exists():
        return {"rain60": pd.DataFrame(), "rain1440": pd.DataFrame()}
    for path in sorted(directory.glob("*.csv")):
        raw = pd.read_csv(path, dtype={"站码": str})
        if "开始日期" not in raw.columns or "站名" not in raw.columns:
            continue
        raw["开始日期"] = pd.to_datetime(raw["开始日期"], errors="coerce")
        rain60_cols = [column for column in raw.columns if "60min_mm" in column]
        rain1440_cols = [column for column in raw.columns if "1440min_mm" in column]
        frame = raw[["开始日期", "站名"]].copy()
        frame["_station_key"] = raw["站码"].astype(str) if "站码" in raw else raw["站名"].astype(str)
        frame["rain60"] = raw[rain60_cols].apply(pd.to_numeric, errors="coerce").max(axis=1) if rain60_cols else np.nan
        frame["rain1440"] = (
            raw[rain1440_cols].apply(pd.to_numeric, errors="coerce").max(axis=1) if rain1440_cols else np.nan
        )
        frames.append(frame)
    if not frames:
        return {"rain60": pd.DataFrame(), "rain1440": pd.DataFrame()}
    combined = pd.concat(frames, ignore_index=True)
    result = {}
    for metric in ("rain60", "rain1440"):
        wide = combined.pivot_table(index="开始日期", columns="_station_key", values=metric, aggfunc="max")
        result[metric] = _normalize_daily_index(wide)
    return result


def _lookup_daily_value(wide: pd.DataFrame, station: str | None, dates: pd.Series) -> np.ndarray:
    if station is None or wide.empty or station not in wide.columns:
        return np.full(len(dates), np.nan, dtype=float)
    values = wide.reindex(pd.to_datetime(dates).dt.normalize())[station].to_numpy(float)
    return values


def _mapping_station_key(mapping: dict[str, str | None], kind: str) -> str | None:
    return mapping.get(f"{kind}_station_code") or mapping.get(f"{kind}_station")


def signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def build_station_hydromet_features(
    data: pd.DataFrame,
    mapping: dict[str, dict[str, str | None]],
    flow: pd.DataFrame,
    level: pd.DataFrame,
    rain: dict[str, pd.DataFrame],
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """[07] 按 station/time 生成历史日尺度水文气象特征。"""
    work = data[["station", "time"]].copy()
    work["station"] = work["station"].astype(str)
    work["time"] = pd.to_datetime(work["time"])
    feature_columns = _feature_columns_for_lags(lags)
    for column in feature_columns:
        work[column] = np.nan

    flow = _normalize_daily_index(flow) if not flow.empty else flow
    level = _normalize_daily_index(level) if not level.empty else level
    rain60 = _normalize_daily_index(rain.get("rain60", pd.DataFrame())) if rain.get("rain60") is not None else pd.DataFrame()
    rain1440 = (
        _normalize_daily_index(rain.get("rain1440", pd.DataFrame()))
        if rain.get("rain1440") is not None
        else pd.DataFrame()
    )

    for station, idx in work.groupby("station", sort=False).groups.items():
        station_mapping = mapping.get(station, {})
        dates = work.loc[idx, "time"].dt.normalize()
        for lag in lags:
            lookup_dates = dates - pd.Timedelta(days=lag)
            flow_values = _lookup_daily_value(flow, _mapping_station_key(station_mapping, "flow"), lookup_dates)
            level_values = _lookup_daily_value(level, _mapping_station_key(station_mapping, "level"), lookup_dates)
            rain60_values = _lookup_daily_value(rain60, _mapping_station_key(station_mapping, "rain"), lookup_dates)
            rain1440_values = _lookup_daily_value(rain1440, _mapping_station_key(station_mapping, "rain"), lookup_dates)

            work.loc[idx, f"hydro_flow_log1p_lag{lag}d"] = signed_log1p(flow_values)
            work.loc[idx, f"hydro_level_lag{lag}d"] = level_values
            work.loc[idx, f"meteo_rain60_log1p_lag{lag}d"] = np.log1p(np.clip(rain60_values, 0, None))
            work.loc[idx, f"meteo_rain1440_log1p_lag{lag}d"] = np.log1p(np.clip(rain1440_values, 0, None))
    return work


def hydromet_coverage(features: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """[08] 按站点统计外部特征可用情况。"""
    rows = []
    for station, group in features.groupby("station", sort=True):
        total = int(len(group) * len(feature_columns))
        available = int(group.loc[:, feature_columns].notna().sum().sum())
        rows.append(
            {
                "station": station,
                "rows": int(len(group)),
                "hydromet_feature_count": len(feature_columns),
                "available_hydromet_values": available,
                "total_hydromet_values": total,
                "available_rate": float(available / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def add_hydromet_features(
    data: pd.DataFrame,
    mapping: dict[str, dict[str, str | None]] | None = None,
    flow: pd.DataFrame | None = None,
    level: pd.DataFrame | None = None,
    rain: dict[str, pd.DataFrame] | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """[09] 将历史日尺度水文气象特征拼到水质长表。"""
    mapping = STATION_HYDROMET_MAP if mapping is None else mapping
    flow = read_daily_flow() if flow is None else flow
    level = read_daily_level() if level is None else level
    rain = read_daily_rain_extremes() if rain is None else rain
    feature_columns = _feature_columns_for_lags(lags)
    features = build_station_hydromet_features(data, mapping, flow, level, rain, lags)
    key = ["station", "time"]
    augmented = data.copy()
    augmented["station"] = augmented["station"].astype(str)
    augmented["time"] = pd.to_datetime(augmented["time"])
    augmented = augmented.merge(features[key + list(feature_columns)], on=key, how="left")
    return augmented, feature_columns, hydromet_coverage(features, feature_columns)


def mapping_frame(mapping: dict[str, dict[str, str | None]] | None = None) -> pd.DataFrame:
    """[10] 输出当前水质站到水文/雨量站的映射表，方便人工检查。"""
    mapping = STATION_HYDROMET_MAP if mapping is None else mapping
    rows = []
    for station, item in mapping.items():
        rows.append(
            {
                "station": station,
                "flow_station": item.get("flow_station"),
                "flow_station_code": item.get("flow_station_code"),
                "level_station": item.get("level_station"),
                "level_station_code": item.get("level_station_code"),
                "rain_station": item.get("rain_station"),
                "rain_station_code": item.get("rain_station_code"),
            }
        )
    return pd.DataFrame(rows).sort_values("station")
