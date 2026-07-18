from __future__ import annotations

from scripts.common.terminal_output import console

import re
import warnings
from pathlib import Path

import pandas as pd


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
)

BASE_DIR = Path("data/降水和水位和补充流量")
OUTPUT_DIR = BASE_DIR / "按站点拆分"
LONG_STATION_OUTPUT_DIR = OUTPUT_DIR / "降水极值表_按站点"
WIDE_STATION_OUTPUT_DIR = OUTPUT_DIR / "降水极值日事件宽表_按站点"
COMBINED_DATA_FILES = [
    OUTPUT_DIR / "降水极值_标准长表.csv",
    OUTPUT_DIR / "降水极值_日事件宽表.csv",
]
YEARS = (2023, 2024, 2025)
COVERAGE_YEAR = 2022

TABLES = [
    {
        "统计类型": "分钟时段最大降水",
        "根目录文件": BASE_DIR / "分钟时段最大降水量表.xls",
        "年份子目录": "分钟时段最大降水量表",
        "年份文件": "分钟时段最大降水量表.xls",
        "时段单位": "分钟",
    },
    {
        "统计类型": "小时时段最大降水",
        "根目录文件": BASE_DIR / "小时时段最大降水量表.xls",
        "年份子目录": "小时时段最大降水量表",
        "年份文件": "小时时段最大降水量表.xls",
        "时段单位": "小时",
    },
]

OUTPUT_COLUMNS = [
    "统计类型",
    "站码",
    "站名",
    "行政区划码",
    "流域水系码",
    "年",
    "起时间",
    "起时间精度",
    "开始日期",
    "最大降水量时段长",
    "时段单位",
    "时段分钟",
    "最大降水量",
    "最大降水量注解码",
    "附注",
    "来源文件",
    "来源格式",
]


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "-", "--", "—", "nan", "NaN"}


def safe_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[\\/:*?\"<>|\\s]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_station_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\\.0", text):
        text = text[:-2]
    return text


def parse_number(value: object) -> float | None:
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def parse_month_day(value: object, year: int) -> pd.Timestamp | None:
    if is_blank(value):
        return None
    text = str(value).strip().replace("—", "-").replace("－", "-").replace("/", "-")
    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", text)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    return pd.Timestamp(year=year, month=month, day=day)


def duration_minutes(duration: object, unit: str) -> int | None:
    duration_value = parse_int(duration)
    if duration_value is None:
        return None
    if unit == "小时":
        return duration_value * 60
    return duration_value


def read_root_long_table(config: dict[str, object]) -> pd.DataFrame:
    path = Path(config["根目录文件"])
    unit = str(config["时段单位"])
    stat_type = str(config["统计类型"])
    df = pd.read_excel(path, sheet_name=0, header=1, engine="xlrd")
    df = df.dropna(how="all").copy()
    df["统计类型"] = stat_type
    df["站码"] = df["站码"].map(normalize_station_code)
    df["站名"] = df["站名"].astype(str).str.strip()
    df["年"] = pd.to_numeric(df["年"], errors="coerce").astype("Int64")
    df["起时间"] = pd.to_datetime(df["起时间"], errors="coerce")
    df["起时间精度"] = "日期时间"
    df["开始日期"] = df["起时间"].dt.strftime("%Y-%m-%d")
    df["时段单位"] = unit
    df["时段分钟"] = df["最大降水量时段长"].map(lambda x: duration_minutes(x, unit))
    df["来源文件"] = str(path)
    df["来源格式"] = "根目录长表"
    return df[OUTPUT_COLUMNS]


def year_file_path(config: dict[str, object], year: int) -> Path:
    return BASE_DIR / f"{year}年" / str(config["年份子目录"]) / str(config["年份文件"])


def read_year_minute_table(path: Path, stat_type: str, year: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    duration_cols = {
        col_idx: parse_int(value)
        for col_idx, value in enumerate(raw.iloc[3])
        if parse_int(value) is not None
    }
    rows: list[dict[str, object]] = []
    for row_idx in range(6, len(raw), 2):
        if row_idx + 1 >= len(raw):
            break
        station_code = normalize_station_code(raw.iat[row_idx, 1])
        station_name = raw.iat[row_idx, 2]
        if not station_code or is_blank(station_name):
            continue
        station_name = str(station_name).strip()
        for col_idx, duration in duration_cols.items():
            rainfall = parse_number(raw.iat[row_idx, col_idx])
            start_date = parse_month_day(raw.iat[row_idx + 1, col_idx], year)
            if rainfall is None and start_date is None:
                continue
            rows.append(
                {
                    "统计类型": stat_type,
                    "站码": station_code,
                    "站名": station_name,
                    "行政区划码": None,
                    "流域水系码": None,
                    "年": year,
                    "起时间": start_date,
                    "起时间精度": "日期",
                    "开始日期": None if start_date is None else start_date.strftime("%Y-%m-%d"),
                    "最大降水量时段长": duration,
                    "时段单位": "分钟",
                    "时段分钟": duration,
                    "最大降水量": rainfall,
                    "最大降水量注解码": None,
                    "附注": None,
                    "来源文件": str(path),
                    "来源格式": "年份矩阵表",
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def read_year_hour_table(path: Path, stat_type: str, year: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    duration_cols = {
        col_idx: parse_int(value)
        for col_idx, value in enumerate(raw.iloc[3])
        if parse_int(value) is not None
    }
    rows: list[dict[str, object]] = []
    for row_idx in range(6, len(raw)):
        station_code = normalize_station_code(raw.iat[row_idx, 1])
        station_name = raw.iat[row_idx, 2]
        if not station_code or is_blank(station_name):
            continue
        station_name = str(station_name).strip()
        for col_idx, duration in duration_cols.items():
            rainfall = parse_number(raw.iat[row_idx, col_idx])
            month = parse_int(raw.iat[row_idx, col_idx + 1])
            day = parse_int(raw.iat[row_idx, col_idx + 2])
            start_date = None
            if month is not None and day is not None:
                start_date = pd.Timestamp(year=year, month=month, day=day)
            if rainfall is None and start_date is None:
                continue
            rows.append(
                {
                    "统计类型": stat_type,
                    "站码": station_code,
                    "站名": station_name,
                    "行政区划码": None,
                    "流域水系码": None,
                    "年": year,
                    "起时间": start_date,
                    "起时间精度": "日期",
                    "开始日期": None if start_date is None else start_date.strftime("%Y-%m-%d"),
                    "最大降水量时段长": duration,
                    "时段单位": "小时",
                    "时段分钟": duration * 60 if duration is not None else None,
                    "最大降水量": rainfall,
                    "最大降水量注解码": None,
                    "附注": None,
                    "来源文件": str(path),
                    "来源格式": "年份矩阵表",
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def read_year_tables(config: dict[str, object]) -> list[pd.DataFrame]:
    stat_type = str(config["统计类型"])
    unit = str(config["时段单位"])
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        path = year_file_path(config, year)
        if not path.exists():
            continue
        if unit == "分钟":
            frames.append(read_year_minute_table(path, stat_type, year))
        else:
            frames.append(read_year_hour_table(path, stat_type, year))
    return frames


def build_long_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config in TABLES:
        frames.append(read_root_long_table(config))
        frames.extend(read_year_tables(config))

    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    df["站码"] = df["站码"].map(normalize_station_code)
    df["站名"] = df["站名"].astype(str).str.strip()
    df["年"] = pd.to_numeric(df["年"], errors="coerce").astype("Int64")
    df["行政区划码"] = pd.to_numeric(df["行政区划码"], errors="coerce").astype("Int64")
    df["流域水系码"] = pd.to_numeric(df["流域水系码"], errors="coerce").astype("Int64")
    df["起时间"] = pd.to_datetime(df["起时间"], errors="coerce")
    df["开始日期"] = df["起时间"].dt.strftime("%Y-%m-%d")
    df["最大降水量时段长"] = pd.to_numeric(df["最大降水量时段长"], errors="coerce").astype("Int64")
    df["时段分钟"] = pd.to_numeric(df["时段分钟"], errors="coerce").astype("Int64")
    df["最大降水量"] = pd.to_numeric(df["最大降水量"], errors="coerce")
    df = df.sort_values(["统计类型", "站码", "站名", "年", "时段分钟"]).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def build_daily_event_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.dropna(subset=["开始日期", "最大降水量"]).copy()
    df["特征名"] = df.apply(
        lambda row: f"{row['统计类型']}_{int(row['时段分钟'])}min_mm",
        axis=1,
    )
    wide = (
        df.pivot_table(
            index=["站码", "站名", "开始日期"],
            columns="特征名",
            values="最大降水量",
            aggfunc="max",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    id_cols = ["站码", "站名", "开始日期"]

    def feature_order(column: str) -> tuple[int, int, str]:
        match = re.search(r"_(\d+)min_mm$", column)
        duration = int(match.group(1)) if match else 10**9
        stat_order = 0 if column.startswith("分钟时段最大降水") else 1
        return stat_order, duration, column

    feature_cols = sorted([col for col in wide.columns if col not in id_cols], key=feature_order)
    wide = wide[id_cols + feature_cols]
    return wide.sort_values(id_cols).reset_index(drop=True)


def split_by_station(long_df: pd.DataFrame, daily_wide: pd.DataFrame) -> pd.DataFrame:
    LONG_STATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WIDE_STATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in LONG_STATION_OUTPUT_DIR.glob("*.csv"):
        old_file.unlink()
    for old_file in WIDE_STATION_OUTPUT_DIR.glob("*.csv"):
        old_file.unlink()

    summary_rows: list[dict[str, object]] = []
    for (station_code, station_name), group in long_df.groupby(["站码", "站名"], sort=True, dropna=False):
        long_output_path = (
            LONG_STATION_OUTPUT_DIR / f"{safe_name(station_code)}_{safe_name(station_name)}_降水极值表.csv"
        )
        group.to_csv(long_output_path, index=False, encoding="utf-8-sig")

        wide_group = daily_wide[
            (daily_wide["站码"].astype(str) == str(station_code))
            & (daily_wide["站名"].astype(str) == str(station_name))
        ].copy()
        wide_output_path = (
            WIDE_STATION_OUTPUT_DIR
            / f"{safe_name(station_code)}_{safe_name(station_name)}_降水极值日事件宽表.csv"
        )
        wide_group.to_csv(wide_output_path, index=False, encoding="utf-8-sig")

        for stat_type, stat_group in group.groupby("统计类型", sort=True):
            duplicate_count = int(
                stat_group.duplicated(["站码", "站名", "年", "时段分钟", "统计类型"]).sum()
            )
            summary_rows.append(
                {
                    "统计类型": stat_type,
                    "站码": station_code,
                    "站名": station_name,
                    "行数": len(stat_group),
                    "起始年": int(stat_group["年"].min()),
                    "结束年": int(stat_group["年"].max()),
                    "年份数": int(stat_group["年"].nunique()),
                    "历时数": int(stat_group["时段分钟"].nunique()),
                    "重复站点年份历时行数": duplicate_count,
                    "数值缺失数": int(stat_group["最大降水量"].isna().sum()),
                    "日期缺失数": int(stat_group["开始日期"].isna().sum()),
                    "长表输出文件": str(long_output_path),
                    "日事件宽表输出文件": str(wide_output_path),
                }
            )
    return pd.DataFrame(summary_rows)


def build_missing_duration_report(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stat_type, station_code, station_name), group in long_df.groupby(
        ["统计类型", "站码", "站名"], sort=True, dropna=False
    ):
        years = sorted(int(year) for year in group["年"].dropna().unique())
        durations = sorted(int(duration) for duration in group["时段分钟"].dropna().unique())
        existing = set(zip(group["年"].astype(int), group["时段分钟"].astype(int)))
        for year in years:
            for duration in durations:
                if (year, duration) in existing:
                    continue
                rows.append(
                    {
                        "统计类型": stat_type,
                        "站码": station_code,
                        "站名": station_name,
                        "年": year,
                        "缺失时段分钟": duration,
                    }
                )
    return pd.DataFrame(rows, columns=["统计类型", "站码", "站名", "年", "缺失时段分钟"])


def build_year_coverage_report(
    long_df: pd.DataFrame,
    daily_wide: pd.DataFrame,
    year: int = COVERAGE_YEAR,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    station_keys = long_df[["站码", "站名"]].drop_duplicates().sort_values(["站码", "站名"])
    daily = daily_wide.copy()
    daily["开始日期"] = pd.to_datetime(daily["开始日期"], errors="coerce")
    daily["站码"] = daily["站码"].astype(str)
    daily["站名"] = daily["站名"].astype(str)
    for _, station in station_keys.iterrows():
        station_code = str(station["站码"])
        station_name = str(station["站名"])
        station_daily = daily[
            (daily["站码"].astype(str) == station_code)
            & (daily["站名"].astype(str) == station_name)
            & (daily["开始日期"].dt.year == year)
        ]
        daily_event_days = int(station_daily["开始日期"].nunique())
        for stat_type in sorted(long_df["统计类型"].dropna().unique()):
            group = long_df[
                (long_df["站码"].astype(str) == station_code)
                & (long_df["站名"].astype(str) == station_name)
                & (long_df["统计类型"].astype(str) == str(stat_type))
                & (pd.to_numeric(long_df["年"], errors="coerce") == year)
            ].copy()
            if group.empty:
                status = "无该年数据"
            elif group["最大降水量"].isna().any() or group["开始日期"].isna().any():
                status = "有2022降水极值但存在缺值"
            else:
                status = "有2022降水极值"
            rows.append(
                {
                    "统计类型": stat_type,
                    "站码": station_code,
                    "站名": station_name,
                    "年份": year,
                    "行数": int(len(group)),
                    "历时数": int(group["时段分钟"].nunique()) if not group.empty else 0,
                    "日事件天数": daily_event_days,
                    "数值缺失数": int(group["最大降水量"].isna().sum()) if not group.empty else 0,
                    "日期缺失数": int(group["开始日期"].isna().sum()) if not group.empty else 0,
                    "覆盖状态": status,
                }
            )
    return pd.DataFrame(rows)


def remove_combined_data_files() -> None:
    for path in COMBINED_DATA_FILES:
        if path.exists():
            path.unlink()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = build_long_table()
    daily_wide = build_daily_event_wide(long_df)
    summary = split_by_station(long_df, daily_wide)
    missing = build_missing_duration_report(long_df)
    coverage = build_year_coverage_report(long_df, daily_wide, COVERAGE_YEAR)
    remove_combined_data_files()

    summary_path = OUTPUT_DIR / "降水极值_转换汇总.csv"
    missing_path = OUTPUT_DIR / "降水极值_缺失年份历时组合.csv"
    coverage_path = OUTPUT_DIR / f"降水极值_{COVERAGE_YEAR}覆盖情况.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    console.print(f"标准长表按站点目录: {LONG_STATION_OUTPUT_DIR} rows={len(long_df)}")
    console.print(f"日事件宽表按站点目录: {WIDE_STATION_OUTPUT_DIR} rows={len(daily_wide)}")
    console.print(f"汇总表: {summary_path}")
    console.print(f"缺失组合表: {missing_path} rows={len(missing)}")
    console.print(f"{COVERAGE_YEAR}覆盖情况: {coverage_path}")
    console.print(summary.to_string(index=False))
    if not missing.empty:
        console.print("缺失年份-历时组合预览:")
        console.print(missing.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
