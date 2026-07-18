from __future__ import annotations

from scripts.common.terminal_output import console

import re
from calendar import monthrange
from pathlib import Path

import pandas as pd


BASE_DIR = Path("data/降水和水位和补充流量")
OUTPUT_DIR = BASE_DIR / "按站点拆分"
YEARS = (2023, 2024, 2025)
COVERAGE_YEAR = 2022
MONTHS = {
    "一月": 1,
    "二月": 2,
    "三月": 3,
    "四月": 4,
    "五月": 5,
    "六月": 6,
    "七月": 7,
    "八月": 8,
    "九月": 9,
    "十月": 10,
    "十一月": 11,
    "十二月": 12,
}

TABLES = [
    {
        "kind": "日平均流量",
        "source": BASE_DIR / "日平均流量表.xls",
        "value_col": "平均流量",
        "note_col": "平均流量注解码",
    },
    {
        "kind": "日平均水位",
        "source": BASE_DIR / "日平均水位表.xls",
        "value_col": "平均水位",
        "note_col": "平均水位注解码",
    },
]


def safe_name(text: object) -> str:
    name = str(text).strip()
    name = re.sub(r"[\\/:*?\"<>|\\s]+", "_", name)
    return name.strip("_") or "unknown"


def normalize_station_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\\.0", text):
        text = text[:-2]
    return text


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "-", "--", "—", "nan", "NaN"}


def read_hydro_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=1, engine="xlrd")
    df = df.dropna(how="all").copy()
    df["站码"] = df["站码"].map(normalize_station_code)
    df["站名"] = df["站名"].astype(str).str.strip()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    return df


def parse_year_file_name(path: Path) -> tuple[str, str, str, int]:
    match = re.fullmatch(
        r"(?P<code>\d+)-(?P<year>\d{4})年(?P<name>.+)站逐日平均(?P<kind>流量|水位)表\.xls",
        path.name,
    )
    if not match:
        raise ValueError(f"无法从文件名解析站点信息: {path}")
    return (
        match.group("code"),
        match.group("name"),
        match.group("kind"),
        int(match.group("year")),
    )


def parse_number(value: object) -> float | None:
    if is_blank(value):
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_water_level(value: object, current_integer: int | None) -> tuple[float | None, int | None]:
    if is_blank(value):
        return None, current_integer

    text = str(value).strip()
    number = parse_number(text)
    if number is None:
        return None, current_integer

    has_decimal = "." in text
    is_integer_like = float(number).is_integer()
    if has_decimal or not is_integer_like:
        current_integer = int(number)
        return number, current_integer

    if current_integer is not None and 0 <= number <= 99:
        return current_integer + number / 100, current_integer

    current_integer = int(number)
    return number, current_integer


def find_month_columns(raw: pd.DataFrame) -> tuple[int, dict[int, int]]:
    for row_idx in range(len(raw)):
        month_cols: dict[int, int] = {}
        for col_idx, value in enumerate(raw.iloc[row_idx]):
            month = MONTHS.get(str(value).strip())
            if month is not None:
                month_cols[month] = col_idx
        if len(month_cols) >= 12:
            return row_idx, month_cols
    raise ValueError("未找到月份标题行")


def find_day_rows(raw: pd.DataFrame, month_row: int) -> dict[int, int]:
    day_rows: dict[int, int] = {}
    for row_idx in range(month_row + 1, len(raw)):
        day = parse_number(raw.iat[row_idx, 0])
        if day is None or not float(day).is_integer():
            continue
        day_int = int(day)
        if 1 <= day_int <= 31:
            day_rows[day_int] = row_idx
    if not day_rows:
        raise ValueError("未找到日期行")
    return day_rows


def read_year_matrix(path: Path, value_col: str, note_col: str) -> pd.DataFrame:
    station_code, station_name, kind, year = parse_year_file_name(path)
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    month_row, month_cols = find_month_columns(raw)
    day_rows = find_day_rows(raw, month_row)

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        current_integer: int | None = None
        _, days_in_month = monthrange(year, month)
        col_idx = month_cols[month]
        for day in range(1, days_in_month + 1):
            row_idx = day_rows.get(day)
            raw_value = None if row_idx is None else raw.iat[row_idx, col_idx]
            if kind == "水位":
                value, current_integer = parse_water_level(raw_value, current_integer)
            else:
                value = parse_number(raw_value)

            rows.append(
                {
                    "序号": None,
                    "原序号": None,
                    "站码": station_code,
                    "站名": station_name,
                    "日期": pd.Timestamp(year=year, month=month, day=day),
                    value_col: value,
                    note_col: None,
                    "来源文件": str(path),
                    "来源格式": "年份矩阵表",
                }
            )

    return pd.DataFrame(rows)


def find_year_files(kind: str) -> list[Path]:
    pattern = f"*逐日平均{kind}表.xls"
    files: list[Path] = []
    for year in YEARS:
        files.extend((BASE_DIR / f"{year}年").glob(f"*/{pattern}"))
    return sorted(files)


def count_year_files(kind: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in find_year_files(kind):
        station_code, station_name, _, _ = parse_year_file_name(path)
        key = (station_code, station_name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_combined_table(config: dict[str, object]) -> pd.DataFrame:
    kind = str(config["kind"]).replace("日平均", "")
    source = Path(config["source"])
    value_col = str(config["value_col"])
    note_col = str(config["note_col"])

    root_df = read_hydro_table(source)
    root_df["来源文件"] = str(source)
    root_df["来源格式"] = "根目录长表"

    year_frames = [read_year_matrix(path, value_col, note_col) for path in find_year_files(kind)]
    combined = pd.concat([root_df, *year_frames], ignore_index=True, sort=False)
    combined["站码"] = combined["站码"].map(normalize_station_code)
    combined["站名"] = combined["站名"].astype(str).str.strip()
    combined["日期"] = pd.to_datetime(combined["日期"], errors="coerce")
    combined = combined.dropna(subset=["站码", "站名", "日期"]).copy()
    combined = combined.sort_values(["站码", "站名", "日期", "来源格式"]).copy()
    combined = combined.drop_duplicates(["站码", "站名", "日期"], keep="last")
    return combined


def split_one_table(config: dict[str, object]) -> list[dict[str, object]]:
    kind = str(config["kind"])
    raw_kind = kind.replace("日平均", "")
    value_col = str(config["value_col"])
    note_col = str(config["note_col"])
    year_file_counts = count_year_files(raw_kind)
    target_dir = OUTPUT_DIR / f"{kind}表_按站点"
    target_dir.mkdir(parents=True, exist_ok=True)
    for old_file in target_dir.glob("*.csv"):
        old_file.unlink()

    df = build_combined_table(config)
    summary_rows: list[dict[str, object]] = []

    for (station_code, station_name), group in df.groupby(["站码", "站名"], sort=True, dropna=False):
        group = group.sort_values("日期").copy()
        if "原序号" not in group.columns:
            group["原序号"] = None
        group["原序号"] = group["原序号"].where(group["原序号"].notna(), group["序号"])
        group["序号"] = range(1, len(group) + 1)

        cols = ["序号", "原序号", "站码", "站名", "日期", value_col, note_col]
        group = group[cols]
        group["日期"] = group["日期"].dt.strftime("%Y-%m-%d")

        filename = f"{safe_name(station_code)}_{safe_name(station_name)}_{kind}表.csv"
        output_path = target_dir / filename
        group.to_csv(output_path, index=False, encoding="utf-8-sig")

        dup_count = int(group.duplicated(["站码", "站名", "日期"]).sum())
        full_dates = pd.date_range(group["日期"].min(), group["日期"].max(), freq="D")
        missing_dates = full_dates.difference(pd.to_datetime(group["日期"]))
        year_file_count = year_file_counts.get((str(station_code), str(station_name)), 0)
        if year_file_count == 0:
            continuation_status = "未找到2023-2025年表，已跳过接续"
        elif len(missing_dates) == 0 and group["日期"].max() == "2025-12-31":
            continuation_status = "已接续到2025-12-31"
        elif len(missing_dates) == 0:
            continuation_status = "已接续但结束日期未到2025-12-31"
        else:
            continuation_status = "接续后仍有断点"
        summary_rows.append(
            {
                "数据类型": kind,
                "站码": station_code,
                "站名": station_name,
                "2023-2025年表文件数": year_file_count,
                "接续状态": continuation_status,
                "行数": len(group),
                "起始日期": group["日期"].min(),
                "结束日期": group["日期"].max(),
                "期望日数": len(full_dates),
                "缺失日期数": len(missing_dates),
                "指标列": value_col,
                "数值缺失数": int(group[value_col].isna().sum()),
                "重复站点日期行数": dup_count,
                "输出文件": str(output_path),
            }
        )

    return summary_rows


def date_ranges(dates: pd.DatetimeIndex) -> list[tuple[str, str, int]]:
    if len(dates) == 0:
        return []

    ranges: list[tuple[str, str, int]] = []
    start = prev = dates[0]
    for date in dates[1:]:
        if (date - prev).days == 1:
            prev = date
            continue
        ranges.append((start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d"), (prev - start).days + 1))
        start = prev = date
    ranges.append((start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d"), (prev - start).days + 1))
    return ranges


def build_gap_report(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ["数据类型", "站码", "站名", "缺失开始", "缺失结束", "缺失天数", "输出文件"]
    for item in summary.to_dict("records"):
        if int(item["缺失日期数"]) == 0:
            continue
        if int(item["2023-2025年表文件数"]) == 0:
            continue

        output_file = Path(str(item["输出文件"]))
        df = pd.read_csv(output_file)
        actual_dates = pd.to_datetime(df["日期"])
        expected_dates = pd.date_range(item["起始日期"], item["结束日期"], freq="D")
        missing_dates = expected_dates.difference(actual_dates)

        for start, end, days in date_ranges(missing_dates):
            rows.append(
                {
                    "数据类型": item["数据类型"],
                    "站码": item["站码"],
                    "站名": item["站名"],
                    "缺失开始": start,
                    "缺失结束": end,
                    "缺失天数": days,
                    "输出文件": item["输出文件"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_year_value_coverage_report(summary: pd.DataFrame, year: int = COVERAGE_YEAR) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    for item in summary.to_dict("records"):
        output_file = Path(str(item["输出文件"]))
        value_col = str(item["指标列"])
        if not output_file.exists():
            rows.append(
                {
                    "数据类型": item["数据类型"],
                    "站码": item["站码"],
                    "站名": item["站名"],
                    "年份": year,
                    "年内首日": None,
                    "年内末日": None,
                    "年内行数": 0,
                    "年内期望日数": len(expected_dates),
                    "年内缺失日期数": len(expected_dates),
                    "年内数值非空数": 0,
                    "年内数值缺失数": 0,
                    "覆盖状态": "无输出文件",
                    "输出文件": str(output_file),
                }
            )
            continue

        data = pd.read_csv(output_file)
        data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
        year_data = data[data["日期"].dt.year == year].copy()
        actual_dates = pd.DatetimeIndex(year_data["日期"].dropna().dt.normalize().unique())
        missing_dates = expected_dates.difference(actual_dates)
        non_null_values = int(pd.to_numeric(year_data.get(value_col), errors="coerce").notna().sum())
        missing_values = int(pd.to_numeric(year_data.get(value_col), errors="coerce").isna().sum())
        if len(year_data) == 0:
            status = "无该年数据"
        elif len(missing_dates) > 0:
            status = "缺日期"
        elif missing_values > 0:
            status = "日期完整但有缺值"
        else:
            status = "完整有值"
        rows.append(
            {
                "数据类型": item["数据类型"],
                "站码": item["站码"],
                "站名": item["站名"],
                "年份": year,
                "年内首日": None if year_data.empty else year_data["日期"].min().strftime("%Y-%m-%d"),
                "年内末日": None if year_data.empty else year_data["日期"].max().strftime("%Y-%m-%d"),
                "年内行数": int(len(year_data)),
                "年内期望日数": len(expected_dates),
                "年内缺失日期数": int(len(missing_dates)),
                "年内数值非空数": non_null_values,
                "年内数值缺失数": missing_values,
                "覆盖状态": status,
                "输出文件": str(output_file),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, object]] = []
    for config in TABLES:
        all_summary.extend(split_one_table(config))

    summary = pd.DataFrame(all_summary)
    summary_path = OUTPUT_DIR / "日平均流量水位_按站点拆分汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gap_report = build_gap_report(summary)
    gap_path = OUTPUT_DIR / "日平均流量水位_连续性断点.csv"
    gap_report.to_csv(gap_path, index=False, encoding="utf-8-sig")
    coverage_report = build_year_value_coverage_report(summary, COVERAGE_YEAR)
    coverage_path = OUTPUT_DIR / f"日平均流量水位_{COVERAGE_YEAR}覆盖情况.csv"
    coverage_report.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    console.print(f"拆分完成: {len(summary)} 个站点文件")
    console.print(f"输出目录: {OUTPUT_DIR}")
    console.print(f"汇总文件: {summary_path}")
    console.print(f"断点文件: {gap_path}")
    console.print(f"{COVERAGE_YEAR}覆盖情况: {coverage_path}")
    console.print(summary.to_string(index=False))
    if not gap_report.empty:
        console.print("断点汇总:")
        console.print(gap_report.to_string(index=False))


if __name__ == "__main__":
    main()
