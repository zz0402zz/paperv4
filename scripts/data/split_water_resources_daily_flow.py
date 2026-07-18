from __future__ import annotations

from scripts.common.terminal_output import console

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path("data/流量数据-水利厅")
SOURCE_FILE = BASE_DIR / "日平均流量表2015-01-01-2025-12-24.xls"
OUTPUT_DIR = BASE_DIR / "按站点拆分"
STATION_OUTPUT_DIR = OUTPUT_DIR / "日平均流量表_按站点"
SUMMARY_FILE = OUTPUT_DIR / "日平均流量_按站点拆分汇总.csv"
GAP_FILE = OUTPUT_DIR / "日平均流量_连续性断点.csv"


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


def read_daily_flow() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_FILE, sheet_name=0, header=1, engine="xlrd")
    df = df.dropna(how="all").copy()
    df["站码"] = df["站码"].map(normalize_station_code)
    df["站名"] = df["站名"].astype(str).str.strip()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["平均流量"] = pd.to_numeric(df["平均流量"], errors="coerce")
    return df.dropna(subset=["站码", "站名", "日期"])


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in STATION_OUTPUT_DIR.glob("*.csv"):
        old_file.unlink()

    df = read_daily_flow()
    summary_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []

    for (station_code, station_name), group in df.groupby(["站码", "站名"], sort=True, dropna=False):
        group = group.sort_values("日期").copy()
        group["原序号"] = group["序号"]
        group["序号"] = range(1, len(group) + 1)
        group["日期"] = group["日期"].dt.strftime("%Y-%m-%d")
        group = group[["序号", "原序号", "站码", "站名", "日期", "平均流量", "平均流量注解码"]]

        output_path = (
            STATION_OUTPUT_DIR / f"{safe_name(station_code)}_{safe_name(station_name)}_日平均流量表.csv"
        )
        group.to_csv(output_path, index=False, encoding="utf-8-sig")

        actual_dates = pd.to_datetime(group["日期"])
        expected_dates = pd.date_range(actual_dates.min(), actual_dates.max(), freq="D")
        missing_dates = expected_dates.difference(actual_dates)
        duplicate_count = int(group.duplicated(["站码", "站名", "日期"]).sum())

        summary_rows.append(
            {
                "站码": station_code,
                "站名": station_name,
                "行数": len(group),
                "起始日期": actual_dates.min().strftime("%Y-%m-%d"),
                "结束日期": actual_dates.max().strftime("%Y-%m-%d"),
                "期望日数": len(expected_dates),
                "缺失日期数": len(missing_dates),
                "平均流量缺失数": int(group["平均流量"].isna().sum()),
                "重复站点日期行数": duplicate_count,
                "输出文件": str(output_path),
            }
        )

        for start, end, days in date_ranges(missing_dates):
            gap_rows.append(
                {
                    "站码": station_code,
                    "站名": station_name,
                    "缺失开始": start,
                    "缺失结束": end,
                    "缺失天数": days,
                    "输出文件": str(output_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    gaps = pd.DataFrame(gap_rows, columns=["站码", "站名", "缺失开始", "缺失结束", "缺失天数", "输出文件"])
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    gaps.to_csv(GAP_FILE, index=False, encoding="utf-8-sig")

    console.print(f"拆分完成: {len(summary)} 个站点文件")
    console.print(f"输出目录: {STATION_OUTPUT_DIR}")
    console.print(f"汇总文件: {SUMMARY_FILE}")
    console.print(f"断点文件: {GAP_FILE}")
    console.print(summary.to_string(index=False))
    if not gaps.empty:
        console.print("断点汇总:")
        console.print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
