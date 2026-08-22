"""在不修改原始数据的前提下，审计国控站五指标数据质量。

这个脚本严格区分：
1. 模型拖累：预测表现差，只用于定位，不作为删站依据；
2. 数据质量证据：缺测、超物理范围、连续相同、突变、孤立尖峰和分布漂移；
3. 可用降雨证据：只能支持“可能是真实水文响应”，不能在未匹配时证明仪器错误。

输出只写入 outputs/数据质量审计，不会覆盖数据或模型结果。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
OBSERVED_PATH = ROOT / "data/processed/v2/quantity_4h_observed.csv"
QUALITY_PATH = ROOT / "data/processed/v2/quantity_4h_quality.csv"
STATION_PATH = ROOT / "data/metadata/station_metadata.csv"
HYDROMET_PATH = ROOT / "data/metadata/hydromet_station_locations_curated.csv"
RAIN_PATH = ROOT / "data/processed/v2/precipitation/precipitation_4h_model.csv"
FORECAST_PATH = (
    ROOT
    / "outputs/多指标联合水质预测/验证集/混合输出表示五种子确认"
    / "两种表示逐站点指标时距结果.csv"
)
OUTPUT_DIR = ROOT / "outputs/数据质量审计/国控站五指标"

TARGETS = (
    "pH(无量纲)",
    "溶解氧(mg/L)",
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
)
RAIN_SENSITIVE_TARGETS = {
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
}
TRAIN_START = datetime(2022, 1, 1)
VALIDATION_START = datetime(2024, 1, 1)
END = datetime(2025, 1, 1)
STEP = timedelta(hours=4)
RAIN_WINDOW = timedelta(hours=72)


@dataclass(frozen=True)
class Point:
    time: datetime
    value: float | None
    target_ok: bool
    hard_invalid: int
    soft_suspect: bool
    duplicate_conflict: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成国控站五指标数据质量审计报告")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--forecast-results", type=Path, default=FORECAST_PATH)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def quantile(values: list[float], probability: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * probability
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return clean[left]
    weight = position - left
    return clean[left] * (1.0 - weight) + clean[right] * weight


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not math.isfinite(denominator):
        return None
    return numerator / denominator


def format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def load_series() -> tuple[dict[tuple[str, str], list[Point]], list[str]]:
    series: dict[tuple[str, str], list[Point]] = defaultdict(list)
    stations: set[str] = set()
    with OBSERVED_PATH.open("r", encoding="utf-8-sig", newline="") as observed_file, QUALITY_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as quality_file:
        observed_reader = csv.DictReader(observed_file)
        quality_reader = csv.DictReader(quality_file)
        for observed, quality in zip(observed_reader, quality_reader, strict=True):
            station = observed["station"]
            if station != quality["station"] or observed["time"] != quality["time"]:
                raise ValueError("观测表与质量表行未对齐")
            time = datetime.fromisoformat(observed["time"])
            if not TRAIN_START <= time < END:
                continue
            stations.add(station)
            duplicate_conflict = as_bool(quality.get("duplicate_conflict"))
            for target in TARGETS:
                series[(station, target)].append(
                    Point(
                        time=time,
                        value=as_float(observed.get(target)),
                        target_ok=as_bool(quality.get(f"{target}__target_ok")),
                        hard_invalid=int(float(quality.get(f"{target}__hard_invalid_count", "0") or 0)),
                        soft_suspect=as_bool(quality.get(f"{target}__soft_suspect")),
                        duplicate_conflict=duplicate_conflict,
                    )
                )
    return series, sorted(stations)


def load_station_coordinates() -> dict[str, tuple[float, float, str]]:
    result: dict[str, tuple[float, float, str]] = {}
    with STATION_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            lon = as_float(row.get("longitude"))
            lat = as_float(row.get("latitude"))
            if lon is not None and lat is not None:
                result[row["station"]] = (lon, lat, row.get("river", ""))
    return result


def load_rain_coordinates() -> dict[str, tuple[float, float, str]]:
    # 当前可验证的连续降雨表只有金华和武义有相对可靠的坐标。
    allowed = {"金华", "武义"}
    result: dict[str, tuple[float, float, str]] = {}
    with HYDROMET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["station"]
            if name not in allowed or name in result:
                continue
            lon = as_float(row.get("longitude"))
            lat = as_float(row.get("latitude"))
            if lon is not None and lat is not None:
                result[name] = (lon, lat, row.get("coordinate_confidence", ""))
    return result


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, left)
    lon2, lat2 = map(math.radians, right)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def build_rain_mapping(
    stations: list[str],
) -> tuple[dict[str, tuple[str, float, str, str]], list[dict[str, object]]]:
    water = load_station_coordinates()
    rain = load_rain_coordinates()
    river_routing = {
        "金华江": "金华",
        "东阳江": "金华",
        "武义江": "武义",
        "永康江": "武义",
    }
    mapping: dict[str, tuple[str, float, str, str]] = {}
    rows: list[dict[str, object]] = []
    for station in stations:
        if station not in water or not rain:
            continue
        water_lon, water_lat, river = water[station]
        routed_name = river_routing.get(river)
        nearest_name, nearest_values = (
            (routed_name, rain[routed_name])
            if routed_name in rain
            else min(rain.items(), key=lambda item: haversine_km((water_lon, water_lat), item[1][:2]))
        )
        rain_lon, rain_lat, confidence = nearest_values
        distance = haversine_km((water_lon, water_lat), (rain_lon, rain_lat))
        # 只对能用河流关系路由的站点做区域匹配；“距离最近”不足以证明同流域。
        usable = routed_name is not None and distance <= 80.0
        scope = "同河系区域证据" if usable else "仅记录，不用于归因"
        mapping[station] = (nearest_name, distance, confidence, scope)
        rows.append(
            {
                "station": station,
                "river": river,
                "nearest_rain_station": nearest_name,
                "distance_km": distance,
                "rain_coordinate_confidence": confidence,
                "rain_evidence_scope": scope,
            }
        )
    return mapping, rows


def load_rain() -> dict[str, list[tuple[datetime, float]]]:
    result: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    if not RAIN_PATH.exists():
        return result
    with RAIN_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            amount = as_float(row.get("rainfall_mm"))
            time = datetime.fromisoformat(row["time"])
            if amount is not None and amount > 0 and TRAIN_START <= time < END:
                result[row["station"]].append((time, amount))
    for values in result.values():
        values.sort()
    return result


def rainfall_before(
    station: str,
    time: datetime,
    rain_mapping: dict[str, tuple[str, float, str, str]],
    rain: dict[str, list[tuple[datetime, float]]],
) -> float | None:
    mapping = rain_mapping.get(station)
    if mapping is None or mapping[3] != "同河系区域证据":
        return None
    rain_station = mapping[0]
    total = 0.0
    matched = False
    for rain_time, amount in rain.get(rain_station, []):
        if rain_time > time:
            break
        if time - RAIN_WINDOW <= rain_time <= time:
            total += amount
            matched = True
    return total if matched else 0.0


def valid_values(points: list[Point], start: datetime, end: datetime) -> list[float]:
    return [
        point.value
        for point in points
        if start <= point.time < end and point.target_ok and point.value is not None
    ]


def consecutive_changes(points: list[Point], start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    result: list[tuple[datetime, float]] = []
    previous: Point | None = None
    for point in points:
        if not start <= point.time < end or not point.target_ok or point.value is None:
            previous = None
            continue
        if previous is not None and point.time - previous.time == STEP and previous.value is not None:
            result.append((point.time, point.value - previous.value))
        previous = point
    return result


def count_ratio_jumps(points: list[Point], start: datetime, end: datetime) -> int:
    count = 0
    previous: Point | None = None
    for point in points:
        if not start <= point.time < end or not point.target_ok or point.value is None:
            previous = None
            continue
        if previous is not None and point.time - previous.time == STEP and previous.value is not None:
            # 依据国家地表水自动监测数据审核规则，前后值超3倍或低于1/3列为存疑。
            if previous.value > 0 and (point.value >= 3 * previous.value or point.value <= previous.value / 3):
                count += 1
        previous = point
    return count


def count_flatline_points(points: list[Point], start: datetime, end: datetime) -> int:
    count = 0
    run_value: float | None = None
    run_length = 0
    previous_time: datetime | None = None
    for point in points:
        if not start <= point.time < end or not point.target_ok or point.value is None:
            run_value = None
            run_length = 0
            previous_time = None
            continue
        if previous_time is not None and point.time - previous_time == STEP and point.value == run_value:
            run_length += 1
        else:
            run_value = point.value
            run_length = 1
        if run_length >= 3:
            count += 1
        previous_time = point.time
    return count


def longest_flatline_run(points: list[Point], start: datetime, end: datetime) -> int:
    longest = 0
    run_value: float | None = None
    run_length = 0
    previous_time: datetime | None = None
    for point in points:
        if not start <= point.time < end or not point.target_ok or point.value is None:
            run_value = None
            run_length = 0
            previous_time = None
            continue
        if previous_time is not None and point.time - previous_time == STEP and point.value == run_value:
            run_length += 1
        else:
            run_value = point.value
            run_length = 1
        longest = max(longest, run_length)
        previous_time = point.time
    return longest


def summarize_target(points: list[Point]) -> tuple[dict[str, object], list[datetime]]:
    train_values = valid_values(points, TRAIN_START, VALIDATION_START)
    validation_values = valid_values(points, VALIDATION_START, END)
    train_changes = consecutive_changes(points, TRAIN_START, VALIDATION_START)
    validation_changes = consecutive_changes(points, VALIDATION_START, END)
    train_abs_changes = [abs(value) for _, value in train_changes]
    rate_threshold = quantile(train_abs_changes, 0.999)
    rate_outlier_times = [
        time
        for time, change in validation_changes
        if rate_threshold is not None and rate_threshold > 0 and abs(change) > rate_threshold
    ]

    train_q25 = quantile(train_values, 0.25)
    train_q75 = quantile(train_values, 0.75)
    train_median = quantile(train_values, 0.5)
    validation_median = quantile(validation_values, 0.5)
    train_iqr = None if train_q25 is None or train_q75 is None else train_q75 - train_q25
    median_shift_iqr = None
    if (
        train_median is not None
        and validation_median is not None
        and train_iqr is not None
        and train_iqr > 0
    ):
        median_shift_iqr = (validation_median - train_median) / train_iqr

    expected_train = int((VALIDATION_START - TRAIN_START) / STEP)
    expected_validation = int((END - VALIDATION_START) / STEP)
    train_points = [point for point in points if TRAIN_START <= point.time < VALIDATION_START]
    validation_points = [point for point in points if VALIDATION_START <= point.time < END]
    result: dict[str, object] = {
        "train_valid_rows": len(train_values),
        "validation_valid_rows": len(validation_values),
        "train_coverage": len(train_values) / expected_train,
        "validation_coverage": len(validation_values) / expected_validation,
        "train_hard_invalid_count": sum(point.hard_invalid for point in train_points),
        "validation_hard_invalid_count": sum(point.hard_invalid for point in validation_points),
        "train_soft_suspect_count": sum(point.soft_suspect for point in train_points),
        "validation_soft_suspect_count": sum(point.soft_suspect for point in validation_points),
        "validation_duplicate_conflict_rows": sum(point.duplicate_conflict for point in validation_points),
        "validation_ratio_jump_count": count_ratio_jumps(points, VALIDATION_START, END),
        "validation_flatline_points": count_flatline_points(points, VALIDATION_START, END),
        "validation_longest_flatline_steps": longest_flatline_run(points, VALIDATION_START, END),
        "validation_rate_outlier_count": len(rate_outlier_times),
        "train_median": train_median,
        "validation_median": validation_median,
        "validation_q99": quantile(validation_values, 0.99),
        "validation_max": max(validation_values) if validation_values else None,
        "median_shift_iqr": median_shift_iqr,
        "rate_threshold_from_train_q999": rate_threshold,
    }
    return result, rate_outlier_times


def load_forecast_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != "mixed_linear":
                continue
            key = (row["station"], row["target"])
            for field in ("rmse", "nse", "relative_rmse_pct"):
                value = as_float(row.get(field))
                if value is not None:
                    grouped[key][field].append(value)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, metrics in grouped.items():
        result[key] = {
            field: sum(values) / len(values)
            for field, values in metrics.items()
            if values
        }
    return result


def classify_quality(row: dict[str, object]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    high_reasons: list[str] = []
    if float(row["train_coverage"]) < 0.8 or float(row["validation_coverage"]) < 0.8:
        high_reasons.append("有效覆盖率低于80%")
    valid_rows = max(int(row["validation_valid_rows"]), 1)
    # 99.9%阈值在约2200个验证点中自然会产生约2个超限点。
    # 即使超限点较多，也可能是昼夜周期或水文非平稳性，不足以单独构成测量故障证据。
    unexplained_threshold = max(11, math.ceil(valid_rows * 0.005))
    if int(row["validation_unexplained_isolated_rate_outliers"]) >= unexplained_threshold:
        reasons.append(
            f"未解释孤立突变达{int(row['validation_unexplained_isolated_rate_outliers'])}次"
            f"（密集复核阈值{unexplained_threshold}次，需排除日周期）"
        )
    # 单个软存疑值值得查看，但不足以把整个指标列为高风险。
    if int(row["validation_soft_suspect_count"]) >= 5:
        high_reasons.append("验证年至少5个软存疑值仍被当作正式标签")
    median_shift = row.get("median_shift_iqr")
    if isinstance(median_shift, float) and abs(median_shift) > 3:
        high_reasons.append("验证年中位数相对训练期漂移超过3个IQR")
    if int(row["validation_longest_flatline_steps"]) >= 12:
        high_reasons.append("存在至少48小时的连续相同值")

    if int(row["validation_hard_invalid_count"]) > 0:
        reasons.append("验证年出现硬无效值")
    if int(row["validation_ratio_jump_count"]) > 0:
        reasons.append("存在前值3倍或1/3突变")
    if int(row["validation_soft_suspect_count"]) > 0:
        reasons.append("存在软存疑值被当作正式标签")
    if int(row["validation_duplicate_conflict_rows"]) > 0:
        reasons.append("存在重复时刻冲突")
    if int(row["validation_rate_outlier_count"]) > 0:
        reasons.append("存在超出训练期99.9%变化量的事件")
    if int(row["validation_longest_flatline_steps"]) >= 3:
        reasons.append(
            f"最长连续相同{int(row['validation_longest_flatline_steps']) * 4}小时"
        )

    if high_reasons:
        return "高优先级人工复核", high_reasons + reasons
    if reasons:
        return "普通复核", reasons
    return "暂无明显质量证据", []


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_number(row.get(field)) for field in fieldnames})


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    if not selected:
        return "（无）"
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in selected:
        values = [format_number(row.get(field), 3).replace("|", "\\|") for field, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    rows: list[dict[str, object]], station_rows: list[dict[str, object]], rain_rows: list[dict[str, object]]
) -> str:
    high = [row for row in rows if row["quality_status"] == "高优先级人工复核"]
    drag = [row for row in rows if row["model_drag"] == "是"]
    station_review = [row for row in station_rows if int(row["high_priority_target_count"]) >= 3]
    high_sorted = sorted(
        high,
        key=lambda row: (
            -int(row["validation_unexplained_isolated_rate_outliers"]),
            float(row["validation_coverage"]),
        ),
    )
    station_sorted = sorted(station_rows, key=lambda row: -int(row["high_priority_target_count"]))
    return f"""# 国控站五指标数据质量审计

## 结论边界

- 本报告仅审计 2022—2024 开发期数据，没有读取 2025 测试集结果，也没有修改任何原始数据。
- “模型拖累”只是定位线索，不是删站依据。否则会形成按验证集结果挑数据的选择偏差。
- 降雨匹配是区域性支持证据：匹配到可能说明突变是真实水文响应；未匹配到不能证明是测量错误，因为降雨表是稀疏的，且当前可靠坐标只覆盖金华、武义。
- “三项及以上需复核”只能进入站点级人工核查；确认仪器故障、校准漂移或维护记录后，才能纳入敏感性排除名单。

## 审计概况

- 站点—指标组合：{len(rows)}
- 高优先级人工复核：{len(high)}
- 模型拖累组合（平均 NSE < 0 或平均 RMSE 差于持续性）：{len(drag)}
- 三项及以上指标需高优先级复核的站点：{len(station_review)}

## 站点级复核顺序

{markdown_table(station_sorted, [("station", "站点"), ("high_priority_target_count", "高优先级指标数"), ("model_drag_target_count", "模型拖累指标数"), ("station_action", "处置")])}

## 高优先级指标组合

{markdown_table(high_sorted, [("station", "站点"), ("target", "指标"), ("validation_coverage", "2024覆盖率"), ("validation_soft_suspect_count", "软存疑"), ("validation_rate_outlier_count", "突变事件"), ("validation_unexplained_isolated_rate_outliers", "未解释孤立突变"), ("mean_nse", "平均NSE"), ("quality_reasons", "复核原因")])}

## 处理建议

1. 优先查看高优先级组合的原始小时数据、审核状态和站房运维记录。
2. 对氨氮、总磷和高锰酸盐指数，先检查降雨/streamflow 以及浊度、电导率等多指标共变，不要把真实污染冲刷峰当成异常值。
3. 先冻结“主分析队列”和“质量敏感性队列”，同时报告两套结果，不删原始文件。
4. 完成质量规则的验证集消融后，才固化预处理并解锁 2025 测试集。

## 方法说明

- 覆盖率：按固定 4 小时网格计算 `target_ok` 有效比例。
- 比值突变：相邻有效值大于等于前值 3 倍或小于等于 1/3。
- 连续相同：至少 3 个连续 4 小时点完全相同。
- 突变事件：2024 绝对变化量超过 2022—2023 训练期绝对变化量的 99.9% 分位数；它只是稳健复核线索，不直接删值。
- 多指标共变：同站同时有两个及以上目标同时出现突变。
- 降雨支持：最近可用区域雨量站过去 72 小时内有经校验的降雨记录。
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    series, stations = load_series()
    rain_mapping, rain_mapping_rows = build_rain_mapping(stations)
    rain = load_rain()
    forecast = load_forecast_metrics(args.forecast_results.resolve())

    rows: list[dict[str, object]] = []
    rate_outlier_times: dict[tuple[str, str], list[datetime]] = {}
    for station in stations:
        for target in TARGETS:
            summary, times = summarize_target(series[(station, target)])
            rate_outlier_times[(station, target)] = times
            row: dict[str, object] = {"station": station, "target": target, **summary}
            row.update(
                {
                    "nearest_rain_station": rain_mapping.get(station, ("", 0.0, "", ""))[0],
                    "rain_station_distance_km": rain_mapping.get(station, ("", None, "", ""))[1],
                    "rain_evidence_scope": rain_mapping.get(station, ("", 0.0, "", "无可用区域降雨证据"))[3],
                }
            )
            model = forecast.get((station, target), {})
            row["mean_rmse"] = model.get("rmse")
            row["mean_nse"] = model.get("nse")
            row["mean_relative_rmse_pct"] = model.get("relative_rmse_pct")
            mean_nse = row["mean_nse"]
            mean_relative = row["mean_relative_rmse_pct"]
            row["model_drag"] = "是" if (
                isinstance(mean_nse, float) and mean_nse < 0
            ) or (
                isinstance(mean_relative, float) and mean_relative > 0
            ) else "否"
            rows.append(row)

    anomaly_count_by_station_time: dict[tuple[str, datetime], int] = defaultdict(int)
    for (station, _target), times in rate_outlier_times.items():
        for time in times:
            anomaly_count_by_station_time[(station, time)] += 1

    for row in rows:
        key = (str(row["station"]), str(row["target"]))
        rain_supported = 0
        multivariate_supported = 0
        unexplained_isolated = 0
        rain_amounts: list[float] = []
        for time in rate_outlier_times[key]:
            rain_amount = rainfall_before(key[0], time, rain_mapping, rain)
            has_rain = rain_amount is not None and rain_amount > 0
            coherent = anomaly_count_by_station_time[(key[0], time)] >= 2
            if has_rain:
                rain_supported += 1
                rain_amounts.append(float(rain_amount))
            if coherent:
                multivariate_supported += 1
            if not has_rain and not coherent:
                unexplained_isolated += 1
        row["validation_rain_supported_rate_outliers"] = rain_supported
        row["validation_multivariate_rate_outliers"] = multivariate_supported
        row["validation_unexplained_isolated_rate_outliers"] = unexplained_isolated
        row["max_matched_72h_rainfall_mm"] = max(rain_amounts) if rain_amounts else None
        row["rain_sensitive_target"] = "是" if key[1] in RAIN_SENSITIVE_TARGETS else "否"
        status, reasons = classify_quality(row)
        row["quality_status"] = status
        row["quality_reasons"] = "；".join(reasons)

    anomaly_rows: list[dict[str, object]] = []
    for key, times in rate_outlier_times.items():
        station, target = key
        point_by_time = {
            point.time: point
            for point in series[key]
            if VALIDATION_START - STEP <= point.time < END
        }
        mapping = rain_mapping.get(station, ("", None, "", "无可用区域降雨证据"))
        threshold = next(
            row["rate_threshold_from_train_q999"]
            for row in rows
            if row["station"] == station and row["target"] == target
        )
        for time in times:
            current = point_by_time.get(time)
            previous = point_by_time.get(time - STEP)
            if current is None or previous is None or current.value is None or previous.value is None:
                continue
            rain_amount = rainfall_before(station, time, rain_mapping, rain)
            coherent_count = anomaly_count_by_station_time[(station, time)]
            if rain_amount is not None and rain_amount > 0:
                evidence = "有区域降雨支持"
            elif coherent_count >= 2:
                evidence = "有多指标共变支持"
            elif rain_amount is None:
                evidence = "无可用降雨覆盖，仅可人工复核"
            else:
                evidence = "可用证据未解释，需人工复核"
            anomaly_rows.append(
                {
                    "station": station,
                    "target": target,
                    "time": time.isoformat(sep=" "),
                    "previous_value": previous.value,
                    "current_value": current.value,
                    "change": current.value - previous.value,
                    "absolute_change": abs(current.value - previous.value),
                    "train_change_q999_threshold": threshold,
                    "same_time_anomalous_target_count": coherent_count,
                    "nearest_rain_station": mapping[0],
                    "rain_station_distance_km": mapping[1],
                    "rainfall_previous_72h_mm": rain_amount,
                    "evidence_interpretation": evidence,
                }
            )

    station_rows: list[dict[str, object]] = []
    for station in stations:
        station_targets = [row for row in rows if row["station"] == station]
        high_count = sum(row["quality_status"] == "高优先级人工复核" for row in station_targets)
        drag_count = sum(row["model_drag"] == "是" for row in station_targets)
        if high_count >= 3:
            action = "站点级人工复核；确认故障后才进入敏感性排除"
        elif high_count > 0:
            action = "指标级复核，暂不删站"
        else:
            action = "保留"
        station_rows.append(
            {
                "station": station,
                "high_priority_target_count": high_count,
                "model_drag_target_count": drag_count,
                "station_action": action,
            }
        )

    detailed_fields = [
        "station", "target", "quality_status", "quality_reasons", "model_drag",
        "mean_rmse", "mean_nse", "mean_relative_rmse_pct", "train_valid_rows",
        "validation_valid_rows", "train_coverage", "validation_coverage",
        "train_hard_invalid_count", "validation_hard_invalid_count",
        "train_soft_suspect_count", "validation_soft_suspect_count",
        "validation_duplicate_conflict_rows", "validation_ratio_jump_count",
        "validation_flatline_points", "validation_longest_flatline_steps",
        "validation_rate_outlier_count",
        "validation_rain_supported_rate_outliers", "validation_multivariate_rate_outliers",
        "validation_unexplained_isolated_rate_outliers", "train_median", "validation_median",
        "validation_q99", "validation_max", "median_shift_iqr",
        "rate_threshold_from_train_q999", "rain_sensitive_target", "nearest_rain_station",
        "rain_station_distance_km", "rain_evidence_scope", "max_matched_72h_rainfall_mm",
    ]
    station_fields = ["station", "high_priority_target_count", "model_drag_target_count", "station_action"]
    rain_fields = [
        "station", "river", "nearest_rain_station", "distance_km",
        "rain_coordinate_confidence", "rain_evidence_scope",
    ]
    anomaly_fields = [
        "station", "target", "time", "previous_value", "current_value", "change",
        "absolute_change", "train_change_q999_threshold",
        "same_time_anomalous_target_count", "nearest_rain_station",
        "rain_station_distance_km", "rainfall_previous_72h_mm", "evidence_interpretation",
    ]
    write_csv(output_dir / "站点指标质量审计.csv", rows, detailed_fields)
    write_csv(output_dir / "站点复核清单.csv", station_rows, station_fields)
    write_csv(output_dir / "降雨匹配范围.csv", rain_mapping_rows, rain_fields)
    write_csv(output_dir / "异常事件复核清单.csv", anomaly_rows, anomaly_fields)
    report = build_report(rows, station_rows, rain_mapping_rows)
    report_path = output_dir / "数据质量审计报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"审计完成 | 站点={len(stations)} | 指标组合={len(rows)}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
