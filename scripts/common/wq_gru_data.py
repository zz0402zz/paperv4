from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# [01] 模型输入/输出使用的 9 个水质指标。
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
TIME_FEATURE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)
FEATURE_ENHANCEMENT_MODES = (
    "diff1",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_6",
)


# [02] 硬异常规则；命中的值会先转成缺失值，再由短缺口插值处理。
OUTLIER_RULES = {
    "水温(℃)": lambda s: (s <= 0) | (s > 40),
    "pH(无量纲)": lambda s: (s < 3) | (s > 12),
    "溶解氧(mg/L)": lambda s: (s <= 0) | (s > 25),
    "浊度(NTU)": lambda s: (s < 0) | (s > 5000),
    "电导率(μS/cm)": lambda s: (s <= 20) | (s > 2000),
    "高锰酸盐指数(mg/L)": lambda s: (s <= 0.02) | (s > 20),
    "氨氮(mg/L)": lambda s: (s < 0) | (s > 20),
    "总磷(mg/L)": lambda s: (s < 0) | (s > 5),
    "总氮(mg/L)": lambda s: (s <= 0) | (s > 50),
}
INTERPOLATION_LIMIT = 3


def target_ok_column(feature: str) -> str:
    """Return the quality-sidecar column used to approve training/evaluation targets."""
    return f"{feature}__target_ok"


def processed_data_provenance(path: str | Path) -> dict[str, str]:
    """Return stable data identity fields for every experiment summary."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "processed_data_path": str(path),
        "processed_data_sha256": digest.hexdigest(),
    }


def clean_feature_values(column: str, values: pd.Series) -> pd.Series:
    """[02-1] 修正单个指标：确认的硬异常置空。"""
    cleaned = values.copy()
    cleaned[OUTLIER_RULES[column](cleaned)] = np.nan
    return cleaned


def interpolate_short_series(values: pd.Series, limit: int = INTERPOLATION_LIMIT) -> pd.Series:
    """[02-2] 只填补长度不超过 limit 的内部缺口，长缺口保持缺失。"""
    interpolated = values.interpolate(method="time", limit_area="inside")
    missing = values.isna()
    groups = missing.ne(missing.shift()).cumsum()
    gap_lengths = missing.groupby(groups).transform("sum")
    interpolated[missing & (gap_lengths > limit)] = np.nan
    return interpolated


def interpolate_short_gaps(data: pd.DataFrame, limit: int = INTERPOLATION_LIMIT) -> pd.DataFrame:
    """[02-3] 按站点分别对 4 小时序列做短缺口插值。"""
    work = data[["station", "time", *FEATURE_COLUMNS]].copy()
    work["time"] = pd.to_datetime(work["time"])
    frames = []

    for station, group in work.groupby("station", sort=True):
        group = group.sort_values("time").set_index("time")
        for column in FEATURE_COLUMNS:
            group[column] = interpolate_short_series(group[column], limit)
        group["station"] = station
        frames.append(group.reset_index())

    if not frames:
        return work
    return pd.concat(frames, ignore_index=True)[["station", "time", *FEATURE_COLUMNS]]


def build_interpolation_flags(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """[02-3-1] 标记每个值来源：original、interpolated 或 remaining_missing。"""
    before_idx = _indexed_feature_frame(before)
    after_idx = _indexed_feature_frame(after)
    if not before_idx.index.equals(after_idx.index):
        raise ValueError("before and after interpolation frames must share station/time index.")

    flags = before_idx.reset_index()[["station", "time"]].copy()
    for column in FEATURE_COLUMNS:
        before_values = before_idx[column]
        after_values = after_idx[column]
        flags[f"{column}_status"] = np.select(
            [
                before_values.notna(),
                before_values.isna() & after_values.notna(),
            ],
            ["original", "interpolated"],
            default="remaining_missing",
        )
    return flags


def interpolation_cell_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    start_date: str | None = None,
    statuses: tuple[str, ...] = ("interpolated", "remaining_missing"),
) -> pd.DataFrame:
    """[02-3-2] 输出发生插值或仍缺失的单元格，便于人工抽查。"""
    before_idx = _indexed_feature_frame(before)
    after_idx = _indexed_feature_frame(after)
    if not before_idx.index.equals(after_idx.index):
        raise ValueError("before and after interpolation frames must share station/time index.")

    rows = []
    work = before_idx.reset_index()
    after_work = after_idx.reset_index()
    if start_date is not None:
        mask = pd.to_datetime(work["time"]) >= pd.Timestamp(start_date)
        work = work[mask].reset_index(drop=True)
        after_work = after_work[mask].reset_index(drop=True)

    for station, group in work.groupby("station", sort=True):
        group = group.sort_values("time").reset_index()
        after_group = after_work.loc[group["index"]].sort_values("time").reset_index(drop=True)
        group = group.reset_index(drop=True)
        for column in FEATURE_COLUMNS:
            missing = group[column].isna()
            gap_group = missing.ne(missing.shift()).cumsum()
            gap_length = missing.groupby(gap_group).transform("sum").astype(int)
            gap_start = group["time"].groupby(gap_group).transform("min")
            gap_end = group["time"].groupby(gap_group).transform("max")

            after_values = after_group[column]
            status = np.select(
                [group[column].notna(), group[column].isna() & after_values.notna()],
                ["original", "interpolated"],
                default="remaining_missing",
            )
            selected = np.isin(status, statuses)
            for idx in np.flatnonzero(selected):
                rows.append(
                    {
                        "station": station,
                        "time": group.at[idx, "time"],
                        "feature": column,
                        "status": status[idx],
                        "value_before_interpolation": group.at[idx, column],
                        "value_after_interpolation": after_values.iloc[idx],
                        "gap_length_steps": int(gap_length.iloc[idx]) if missing.iloc[idx] else 0,
                        "gap_length_hours": int(gap_length.iloc[idx] * 4) if missing.iloc[idx] else 0,
                        "gap_start": gap_start.iloc[idx] if missing.iloc[idx] else pd.NaT,
                        "gap_end": gap_end.iloc[idx] if missing.iloc[idx] else pd.NaT,
                    }
                )
    columns = [
        "station",
        "time",
        "feature",
        "status",
        "value_before_interpolation",
        "value_after_interpolation",
        "gap_length_steps",
        "gap_length_hours",
        "gap_start",
        "gap_end",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_interpolation(
    before: pd.DataFrame,
    after: pd.DataFrame,
    start_date: str | None = None,
) -> pd.DataFrame:
    """[02-3-3] 按站点和指标汇总插值数量、剩余缺失和最长缺口。"""
    before_idx = _indexed_feature_frame(before)
    after_idx = _indexed_feature_frame(after)
    if not before_idx.index.equals(after_idx.index):
        raise ValueError("before and after interpolation frames must share station/time index.")

    work = before_idx.reset_index()
    after_work = after_idx.reset_index()
    if start_date is not None:
        mask = pd.to_datetime(work["time"]) >= pd.Timestamp(start_date)
        work = work[mask].reset_index(drop=True)
        after_work = after_work[mask].reset_index(drop=True)

    rows = []
    for station, group in work.groupby("station", sort=True):
        group = group.sort_values("time").reset_index()
        after_group = after_work.loc[group["index"]].sort_values("time").reset_index(drop=True)
        group = group.reset_index(drop=True)
        for column in FEATURE_COLUMNS:
            before_values = group[column]
            after_values = after_group[column]
            missing_before = before_values.isna()
            missing_after = after_values.isna()
            interpolated = missing_before & after_values.notna()
            gap_length = _missing_gap_lengths(missing_before)
            remaining_gap_lengths = gap_length[missing_after].to_numpy(dtype=int)
            original_gap_lengths = gap_length[missing_before].to_numpy(dtype=int)
            interpolated_gap_lengths = gap_length[interpolated].to_numpy(dtype=int)

            rows.append(
                {
                    "station": station,
                    "feature": column,
                    "start_date": start_date or "",
                    "total_points": int(len(group)),
                    "original_valid_points": int(before_values.notna().sum()),
                    "original_missing_points": int(missing_before.sum()),
                    "interpolated_points": int(interpolated.sum()),
                    "remaining_missing_points": int(missing_after.sum()),
                    "original_missing_rate": float(missing_before.mean()) if len(group) else 0.0,
                    "post_missing_rate": float(missing_after.mean()) if len(group) else 0.0,
                    "max_original_gap_steps": int(original_gap_lengths.max()) if len(original_gap_lengths) else 0,
                    "max_interpolated_gap_steps": int(interpolated_gap_lengths.max()) if len(interpolated_gap_lengths) else 0,
                    "max_remaining_gap_steps": int(remaining_gap_lengths.max()) if len(remaining_gap_lengths) else 0,
                    "latest_interpolated_time": str(group.loc[interpolated, "time"].max()) if interpolated.any() else "",
                    "latest_remaining_missing_time": str(group.loc[missing_after, "time"].max()) if missing_after.any() else "",
                }
            )
    return pd.DataFrame(rows)


def _indexed_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """[02-3-4] 生成按 station/time 对齐的指标表。"""
    work = data[["station", "time", *FEATURE_COLUMNS]].copy()
    work["time"] = pd.to_datetime(work["time"])
    return work.sort_values(["station", "time"]).set_index(["station", "time"])


def _missing_gap_lengths(missing: pd.Series) -> pd.Series:
    """[02-3-5] 计算每个缺失点所在连续缺口的长度。"""
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("sum").fillna(0).astype(int)
    lengths[~missing] = 0
    return lengths


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """[02-4] 为每个时间点生成小时和月份的周期特征。"""
    work = data.copy()
    time = pd.to_datetime(work["time"])
    hour = time.dt.hour
    month = time.dt.month - 1
    work["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    work["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    work["month_sin"] = np.sin(2 * np.pi * month / 12)
    work["month_cos"] = np.cos(2 * np.pi * month / 12)
    return work


def feature_enhancement_columns(modes: tuple[str, ...]) -> tuple[str, ...]:
    """[02-5] 根据增强模式生成输入列名。"""
    columns = []
    for mode in modes:
        if mode not in FEATURE_ENHANCEMENT_MODES:
            raise ValueError(f"Unsupported feature enhancement mode: {mode}")
        suffix = {
            "diff1": "diff1",
            "rolling_mean_3": "mean3",
            "rolling_mean_6": "mean6",
            "rolling_std_6": "std6",
        }[mode]
        columns.extend(f"{feature}_{suffix}" for feature in FEATURE_COLUMNS)
    return tuple(columns)


def add_feature_enhancements(data: pd.DataFrame, modes: tuple[str, ...]) -> pd.DataFrame:
    """[02-6] 按站点生成只依赖当前和过去值的增强特征。"""
    if not modes:
        return data.copy()
    feature_enhancement_columns(modes)
    frames = []
    for _, group in data.groupby("station", sort=True):
        group = group.sort_values("time").copy()
        values = group.loc[:, FEATURE_COLUMNS]
        if "diff1" in modes:
            diffs = values.diff()
            diffs.columns = [f"{feature}_diff1" for feature in FEATURE_COLUMNS]
            group = pd.concat([group, diffs], axis=1)
        if "rolling_mean_3" in modes:
            means = values.rolling(window=3, min_periods=1).mean()
            means.columns = [f"{feature}_mean3" for feature in FEATURE_COLUMNS]
            group = pd.concat([group, means], axis=1)
        if "rolling_mean_6" in modes:
            means = values.rolling(window=6, min_periods=1).mean()
            means.columns = [f"{feature}_mean6" for feature in FEATURE_COLUMNS]
            group = pd.concat([group, means], axis=1)
        if "rolling_std_6" in modes:
            stds = values.rolling(window=6, min_periods=1).std(ddof=0)
            stds.columns = [f"{feature}_std6" for feature in FEATURE_COLUMNS]
            group = pd.concat([group, stds], axis=1)
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["station", "time"])


class StandardScaler:
    """[03] 对 9 个特征做标准化，并支持把预测结果还原到原始量纲。"""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        """[03-1] 用训练集计算每个特征的均值和标准差。"""
        flat = np.asarray(values, dtype=float).reshape(-1, values.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        self.scale_ = np.nanstd(flat, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """[03-2] 把原始数据转成标准化数据。"""
        self._check_fitted()
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """[03-3] 把标准化后的数据还原成原始单位。"""
        self._check_fitted()
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, list[float]]:
        """[03-4] 保存均值和标准差，方便复现实验。"""
        self._check_fitted()
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    def _check_fitted(self) -> None:
        """[03-5] 防止未 fit 就调用 transform。"""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted.")


def station_name(path: Path) -> str:
    """[04] 从文件名中提取站点名。"""
    return path.name.split("2015年")[0]


def clean_quantity_frame(raw: pd.DataFrame, start_date: str, drop_outliers: bool = True) -> pd.DataFrame:
    """[05] 清洗单个站点原始表：解析时间、转数值、剔除异常值。"""
    work = raw[pd.notna(raw["监测时间"])].copy()
    work = work[~work["监测时间"].astype(str).str.contains("标准", na=False)]
    work["time"] = pd.to_datetime(work["监测时间"].astype(str).str.strip(), format="%Y-%m-%d %H", errors="coerce")
    work = work[pd.notna(work["time"]) & (work["time"] >= pd.Timestamp(start_date))]

    for column in FEATURE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce") if column in work else np.nan
        if drop_outliers:
            work[column] = clean_feature_values(column, work[column])

    return work[["time", *FEATURE_COLUMNS]].sort_values("time")


def load_resampled_4h_quantity_data(
    data_dir: Path,
    start_date: str = "2020-01-01",
    rule: str = "4h",
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """[06] 读取所有站点 .xls，并统一重采样到 4 小时尺度，暂不插值。"""
    frames = []
    for path in sorted(data_dir.glob("*.xls")):
        try:
            raw = pd.read_excel(path, sheet_name=0)
        except ImportError as exc:
            raise SystemExit("读取 .xls 需要安装 xlrd：python -m pip install xlrd") from exc

        cleaned = clean_quantity_frame(raw, start_date, drop_outliers)
        by_time = cleaned.groupby("time")[list(FEATURE_COLUMNS)].mean()
        frame = by_time.resample(rule).mean().reset_index()
        frame["station"] = station_name(path)
        frames.append(frame[["station", "time", *FEATURE_COLUMNS]])

    if not frames:
        raise FileNotFoundError(f"No .xls files found in {data_dir}")
    data = pd.concat(frames, ignore_index=True).sort_values(["station", "time"])
    return data[["station", "time", *FEATURE_COLUMNS]]


def load_4h_quantity_data(
    data_dir: Path,
    start_date: str = "2020-01-01",
    rule: str = "4h",
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """[06-0] 读取、重采样并完成短缺口插值，返回可建模标准表。"""
    data = load_resampled_4h_quantity_data(data_dir, start_date, rule, drop_outliers)
    return interpolate_short_gaps(data, INTERPOLATION_LIMIT)


def save_processed_4h_data(data: pd.DataFrame, path: str | Path) -> None:
    """[06-1] 保存清洗后的 4 小时标准数据，后续训练直接读取它。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data[["station", "time", *FEATURE_COLUMNS]].copy()
    data["time"] = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    data.to_csv(path, index=False, encoding="utf-8-sig")


def load_processed_4h_data(path: str | Path) -> pd.DataFrame:
    """[06-2] 读取清洗后的 4 小时标准数据。"""
    path = Path(path)
    data = pd.read_csv(path)
    data["time"] = pd.to_datetime(data["time"])
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    quality_path = path.with_name("quantity_4h_quality.csv")
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        quality["time"] = pd.to_datetime(quality["time"])
        quality_columns = [column for column in quality if column not in {"station", "time"}]
        data = data.merge(quality[["station", "time", *quality_columns]], on=["station", "time"], how="left")
        for feature in FEATURE_COLUMNS:
            column = target_ok_column(feature)
            if column in data:
                data[column] = data[column].fillna(False).astype(bool)

    return data.sort_values(["station", "time"])


def filter_by_start_date(data: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """[06-2-1] 按实验起始日期截断数据。"""
    filtered = data[pd.to_datetime(data["time"]) >= pd.Timestamp(start_date)].copy()
    return filtered.sort_values(["station", "time"])


def load_or_build_4h_quantity_data(
    data_dir: Path,
    processed_path: Path,
    start_date: str = "2020-01-01",
    rule: str = "4h",
    drop_outliers: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """[06-3] 优先读取 processed 数据；需要时才从原始 .xls 重建。"""
    processed_path = Path(processed_path)
    if processed_path.exists() and not rebuild:
        return filter_by_start_date(load_processed_4h_data(processed_path), start_date)
    if "v2" in processed_path.parts:
        raise FileNotFoundError(
            f"Missing canonical V2 data: {processed_path}. Run scripts/data/build_processed_quantity_data_v2.py first."
        )
    data = load_4h_quantity_data(data_dir, start_date, rule, drop_outliers)
    save_processed_4h_data(data, processed_path)
    return filter_by_start_date(data, start_date)


def make_supervised_windows(
    values: np.ndarray,
    times: np.ndarray,
    input_steps: int = 4,
    output_steps: int = 4,
    target_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[07] 构造监督学习窗口：过去 input_steps 预测未来 output_steps。"""
    input_values = np.asarray(values, dtype=float)
    target_values = input_values if target_values is None else np.asarray(target_values, dtype=float)
    total_steps = input_steps + output_steps
    xs, ys, starts, ends = [], [], [], []

    for start in range(len(input_values) - total_steps + 1):
        x = input_values[start : start + input_steps]
        y = target_values[start + input_steps : start + total_steps]
        if np.isfinite(x).all() and np.isfinite(y).all():
            xs.append(x)
            ys.append(y)
            starts.append(times[start + input_steps])
            ends.append(times[start + total_steps - 1])

    input_feature_count = input_values.shape[1]
    target_feature_count = target_values.shape[1]
    if not xs:
        return (
            np.empty((0, input_steps, input_feature_count)),
            np.empty((0, output_steps, target_feature_count)),
            np.asarray([], dtype=np.asarray(times).dtype),
            np.asarray([], dtype=np.asarray(times).dtype),
        )
    return np.stack(xs), np.stack(ys), np.asarray(starts), np.asarray(ends)


def build_dataset(
    data: pd.DataFrame,
    input_steps: int = 4,
    output_steps: int = 4,
    input_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, np.ndarray]:
    """[08] 按站点分别构造窗口，再合并成 X/y 数据集。"""
    parts = {"x": [], "y": [], "target_start": [], "target_end": [], "station": []}

    for station, group in data.groupby("station", sort=True):
        group = group.sort_values("time")
        target_values = group.loc[:, target_columns].copy()
        for column in target_columns:
            quality_column = target_ok_column(column)
            if quality_column in group:
                target_values.loc[~group[quality_column].fillna(False).astype(bool), column] = np.nan
        x, y, target_start, target_end = make_supervised_windows(
            group.loc[:, input_columns].to_numpy(float),
            group["time"].to_numpy("datetime64[ns]"),
            input_steps,
            output_steps,
            target_values.to_numpy(float),
        )
        if len(x) == 0:
            continue
        parts["x"].append(x)
        parts["y"].append(y)
        parts["target_start"].append(target_start)
        parts["target_end"].append(target_end)
        parts["station"].append(np.full(len(x), station, dtype=object))

    if not parts["x"]:
        return {
            "x": np.empty((0, input_steps, len(input_columns))),
            "y": np.empty((0, output_steps, len(target_columns))),
            "target_start": np.asarray([], dtype="datetime64[ns]"),
            "target_end": np.asarray([], dtype="datetime64[ns]"),
            "station": np.asarray([], dtype=object),
        }

    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def split_by_time(dataset: dict[str, np.ndarray], train_end: str, val_end: str) -> dict[str, dict[str, np.ndarray]]:
    """[09] 按目标时间切分训练集、验证集和测试集。"""
    start = pd.to_datetime(dataset["target_start"])
    end = pd.to_datetime(dataset["target_end"])
    masks = {
        "train": end < pd.Timestamp(train_end),
        "val": (start >= pd.Timestamp(train_end)) & (end < pd.Timestamp(val_end)),
        "test": start >= pd.Timestamp(val_end),
    }
    return {
        name: {key: value[np.asarray(mask)] for key, value in dataset.items()}
        for name, mask in masks.items()
    }


def split_summary(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, object]]:
    """[10] 汇总每个数据集的样本数、shape、时间范围和站点数。"""
    summary = {}
    for name, split in splits.items():
        starts = split["target_start"]
        ends = split["target_end"]
        summary[name] = {
            "samples": int(len(split["x"])),
            "x_shape": list(split["x"].shape),
            "y_shape": list(split["y"].shape),
            "start": str(starts.min()) if len(starts) else "",
            "end": str(ends.max()) if len(ends) else "",
            "stations": int(len(set(split["station"].tolist()))) if len(starts) else 0,
        }
    return summary
