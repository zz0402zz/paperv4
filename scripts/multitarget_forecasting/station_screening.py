#!/usr/bin/env python3
"""Select representative stations from training coverage and dynamics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config, data


def audit_station(panel: pd.DataFrame, station: str) -> dict[str, object]:
    train = data.split_by_time(data.build_station_dataset(panel, station))["train"]
    rows: dict[str, object] = {
        "station": station,
        "train_origins": int(len(train["target_start"])),
    }
    coverage_values = []
    dynamics_values = []
    for target_index, target in enumerate(config.TARGETS):
        label_mask = np.asarray(train["y_mask"], dtype=bool)[:, :, target_index]
        label_values = np.asarray(train["y_delta"], dtype=float)[:, :, target_index]
        valid = label_mask & np.isfinite(label_values)
        coverage = float(valid.mean()) if valid.size else 0.0
        coverage_values.append(coverage)

        current = np.asarray(train["current"], dtype=float)[:, target_index]
        current_mask = (
            np.asarray(train["current_mask"], dtype=bool)[:, target_index]
            & np.isfinite(current)
        )
        approved_current = current[current_mask]
        iqr = (
            float(np.quantile(approved_current, 0.75) - np.quantile(approved_current, 0.25))
            if len(approved_current)
            else np.nan
        )
        valid_delta = valid[:, 0]
        delta = np.abs(label_values[valid_delta, 0])
        normalized_dynamics = (
            float(np.median(delta) / iqr)
            if len(delta) and np.isfinite(iqr) and iqr > 0
            else np.nan
        )
        dynamics_values.append(normalized_dynamics)
        rows[f"{target}_训练标签有效率"] = coverage
        rows[f"{target}_归一化4小时变化"] = normalized_dynamics

    finite_dynamics = [value for value in dynamics_values if np.isfinite(value)]
    rows["五指标平均有效率"] = float(np.mean(coverage_values))
    rows["五指标最低有效率"] = float(np.min(coverage_values))
    rows["动力变化得分"] = (
        float(np.median(finite_dynamics)) if finite_dynamics else np.nan
    )
    return rows


def recommend(
    audit: pd.DataFrame, *, anchor: str, count: int, minimum_coverage: float
) -> pd.DataFrame:
    if count < 2:
        raise ValueError("跨站筛选至少需要两个站点。")
    if anchor not in set(audit["station"]):
        raise ValueError(f"锚点站不在数据中: {anchor}")
    eligible = audit.loc[
        audit["五指标最低有效率"].ge(minimum_coverage)
        & np.isfinite(audit["动力变化得分"])
        & audit["station"].ne(anchor)
    ].copy()
    if len(eligible) < count - 1:
        raise ValueError("满足有效率要求的候选站点不足。")

    quantiles = np.linspace(0.1, 0.9, count - 1)
    selected = []
    remaining = eligible.copy()
    for quantile in quantiles:
        target_score = float(eligible["动力变化得分"].quantile(quantile))
        candidate_index = (
            remaining["动力变化得分"].sub(target_score).abs().sort_values().index[0]
        )
        selected.append(remaining.loc[candidate_index])
        remaining = remaining.drop(index=candidate_index)
    result = pd.DataFrame([audit.loc[audit["station"].eq(anchor)].iloc[0], *selected])
    result.insert(0, "selection_order", np.arange(1, len(result) + 1))
    result.insert(
        2,
        "selection_reason",
        ["既有锚点站", *[f"训练动力分位点{q:.2f}" for q in quantiles]],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="仅用训练期覆盖率和动力变化筛选代表站点")
    parser.add_argument("--anchor", default="上仙屋")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    args = parser.parse_args()

    panel = data.load_development_panel()
    rows = [
        audit_station(panel, station) for station in data.available_stations(panel)
    ]
    audit = pd.DataFrame(rows).sort_values("动力变化得分").reset_index(drop=True)
    selected = recommend(
        audit,
        anchor=args.anchor,
        count=args.count,
        minimum_coverage=args.minimum_coverage,
    )
    output = config.OUTPUT_DIR / "站点筛选"
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "训练期站点覆盖与动力审计.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output / "推荐跨站验证站点.csv", index=False, encoding="utf-8-sig")
    (output / "推荐跨站验证站点.json").write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "count": args.count,
                "minimum_coverage": args.minimum_coverage,
                "selection_uses_model_results": False,
                "selection_uses_2024_labels": False,
                "stations": selected["station"].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selected[["selection_order", "station", "selection_reason", "五指标最低有效率", "动力变化得分"]].to_string(index=False))
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
