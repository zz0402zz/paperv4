from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.tabpfn_comparison import config, data, io, models


def test_protocol_matches_frozen_mainline() -> None:
    assert config.START_DATE == "2020-01-01"
    assert config.TRAIN_END == "2024-01-01"
    assert config.VAL_END == "2025-01-01"
    assert len(config.STATIONS) == 7
    assert len(config.TARGETS) == 5
    assert config.INPUT_STEPS == 6
    assert config.OUTPUT_STEPS == 18


def test_native_batch_never_uses_values_after_each_origin() -> None:
    series = pd.DataFrame(
        {
            "time": pd.date_range("2023-12-31", periods=8, freq="4h"),
            "target": np.arange(8, dtype=float),
        }
    )
    origins = np.asarray(["2023-12-31T12:00", "2023-12-31T20:00"], dtype="datetime64[ns]")
    context, future = data.native_origin_batch(
        series,
        origins,
        max_context_length=100,
    )
    assert context.groupby("item_id")["target"].max().to_dict() == {"0": 3.0, "1": 5.0}
    assert context.groupby("item_id").size().to_dict() == {"0": 4, "1": 6}
    assert future.groupby("item_id").size().eq(config.OUTPUT_STEPS).all()


def test_delta_features_use_train_medians_only() -> None:
    train = np.array([[1.0, np.nan], [3.0, 2.0]])
    val = np.array([[np.nan, np.inf]])
    medians = models.finite_feature_medians(train)
    assert np.allclose(medians, [2.0, 2.0])
    assert np.allclose(models.apply_feature_medians(val, medians), [[2.0, 2.0]])


def test_prediction_roundtrip_and_metadata_guard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    path = io.prediction_path("model", 17, "target", "station")
    arrays = {
        "pred": np.zeros((2, config.OUTPUT_STEPS)),
        "true": np.ones((2, config.OUTPUT_STEPS)),
        "mask": np.ones((2, config.OUTPUT_STEPS), dtype=bool),
        "current": np.zeros((2, 1)),
        "target_start": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
    }
    io.save_prediction(path, arrays, {"version": 1})
    loaded, metadata = io.load_prediction(path)
    assert metadata == {"version": 1}
    assert np.array_equal(loaded["pred"], arrays["pred"])
    assert io.is_complete(path, {"version": 1})
    assert not io.is_complete(path, {"version": 2})


def test_partial_prediction_resume_audits_prefix(tmp_path) -> None:
    base = {
        "pred": np.zeros((3, config.OUTPUT_STEPS)),
        "true": np.ones((3, config.OUTPUT_STEPS)),
        "mask": np.ones((3, config.OUTPUT_STEPS), dtype=bool),
        "current": np.zeros((3, 1)),
        "target_start": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            dtype="datetime64[ns]",
        ),
    }
    partial = tmp_path / "result.partial.npz"
    prefix = {key: value[:2] for key, value in base.items()}
    io.save_prediction(partial, prefix, {"protocol": 2})
    loaded = io.load_prediction_prefix(partial, {"protocol": 2}, base)
    assert loaded.shape == (2, config.OUTPUT_STEPS)

    wrong = {**base, "current": np.ones((3, 1))}
    with np.testing.assert_raises_regex(RuntimeError, "prefix mismatch"):
        io.load_prediction_prefix(partial, {"protocol": 2}, wrong)
