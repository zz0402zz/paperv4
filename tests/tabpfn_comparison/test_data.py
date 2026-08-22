from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import target_ok_column
from scripts.tabpfn_comparison import config, data, io, models


class ShortHistoryTabpfnDataTests(unittest.TestCase):
    def _panel(self, *, invalid_target_index: int | None = None) -> pd.DataFrame:
        rows = []
        times = pd.date_range("2023-12-30", periods=10, freq="4h")
        target = config.TARGETS[0]
        for station, offset in (("station-a", 0.0), ("station-b", 1000.0)):
            for index, timestamp in enumerate(times):
                row = {"station": station, "time": timestamp}
                for feature_index, feature in enumerate(config.INPUT_FEATURES):
                    row[feature] = offset + feature_index * 100.0 + index
                row[target_ok_column(target)] = not (
                    station == "station-a" and invalid_target_index == index
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def test_protocol_uses_current_v2_short_history(self) -> None:
        self.assertEqual(config.START_DATE, "2022-01-01")
        self.assertEqual(config.INPUT_STEPS, 6)
        self.assertEqual(config.OUTPUT_STEPS, 1)
        self.assertEqual(config.STEP_HOURS, 4)
        self.assertIn(config.DELTA_GRU_KEY, config.MODEL_KEYS)
        self.assertIn(config.DELTA_TABPFN_KEY, config.MODEL_KEYS)

    def test_station_windows_are_causal_and_do_not_read_other_stations(self) -> None:
        target = config.TARGETS[0]
        panel = self._panel()
        dataset = data.build_station_target_dataset(panel, "station-a", target)
        self.assertEqual(len(dataset["target_start"]), 4)
        target_index = config.INPUT_FEATURES.index(target)
        self.assertEqual(dataset["x_raw"][0, -1, target_index], 100.0 + 5.0)
        self.assertEqual(dataset["y_abs"][0, 0], 100.0 + 6.0)
        self.assertEqual(dataset["y_delta"][0, 0], 1.0)
        self.assertEqual(
            dataset["target_start"][0],
            np.datetime64("2023-12-31T00:00:00"),
        )
        self.assertLess(dataset["x_raw"].max(), 1000.0)

    def test_quality_sidecar_masks_current_or_future_target(self) -> None:
        target = config.TARGETS[0]
        dataset = data.build_station_target_dataset(
            self._panel(invalid_target_index=6), "station-a", target
        )
        self.assertFalse(dataset["y_mask"][0, 0])
        self.assertFalse(dataset["y_mask"][1, 0])
        self.assertTrue(dataset["y_mask"][2, 0])

    def test_tabpfn_features_keep_masks_and_current_state(self) -> None:
        dataset = data.build_station_target_dataset(self._panel(), "station-a", config.TARGETS[0])
        features = data.tabpfn_features(dataset)
        expected_columns = config.INPUT_STEPS * len(config.INPUT_FEATURES) * 4 + 2
        self.assertEqual(features.shape, (len(dataset["target_start"]), expected_columns))
        self.assertTrue(np.isfinite(features[:, -1]).all())

    def test_training_medians_only_fill_missing_values(self) -> None:
        train = np.array([[1.0, np.nan], [3.0, 2.0]])
        evaluation = np.array([[np.nan, np.inf]])
        medians = models.finite_feature_medians(train)
        self.assertTrue(np.allclose(medians, [2.0, 2.0]))
        self.assertTrue(np.allclose(models.apply_feature_medians(evaluation, medians), [[2.0, 2.0]]))

    def test_prediction_roundtrip_and_metadata_guard(self) -> None:
        arrays = {
            "pred": np.zeros((2, config.OUTPUT_STEPS)),
            "true": np.ones((2, config.OUTPUT_STEPS)),
            "mask": np.ones((2, config.OUTPUT_STEPS), dtype=bool),
            "current": np.zeros((2, 1)),
            "target_start": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(config, "OUTPUT_DIR", Path(temporary)):
                path = io.prediction_path(
                    "val", config.DELTA_TABPFN_KEY, 17, "target", "station"
                )
                io.save_prediction(path, arrays, {"version": 1})
                loaded, metadata = io.load_prediction(path)
                self.assertEqual(metadata, {"version": 1})
                self.assertTrue(np.array_equal(loaded["pred"], arrays["pred"]))
                self.assertTrue(io.is_complete(path, {"version": 1}))
                self.assertFalse(io.is_complete(path, {"version": 2}))
                self.assertEqual(path.parent.name, "预测结果")
                self.assertEqual(
                    path.name,
                    "变化量TabPFN-v2__种子17__station__target.npz",
                )

    def test_output_only_migration_accepts_exact_legacy_code_bundle(self) -> None:
        saved = {
            "experiment": config.EXPERIMENT_ID,
            "observed_data_path": r"data\processed\v2\quantity_4h_observed.csv",
            "quality_data_path": r"data\processed\v2\quantity_4h_quality.csv",
            "code_sha256": dict(io.LEGACY_PREDICTION_CODE_SHA256),
        }
        expected = {
            "experiment": config.EXPERIMENT_ID,
            "observed_data_path": "data/processed/v2/quantity_4h_observed.csv",
            "quality_data_path": "data/processed/v2/quantity_4h_quality.csv",
            "code_sha256": {"scripts/tabpfn_comparison/run.py": "current"},
        }
        self.assertTrue(io._metadata_matches(saved, expected))
        saved["code_sha256"]["scripts/tabpfn_comparison/run.py"] = "changed"
        self.assertFalse(io._metadata_matches(saved, expected))

    def test_safe_filename_preserves_distinct_chinese_targets(self) -> None:
        self.assertNotEqual(
            io.safe_filename("氨氮(mg/L)"),
            io.safe_filename("总磷(mg/L)"),
        )
        self.assertNotIn("/", io.safe_filename("站点/名称"))


if __name__ == "__main__":
    unittest.main()
