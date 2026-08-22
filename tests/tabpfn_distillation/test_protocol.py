from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import target_ok_column
from scripts.tabpfn_distillation import config, data, io, models
from scripts.tabpfn_distillation.inertia_gate import (
    apply_persistence_gate,
    fit_persistence_gate,
    gate_cache_path,
)
from scripts.tabpfn_distillation.distilled_xgboost import (
    build_distillation_targets,
    oof_cache_path as distilled_xgboost_oof_path,
    self_gate_path as distilled_xgboost_gate_path,
    validation_prediction_path as distilled_xgboost_validation_path,
)
from scripts.tabpfn_distillation.protocol_baselines import (
    BASELINE_KEYS,
    LSTM_KEY,
    XGBOOST_KEY,
    baseline_prediction_path,
)
from scripts.tabpfn_distillation.report import paired_comparison
from scripts.tabpfn_distillation.teacher import _fit_predict, _parse_horizons
from scripts.tabpfn_distillation.xgboost_gate_attribution import (
    _empty_oof_checkpoint,
    xgboost_oof_cache_path,
    xgboost_self_gate_path,
)


class LongHorizonProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        times = pd.date_range("2022-01-01", "2025-01-05", freq="4h")
        target = config.TARGETS[0]
        rows = []
        for index, timestamp in enumerate(times):
            row = {"station": "测试站", "time": timestamp}
            for feature_index, feature in enumerate(config.INPUT_FEATURES):
                row[feature] = float(index + feature_index * 10_000)
            row[target_ok_column(target)] = True
            rows.append(row)
        cls.panel = pd.DataFrame(rows)
        cls.target = target
        cls.dataset = data.build_station_target_dataset(cls.panel, "测试站", target)

    def test_protocol_has_eighteen_direct_horizons(self) -> None:
        self.assertEqual(config.INPUT_STEPS, 6)
        self.assertEqual(config.HORIZON_HOURS, tuple(range(4, 73, 4)))
        self.assertEqual(config.OUTPUT_STEPS, 18)
        self.assertEqual(self.dataset["y_abs"].shape[1], 18)
        self.assertTrue(np.allclose(self.dataset["y_delta"][0], np.arange(1.0, 19.0)))
        self.assertEqual(
            self.dataset["target_end"][0] - self.dataset["target_start"][0],
            np.timedelta64(68, "h"),
        )
        self.assertEqual(_parse_horizons("4,24,72"), (0, 5, 17))

    def test_split_uses_full_seventy_two_hour_extent(self) -> None:
        splits = data.split_by_time(self.dataset)
        self.assertLess(
            splits["train"]["target_end"].max(), np.datetime64(config.TRAIN_END)
        )
        self.assertGreaterEqual(
            splits["val"]["target_start"].min(), np.datetime64(config.TRAIN_END)
        )
        self.assertLess(splits["val"]["target_end"].max(), np.datetime64(config.VAL_END))
        self.assertGreaterEqual(
            splits["test"]["target_start"].min(), np.datetime64(config.VAL_END)
        )

    def test_every_oof_fold_is_strictly_forward_in_time(self) -> None:
        train = data.split_by_time(self.dataset)["train"]
        for fold in data.causal_oof_folds(train):
            fit_ends = train["target_end"][fold["fit_mask"]]
            prediction_starts = train["target_start"][fold["prediction_mask"]]
            self.assertGreater(len(fit_ends), 0)
            self.assertGreater(len(prediction_starts), 0)
            self.assertLess(fit_ends.max(), prediction_starts.min())

    def test_xgboost_oof_checkpoint_tracks_every_fold_and_horizon(self) -> None:
        rows = 12
        fold_count = len(config.OOF_FOLDS)
        times = np.arange(rows).astype("timedelta64[h]") + np.datetime64(
            "2022-01-01"
        )
        checkpoint = _empty_oof_checkpoint(rows, fold_count, times)
        self.assertEqual(checkpoint["pred_delta"].shape, (rows, 18))
        self.assertEqual(checkpoint["pred_mask"].shape, (rows, 18))
        self.assertEqual(checkpoint["completed"].shape, (fold_count, 18))
        self.assertFalse(checkpoint["completed"].any())
        self.assertTrue(np.isnan(checkpoint["pred_delta"]).all())

    def test_absolute_and_delta_representations_are_exactly_convertible(self) -> None:
        current = self.dataset["current"][:3]
        delta = self.dataset["y_delta"][:3]
        absolute = data.to_absolute(delta, current, "delta")
        self.assertTrue(np.array_equal(absolute, self.dataset["y_abs"][:3]))
        self.assertTrue(
            np.array_equal(
                data.target_values(self.dataset, "absolute"), self.dataset["y_abs"]
            )
        )

    def test_quality_mask_controls_each_horizon_label(self) -> None:
        panel = self.panel.copy()
        invalid_time = np.datetime64("2022-01-02T00:00:00")
        quality_column = target_ok_column(self.target)
        panel.loc[panel["time"].eq(invalid_time), quality_column] = False
        dataset = data.build_station_target_dataset(panel, "测试站", self.target)
        self.assertFalse(dataset["y_mask"][0, 0])
        self.assertFalse(dataset["y_mask"][1].any())

    def test_station_windows_never_read_another_station(self) -> None:
        other = self.panel.copy()
        other["station"] = "其他站"
        for feature in config.INPUT_FEATURES:
            other[feature] = other[feature] + 1_000_000.0
        mixed = pd.concat((self.panel, other), ignore_index=True)
        dataset = data.build_station_target_dataset(mixed, "测试站", self.target)
        self.assertLess(float(np.nanmax(dataset["x_raw"])), 1_000_000.0)

    def test_tabpfn_and_gru_share_the_same_local_input_information(self) -> None:
        features = data.tabpfn_features(self.dataset)
        expected_columns = config.INPUT_STEPS * len(config.INPUT_FEATURES) * 4 + 2
        self.assertEqual(features.shape[1], expected_columns)
        self.assertEqual(BASELINE_KEYS, (LSTM_KEY, XGBOOST_KEY))

    def test_validation_teacher_predicts_from_a_separate_feature_matrix(self) -> None:
        class FakeRegressor:
            def __init__(self) -> None:
                self.fit_x = None
                self.fit_y = None

            def fit(self, features, labels) -> None:
                self.fit_x = np.asarray(features)
                self.fit_y = np.asarray(labels)

            def predict(self, features):
                return np.asarray(features)[:, 0]

        regressor = FakeRegressor()
        train_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        train_y = np.array([10.0, 30.0, 50.0])
        fit_rows = np.array([True, False, True])
        validation_x = np.array([[7.0, 8.0], [9.0, 10.0]])
        prediction_rows = np.array([False, True])
        with (
            patch.object(models, "make_teacher", return_value=regressor),
            patch(
                "scripts.tabpfn_distillation.teacher._release_teacher",
                return_value=None,
            ),
        ):
            predicted = _fit_predict(
                train_x,
                train_y,
                fit_rows,
                validation_x,
                prediction_rows,
            )
        self.assertTrue(np.array_equal(regressor.fit_x, train_x[fit_rows]))
        self.assertTrue(np.array_equal(regressor.fit_y, train_y[fit_rows]))
        self.assertTrue(np.array_equal(predicted, np.array([9.0])))

    def test_masked_scaler_ignores_unapproved_values(self) -> None:
        values = np.array([[1.0, 100.0], [3.0, 200.0], [999.0, 300.0]])
        mask = np.array([[True, True], [True, False], [False, True]])
        scaler = models.MaskedScaler.fit(values, mask)
        self.assertTrue(np.allclose(scaler.mean, [2.0, 200.0]))
        self.assertTrue(np.allclose(scaler.scale, [1.0, 100.0]))

    def test_xgboost_distillation_target_exactly_combines_two_mse_terms(self) -> None:
        truth = np.array([[2.0, 8.0, 10.0]])
        teacher = np.array([[4.0, 6.0, 12.0]])
        truth_mask = np.array([[True, False, True]])
        teacher_mask = np.array([[True, True, False]])
        target, weight, true_valid, teacher_valid = build_distillation_targets(
            truth,
            truth_mask,
            teacher,
            teacher_mask,
            teacher_weight=0.5,
        )
        self.assertTrue(np.allclose(target, [[8.0 / 3.0, 6.0, 10.0]]))
        self.assertTrue(np.allclose(weight, [[1.5, 0.5, 1.0]]))
        self.assertTrue(np.array_equal(true_valid, truth_mask))
        self.assertTrue(np.array_equal(teacher_valid, teacher_mask))

    def test_persistence_gate_is_fitted_only_from_supplied_oof_cells(self) -> None:
        teacher_delta = np.column_stack(
            (
                np.array([2.0, 4.0, 1000.0]),
                np.array([1.0, -1.0, 1000.0]),
                *[np.ones(3) for _ in range(config.OUTPUT_STEPS - 2)],
            )
        )
        true_delta = np.column_stack(
            (
                np.array([1.0, 2.0, -1000.0]),
                np.array([-1.0, 1.0, -1000.0]),
                *[np.full(3, 3.0) for _ in range(config.OUTPUT_STEPS - 2)],
            )
        )
        mask = np.ones_like(teacher_delta, dtype=bool)
        mask[2] = False
        fitted = fit_persistence_gate(teacher_delta, true_delta, mask)
        self.assertAlmostEqual(float(fitted["alpha"][0]), 0.5)
        self.assertAlmostEqual(float(fitted["alpha"][1]), 0.0)
        self.assertTrue(np.allclose(fitted["alpha"][2:], 1.0))
        self.assertTrue(np.array_equal(fitted["valid_count"], np.full(18, 2)))

    def test_persistence_gate_has_exact_identity_and_persistence_limits(self) -> None:
        current = np.array([[10.0], [20.0]])
        prediction = np.vstack((np.arange(18.0), np.arange(18.0) + 30.0))
        identity = apply_persistence_gate(prediction, current, np.ones(18))
        persistence = apply_persistence_gate(prediction, current, np.zeros(18))
        self.assertTrue(np.array_equal(identity, prediction))
        self.assertTrue(np.array_equal(persistence, np.repeat(current, 18, axis=1)))

    def test_raw_delta_comparison_is_paired_by_task_seed_and_horizon(self) -> None:
        cells = pd.DataFrame(
            [
                {
                    "station": "测试站",
                    "target": self.target,
                    "seed": 42,
                    "horizon_hours": 4,
                    "variant": config.SUPERVISED_ABSOLUTE_KEY,
                    "rmse": 0.8,
                },
                {
                    "station": "测试站",
                    "target": self.target,
                    "seed": 42,
                    "horizon_hours": 4,
                    "variant": config.SUPERVISED_DELTA_KEY,
                    "rmse": 1.0,
                },
            ]
        )
        paired = paired_comparison(
            cells,
            config.SUPERVISED_ABSOLUTE_KEY,
            config.SUPERVISED_DELTA_KEY,
            "test",
        )
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(float(paired.loc[0, "relative_pct"]), -20.0)

    def test_chinese_output_paths_are_shallow_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(config, "OUTPUT_DIR", Path(temporary)):
                teacher_path = io.teacher_cache_path("训练OOF", "测试站", self.target)
                student_path = io.student_prediction_path(
                    "val", config.DISTILLED_DELTA_KEY, 42, "测试站", self.target
                )
                self.assertEqual(teacher_path.parent.name, "教师缓存")
                self.assertEqual(student_path.parent.name, "预测结果")
                self.assertIn("变化量因果蒸馏GRU", student_path.name)
                self.assertNotIn("/", io.safe_filename("氨氮(mg/L)"))
                gate_path = gate_cache_path("测试站", self.target)
                self.assertEqual(gate_path.parent.name, "门控参数")
                self.assertIn("OOF惯性门控", gate_path.name)
                lstm_path = baseline_prediction_path(
                    LSTM_KEY, 42, "测试站", self.target
                )
                xgboost_path = baseline_prediction_path(
                    XGBOOST_KEY, 42, "测试站", self.target
                )
                self.assertEqual(lstm_path.parent.name, "预测结果")
                self.assertEqual(lstm_path.parent.parent.name, "同协议基线")
                self.assertIn("变化量LSTM", lstm_path.name)
                self.assertIn("变化量XGBoost", xgboost_path.name)
                xgboost_oof_path = xgboost_oof_cache_path(
                    42, "测试站", self.target
                )
                xgboost_gate_path = xgboost_self_gate_path(
                    42, "测试站", self.target
                )
                self.assertEqual(xgboost_oof_path.parent.name, "XGBoost训练OOF")
                self.assertEqual(
                    xgboost_gate_path.parent.name, "XGBoost自门控参数"
                )
                self.assertIn("种子42", xgboost_oof_path.name)
                self.assertIn("XGBoost自OOF惯性门控", xgboost_gate_path.name)
                distilled_validation_path = distilled_xgboost_validation_path(
                    42, "测试站", self.target
                )
                distilled_oof_path = distilled_xgboost_oof_path(
                    42, "测试站", self.target
                )
                distilled_gate_path = distilled_xgboost_gate_path(
                    42, "测试站", self.target
                )
                self.assertEqual(
                    distilled_validation_path.parent.parent.name,
                    "蒸馏XGBoost实验",
                )
                self.assertEqual(distilled_oof_path.parent.name, "训练OOF")
                self.assertEqual(distilled_gate_path.parent.name, "自门控参数")
                self.assertIn("因果蒸馏XGBoost", distilled_oof_path.name)

                arrays = {
                    "pred_delta": np.zeros((2, config.OUTPUT_STEPS)),
                    "completed": np.ones((1, config.OUTPUT_STEPS), dtype=bool),
                }
                metadata = {"protocol": 1}
                io.save_archive(teacher_path, arrays, metadata)
                loaded = io.load_exact(teacher_path, metadata)
                self.assertIsNotNone(loaded)
                self.assertTrue(np.array_equal(loaded["pred_delta"], arrays["pred_delta"]))
                with self.assertRaises(RuntimeError):
                    io.load_exact(teacher_path, {"protocol": 2})


if __name__ == "__main__":
    unittest.main()
