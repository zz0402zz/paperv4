from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.common.wq_gru_data import target_ok_column
from scripts.multitarget_forecasting import config, data
from scripts.multitarget_forecasting.io import _scientifically_compatible
from scripts.multitarget_forecasting.model import (
    balanced_masked_mse,
    balanced_masked_mse_numpy,
    build_model,
)
from scripts.multitarget_forecasting.report import event_flags, warning_metrics
from scripts.multitarget_forecasting import report as joint_report


class JointFiveTargetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        times = pd.date_range("2022-01-01", "2025-01-05", freq="4h")
        rows = []
        for index, timestamp in enumerate(times):
            row = {"station": "测试站", "time": timestamp}
            for feature_index, feature in enumerate(config.INPUT_FEATURES):
                row[feature] = float(index + feature_index * 10_000)
            for target in config.TARGETS:
                row[target_ok_column(target)] = True
            rows.append(row)
        cls.panel = pd.DataFrame(rows)
        cls.dataset = data.build_station_dataset(cls.panel, "测试站")

    def test_one_dataset_contains_all_five_targets_and_eighteen_horizons(self):
        self.assertEqual(self.dataset["y_abs"].shape[1:], (18, 5))
        self.assertEqual(self.dataset["y_delta"].shape[1:], (18, 5))
        self.assertTrue(
            np.array_equal(
                self.dataset["y_delta"][0],
                np.repeat(np.arange(1.0, 19.0)[:, None], 5, axis=1),
            )
        )

    def test_one_forward_pass_outputs_eighteen_by_five(self):
        import torch

        network = build_model(torch, sequence_dim=36, context_dim=10)
        predicted = network(torch.zeros(3, 6, 36), torch.zeros(3, 10))
        self.assertEqual(tuple(predicted.shape), (3, 18, 5))

    def test_absolute_and_delta_joint_models_have_distinct_paths(self):
        absolute = config.prediction_path("测试站", "24h", "absolute", 42)
        delta = config.prediction_path("测试站", "24h", "delta", 42)
        self.assertNotEqual(absolute, delta)
        self.assertIn("原值输出", absolute.name)
        self.assertIn("变化量输出", delta.name)

    def test_resume_ignores_only_legacy_cli_hashes(self):
        expected = {
            "experiment": "test",
            "code_sha256": {
                "scripts/multitarget_forecasting/config.py": "a",
                "scripts/multitarget_forecasting/data.py": "b",
                "scripts/multitarget_forecasting/model.py": "c",
            },
        }
        legacy = {
            "experiment": "test",
            "code_sha256": {
                **expected["code_sha256"],
                "scripts/multitarget_forecasting/run.py": "old-print-only-hash",
            },
        }
        self.assertTrue(_scientifically_compatible(legacy, expected))
        changed_model = {
            **legacy,
            "code_sha256": {**legacy["code_sha256"], "model.py": "changed"},
        }
        self.assertFalse(_scientifically_compatible(changed_model, expected))

    def test_masked_loss_balances_available_outputs(self):
        import torch

        predicted = torch.tensor([[[1.0, 2.0]], [[3.0, 100.0]]])
        target = torch.zeros_like(predicted)
        mask = torch.tensor([[[True, True]], [[True, False]]])
        # First output MSE=(1+9)/2=5; second output MSE=4; equal output mean=4.5.
        loss = balanced_masked_mse(torch, predicted, target, mask)
        self.assertAlmostEqual(float(loss), 4.5)
        numpy_loss = balanced_masked_mse_numpy(
            predicted.numpy(), target.numpy(), mask.numpy()
        )
        self.assertAlmostEqual(numpy_loss, 4.5)

    def test_all_contexts_use_exactly_paired_rows(self):
        splits = data.split_by_time(self.dataset)
        for context, steps in config.CONTEXT_STEPS.items():
            train_sequence, _, val_sequence, _, _ = data.prepare_inputs(
                splits["train"], splits["val"], context
            )
            self.assertEqual(train_sequence.shape[1], steps)
            self.assertEqual(val_sequence.shape[0], len(splits["val"]["target_start"]))

    def test_multiscale_features_include_daily_and_annual_information(self):
        names = data.multiscale_feature_names()
        self.assertIn("日周期正弦", names)
        self.assertIn("年周期余弦", names)
        self.assertTrue(any("滞后8760小时" in name for name in names))
        self.assertTrue(any("过去720小时" in name for name in names))
        self.assertEqual(self.dataset["x_multiscale"].shape[1], len(names))

    def test_time_splits_keep_full_label_extent_inside_each_block(self):
        splits = data.split_by_time(self.dataset)
        self.assertLess(splits["train"]["target_end"].max(), np.datetime64("2024-01-01"))
        self.assertGreaterEqual(
            splits["val"]["target_start"].min(), np.datetime64("2024-01-01")
        )
        self.assertLess(splits["val"]["target_end"].max(), np.datetime64("2025-01-01"))

    def test_epoch_selection_split_is_strictly_forward_inside_training(self):
        train = data.split_by_time(self.dataset)["train"]
        fit, internal_validation = data.internal_time_split(train)
        self.assertGreater(len(fit["target_end"]), 0)
        self.assertGreater(len(internal_validation["target_start"]), 0)
        self.assertLess(
            fit["target_end"].max(), internal_validation["target_start"].min()
        )
        self.assertGreaterEqual(
            internal_validation["target_start"].min(),
            np.datetime64(config.INTERNAL_VAL_START),
        )

    def test_warning_threshold_directions_are_target_specific(self):
        train = data.split_by_time(self.dataset)["train"]
        lower, upper = data.warning_thresholds(train)
        ph = config.TARGETS.index("pH(无量纲)")
        oxygen = config.TARGETS.index("溶解氧(mg/L)")
        phosphorus = config.TARGETS.index("总磷(mg/L)")
        self.assertTrue(np.isfinite(lower[ph]) and np.isfinite(upper[ph]))
        self.assertTrue(np.isfinite(lower[oxygen]) and np.isnan(upper[oxygen]))
        self.assertTrue(np.isnan(lower[phosphorus]) and np.isfinite(upper[phosphorus]))

    def test_warning_metrics_are_computed_from_forecasted_events(self):
        observed = event_flags(np.array([0.0, 1.0, 2.0]), np.nan, 1.5)
        predicted = event_flags(np.array([0.0, 2.0, 3.0]), np.nan, 1.5)
        metrics = warning_metrics(observed, predicted)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_early_stopping_report_writes_curve_and_selected_epoch(self):
        common = {
            "station": "测试站",
            "context": "24h",
            "context_label": config.CONTEXT_LABELS["24h"],
            "seed": 42,
            "target": config.TARGETS[0],
            "horizon_hours": 4,
        }
        forecast = pd.DataFrame(
            [
                {
                    **common,
                    "target_mode": mode,
                    "target_mode_label": config.TARGET_MODE_LABELS[mode],
                    "rmse": rmse,
                    "relative_rmse_pct": 1.0,
                    "beats_persistence": False,
                    "nse": 0.0,
                }
                for mode, rmse in (("absolute", 1.0), ("delta", 0.9))
            ]
        )
        warnings = pd.DataFrame(
            [
                {
                    **common,
                    "target_mode": "absolute",
                    "target_mode_label": "原值",
                    "model": model,
                    "recall": 0.5,
                    "f1": 0.4,
                    "false_alarm_rate": 0.1,
                }
                for model in ("joint_gru", "persistence")
            ]
        )
        runtime = pd.DataFrame(
            [
                {
                    "context_label": config.CONTEXT_LABELS["24h"],
                    "target_mode_label": "原值",
                    "selected_epoch": 5,
                    "best_internal_val_loss": 0.3,
                    "selection_training_seconds": 1.0,
                    "refit_training_seconds": 0.5,
                    "training_seconds": 1.5,
                    "inference_seconds": 0.1,
                    "parameter_count": 100,
                }
            ]
        )
        curves = pd.DataFrame(
            [{"epoch": 5, "train_loss": 0.4, "internal_validation_loss": 0.3}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(config, "VALIDATION_DIR", Path(temporary)):
                path = joint_report.write_report(
                    forecast, warnings, runtime, curves
                )
                self.assertTrue(path.exists())
                self.assertTrue((Path(temporary) / "训练期内部验证曲线.csv").exists())
                self.assertIn("选择轮数", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
