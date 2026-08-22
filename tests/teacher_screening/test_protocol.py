from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.teacher_screening import config, data
from scripts.multitarget_forecasting.preprocessing_ablation_data import (
    RobustScaler,
    TargetTransform,
)
from scripts.teacher_screening.data import PreparedFold
from scripts.teacher_screening.models import (
    _patch_transformer_predict,
    catboost_parameters,
)
from scripts.teacher_screening.report import (
    aggregate_cells,
    horizon_weights,
    winner_change_diagnostics,
)


class TeacherScreeningProtocolTests(unittest.TestCase):
    def test_catboost_gpu_protocol_forces_plain_boosting(self) -> None:
        parameters = catboost_parameters(42, "cuda")
        self.assertEqual(parameters["loss_function"], "MultiRMSE")
        self.assertEqual(parameters["boosting_type"], "Plain")
        self.assertEqual(parameters["task_type"], "GPU")

    def test_horizon_groups_cover_all_eighteen_steps(self) -> None:
        labels = [config.horizon_group(hour) for hour in config.ALL_HOURS]
        self.assertEqual(labels.count("短时距_4至24小时"), 6)
        self.assertEqual(labels.count("中时距_28至48小时"), 6)
        self.assertEqual(labels.count("长时距_52至72小时"), 6)

    def test_causal_folds_leave_full_label_gap(self) -> None:
        starts = pd.date_range("2022-01-01 04:00", "2023-12-31", freq="4h")
        train = {
            "target_start": starts.to_numpy(dtype="datetime64[ns]"),
            "target_end": (starts + pd.Timedelta(hours=68)).to_numpy(
                dtype="datetime64[ns]"
            ),
        }
        folds = data.causal_oof_folds(train)
        for fold in folds:
            fit_end = pd.to_datetime(train["target_end"])[fold["fit_mask"]]
            prediction_start = pd.to_datetime(train["target_start"])[
                fold["prediction_mask"]
            ]
            self.assertTrue(len(fit_end))
            self.assertTrue(len(prediction_start))
            self.assertLess(fit_end.max(), prediction_start.min())

    def test_horizon_specific_winner_is_preserved(self) -> None:
        rows = []
        for station in ("甲", "乙"):
            for model, short_ratio, long_ratio in (
                ("tabpfn", 0.80, 0.95),
                ("xgboost", 0.90, 0.75),
            ):
                for horizon, ratio in ((4, short_ratio), (72, long_ratio)):
                    rows.append(
                        {
                            "station": station,
                            "seed": 42,
                            "model": model,
                            "model_label": config.MODEL_LABELS[model],
                            "target": config.TARGETS[0],
                            "horizon_hours": horizon,
                            "horizon_group": config.horizon_group(horizon),
                            "rmse": ratio,
                            "nse": 0.5,
                            "warning_tp": 4,
                            "warning_fp": 1,
                            "warning_fn": 2,
                            "warning_tn": 20,
                            "persistence_warning_f1": 0.5,
                            "rmse_ratio_to_persistence": ratio,
                            "log_rmse_ratio_to_persistence": np.log(ratio),
                        }
                    )
        cells = aggregate_cells(pd.DataFrame(rows))
        diagnostic = winner_change_diagnostics(cells).iloc[0]
        self.assertTrue(bool(diagnostic["winner_changes_with_horizon"]))
        self.assertIn("4h=TabPFN", diagnostic["winner_by_horizon"])
        self.assertIn("72h=XGBoost逐输出", diagnostic["winner_by_horizon"])

    def test_reliability_zero_when_teacher_loses_to_persistence(self) -> None:
        cells = pd.DataFrame(
            {
                "model": ["tabpfn", "xgboost"],
                "model_label": ["TabPFN", "XGBoost逐输出"],
                "target": [config.TARGETS[0], config.TARGETS[0]],
                "horizon_hours": [4, 4],
                "horizon_group": [config.horizon_group(4)] * 2,
                "mean_log_rmse_ratio": np.log([1.1, 1.2]),
                "mean_nse": [0.0, -0.1],
                "station_seed_cells": [2, 2],
                "station_win_rate_vs_persistence": [0.0, 0.0],
                "geometric_rmse_ratio_to_persistence": [1.1, 1.2],
                "relative_rmse_to_persistence_pct": [10.0, 20.0],
                "rank_within_target_horizon": [1.0, 2.0],
            }
        )
        weighted = horizon_weights(cells)
        self.assertAlmostEqual(float(weighted["teacher_weight"].sum()), 1.0)
        self.assertTrue((weighted["teacher_reliability"] == 0.0).all())
        self.assertFalse(weighted["eligible_for_distillation"].any())

    def test_patch_teacher_smoke_shape(self) -> None:
        generator = np.random.default_rng(42)
        fit_rows = 300
        prediction_rows = 11
        horizons = 2
        targets = len(config.TARGETS)
        fit_target = generator.normal(size=(fit_rows, horizons, targets)).astype(
            np.float32
        )
        fold = PreparedFold(
            fit_flat=generator.normal(size=(fit_rows, 130)).astype(np.float32),
            prediction_flat=generator.normal(size=(prediction_rows, 130)).astype(
                np.float32
            ),
            fit_sequence=generator.normal(size=(fit_rows, 6, 20)).astype(np.float32),
            prediction_sequence=generator.normal(
                size=(prediction_rows, 6, 20)
            ).astype(np.float32),
            fit_context=generator.normal(size=(fit_rows, 10)).astype(np.float32),
            prediction_context=generator.normal(size=(prediction_rows, 10)).astype(
                np.float32
            ),
            fit_scaled_target=fit_target,
            fit_target_mask=np.ones_like(fit_target, dtype=bool),
            target_transform=TargetTransform(
                log_scales=np.ones(targets), log_enabled=False
            ),
            target_scaler=RobustScaler(
                center=np.zeros((horizons, targets)),
                scale=np.ones((horizons, targets)),
            ),
            prediction_current=np.ones((prediction_rows, targets)),
        )
        overrides = (
            mock.patch.object(config, "PATCH_DIMENSION", 16),
            mock.patch.object(config, "PATCH_LAYERS", 1),
            mock.patch.object(config, "PATCH_HEADS", 4),
            mock.patch.object(config, "PATCH_MAX_EPOCHS", 1),
            mock.patch.object(config, "PATCH_EVALUATION_EVERY", 1),
            mock.patch.object(config, "PATCH_PATIENCE", 1),
            mock.patch.object(config, "PATCH_MIN_EPOCHS", 1),
        )
        with overrides[0], overrides[1], overrides[2], overrides[3], overrides[4], overrides[5], overrides[6]:
            prediction, diagnostics = _patch_transformer_predict(fold, 42, "cpu")
        self.assertEqual(prediction.shape, (prediction_rows, horizons, targets))
        self.assertEqual(int(diagnostics["fitted_models"]), 1)


if __name__ == "__main__":
    unittest.main()
