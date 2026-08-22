from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import preprocessing_component_config as config
from scripts.multitarget_forecasting.preprocessing_component_model import (
    _fit_target_scaler,
    _loss_numpy,
)
from scripts.multitarget_forecasting.preprocessing_component_report import (
    build_contrasts,
)


class PreprocessingComponentTests(unittest.TestCase):
    def test_only_missing_b_and_c_are_trainable(self):
        self.assertEqual(
            config.TRAIN_VARIANTS,
            ("robust_mse", "standard_huber"),
        )
        self.assertEqual(config.TRAIN_VARIANT_SPECS["robust_mse"]["loss"], "mse")
        self.assertEqual(
            config.TRAIN_VARIANT_SPECS["standard_huber"]["input_scaler"],
            "mean_std",
        )

    def test_standard_and_robust_scalers_are_distinct(self):
        values = np.asarray([[[0.0]], [[1.0]], [[2.0]], [[1000.0]]])
        mask = np.ones_like(values, dtype=bool)
        standard = _fit_target_scaler(values, mask, "mean_std")
        robust = _fit_target_scaler(values, mask, "median_iqr")
        self.assertGreater(float(standard.scale[0, 0]), float(robust.scale[0, 0]))

    def test_mse_and_huber_losses_are_distinct(self):
        prediction = np.asarray([[[0.0]], [[100.0]]])
        target = np.zeros_like(prediction)
        mask = np.ones_like(prediction, dtype=bool)
        mse = _loss_numpy(prediction, target, mask, "mse")
        huber = _loss_numpy(prediction, target, mask, "huber")
        self.assertLess(huber, mse)

    def test_report_contains_each_predeclared_strict_contrast(self):
        rmse = {
            "original": 10.0,
            "robust_mse": 9.0,
            "standard_huber": 8.0,
            "robust_huber": 7.0,
            "robust_huber_log": 6.0,
        }
        frame = pd.DataFrame(
            [
                {
                    "station": "测试站",
                    "seed": 42,
                    "target": "pH(无量纲)",
                    "horizon_hours": 4,
                    "variant": variant,
                    "rmse": value,
                }
                for variant, value in rmse.items()
            ]
        )
        contrasts = build_contrasts(frame)
        self.assertEqual(set(contrasts["比较"]), {item[0] for item in config.CONTRASTS})
        log_increment = contrasts.loc[contrasts["比较"].eq("E对D_对数变换增量")]
        self.assertAlmostEqual(
            float(log_increment.iloc[0]["候选相对参照RMSE变化百分比"]),
            -100.0 / 7.0,
        )


if __name__ == "__main__":
    unittest.main()
