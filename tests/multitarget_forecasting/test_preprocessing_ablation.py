from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import preprocessing_ablation_config as config
from scripts.multitarget_forecasting.preprocessing_ablation_data import (
    RobustScaler,
    TargetTransform,
    _elapsed_missing,
)
from scripts.multitarget_forecasting.preprocessing_ablation_model import (
    balanced_masked_huber_numpy,
)
from scripts.multitarget_forecasting.preprocessing_ablation_report import (
    _paired_summary,
)


class PreprocessingAblationTests(unittest.TestCase):
    def test_robust_scaler_uses_median_and_iqr(self):
        values = np.asarray([[0.0], [1.0], [2.0], [1000.0]])
        scaler = RobustScaler.fit(values)
        self.assertAlmostEqual(float(scaler.center[0]), 1.5)
        self.assertLess(float(scaler.scale[0]), 300.0)

    def test_log_mixed_target_round_trip(self):
        shape = (3, config.OUTPUT_STEPS, len(config.TARGETS))
        absolute = np.linspace(0.01, 10.0, np.prod(shape)).reshape(shape)
        current = np.linspace(0.02, 2.0, 3 * len(config.TARGETS)).reshape(3, -1)
        split = {
            "current": current,
            "current_mask": np.ones_like(current, dtype=bool),
            "y_abs": absolute,
            "y_delta": absolute - current[:, None, :],
            "y_mask": np.ones(shape, dtype=bool),
            "quality_y_mask": np.ones(shape, dtype=bool),
        }
        transform = TargetTransform.fit(split, enabled=True)
        mixed, mask = transform.training_values(split, quality_aware=False)
        reconstructed = transform.to_absolute(mixed, current)
        np.testing.assert_allclose(reconstructed, absolute, rtol=1e-10, atol=1e-10)
        self.assertTrue(mask.all())

    def test_quality_mask_is_used_only_when_requested(self):
        shape = (1, config.OUTPUT_STEPS, len(config.TARGETS))
        split = {
            "current": np.ones((1, len(config.TARGETS))),
            "current_mask": np.ones((1, len(config.TARGETS)), dtype=bool),
            "y_abs": np.ones(shape),
            "y_delta": np.zeros(shape),
            "y_mask": np.ones(shape, dtype=bool),
            "quality_y_mask": np.ones(shape, dtype=bool),
        }
        split["quality_y_mask"][0, 0, 0] = False
        transform = TargetTransform.fit(split, enabled=False)
        _, normal = transform.training_values(split, quality_aware=False)
        _, quality = transform.training_values(split, quality_aware=True)
        self.assertTrue(normal[0, 0, 0])
        self.assertFalse(quality[0, 0, 0])

    def test_elapsed_missing_distinguishes_short_and_long_gaps(self):
        mask = np.asarray([[[True], [False], [False], [True], [False]]])
        elapsed = _elapsed_missing(mask) * 42.0
        np.testing.assert_array_equal(elapsed.ravel(), [0.0, 1.0, 2.0, 0.0, 1.0])

    def test_huber_limits_single_extreme_error(self):
        prediction = np.asarray([[[0.0]], [[100.0]]])
        target = np.zeros_like(prediction)
        mask = np.ones_like(prediction, dtype=bool)
        loss = balanced_masked_huber_numpy(prediction, target, mask, delta=1.0)
        self.assertAlmostEqual(loss, 49.75)
        self.assertLess(loss, float(np.mean(np.square(prediction))))

    def test_multiseed_summary_uses_symmetric_geometric_rmse_change(self):
        frame = pd.DataFrame(
            {
                "variant": ["candidate", "candidate"],
                "candidate_relative_rmse_pct": [100.0, -50.0],
                "candidate_log_baseline_rmse_ratio": [
                    np.log(2.0),
                    np.log(0.5),
                ],
                "candidate_wins": [False, True],
                "candidate_rmse": [2.0, 0.5],
            }
        )
        summary = _paired_summary(frame, ["variant"])
        self.assertAlmostEqual(
            float(summary.iloc[0]["相对当前基线RMSE几何变化百分比"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
