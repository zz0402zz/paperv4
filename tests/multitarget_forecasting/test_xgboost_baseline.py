from __future__ import annotations

import unittest

import numpy as np

from scripts.multitarget_forecasting import config
from scripts.multitarget_forecasting import head_ablation_config
from scripts.multitarget_forecasting.xgboost_baseline import (
    target_values,
    to_absolute_prediction,
)
from scripts.multitarget_forecasting.xgboost_report import mixed_prediction


class SameSampleXGBoostTests(unittest.TestCase):
    def setUp(self) -> None:
        shape = (2, config.OUTPUT_STEPS, len(config.TARGETS))
        self.absolute = np.arange(np.prod(shape), dtype=float).reshape(shape)
        self.delta = self.absolute + 1_000.0
        self.mask = np.ones(shape, dtype=bool)

    def test_target_modes_use_same_joint_label_shape_and_mask(self):
        split = {
            "y_abs": self.absolute,
            "y_delta": self.delta,
            "y_mask": self.mask,
        }
        absolute, absolute_mask = target_values(split, "absolute")
        delta, delta_mask = target_values(split, "delta")
        np.testing.assert_array_equal(absolute, self.absolute)
        np.testing.assert_array_equal(delta, self.delta)
        np.testing.assert_array_equal(absolute_mask, delta_mask)

    def test_delta_predictions_are_reconstructed_to_absolute(self):
        current = np.full((2, len(config.TARGETS)), 10.0)
        reconstructed = to_absolute_prediction(self.delta, current, "delta")
        np.testing.assert_array_equal(
            reconstructed, self.delta + current[:, None, :]
        )
        np.testing.assert_array_equal(
            to_absolute_prediction(self.absolute, current, "absolute"),
            self.absolute,
        )

    def test_mixed_prediction_uses_frozen_target_mapping(self):
        mixed = mixed_prediction(self.absolute, self.delta)
        for target_index, target in enumerate(config.TARGETS):
            source = (
                self.delta
                if head_ablation_config.TARGET_OUTPUT_MODES[target] == "delta"
                else self.absolute
            )
            np.testing.assert_array_equal(
                mixed[:, :, target_index], source[:, :, target_index]
            )


if __name__ == "__main__":
    unittest.main()
