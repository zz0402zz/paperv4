from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import config
from scripts.multitarget_forecasting.output_mode_dynamics_report import (
    _spearman,
    normalized_four_hour_dynamics,
)


class OutputModeDynamicsTests(unittest.TestCase):
    def test_normalized_dynamics_uses_train_only_four_hour_change_and_iqr(self):
        rows = 5
        targets = len(config.TARGETS)
        current = np.tile(np.arange(rows, dtype=float)[:, None], (1, targets))
        y_delta = np.ones((rows, config.OUTPUT_STEPS, targets), dtype=float)
        split = {
            "current": current,
            "current_mask": np.ones_like(current, dtype=bool),
            "y_delta": y_delta,
            "y_mask": np.ones_like(y_delta, dtype=bool),
        }
        result = normalized_four_hour_dynamics(split, 0)
        self.assertAlmostEqual(result["value_iqr"], 2.0)
        self.assertAlmostEqual(result["median_absolute_4h_delta"], 1.0)
        self.assertAlmostEqual(result["normalized_4h_dynamics"], 0.5)

    def test_spearman_sign_matches_large_dynamics_favoring_absolute(self):
        dynamics = pd.Series([0.1, 0.2, 0.3, 0.4])
        delta_relative_absolute = pd.Series([-3.0, -1.0, 2.0, 5.0])
        self.assertAlmostEqual(_spearman(dynamics, delta_relative_absolute), 1.0)


if __name__ == "__main__":
    unittest.main()
