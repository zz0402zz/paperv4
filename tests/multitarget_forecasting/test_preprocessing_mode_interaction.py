from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.multitarget_forecasting import preprocessing_mode_interaction_config as config
from scripts.multitarget_forecasting.preprocessing_mode_interaction_model import (
    ModeTargetTransform,
)
from scripts.multitarget_forecasting.preprocessing_mode_interaction_report import (
    _station_bootstrap_geometric_percent_interval,
    build_mode_detail,
)


class PreprocessingModeInteractionTests(unittest.TestCase):
    def test_each_candidate_flips_exactly_one_target(self):
        self.assertEqual(set(config.FLIP_TARGETS.values()), set(config.TARGETS))
        for flip, target in config.FLIP_TARGETS.items():
            changed = {
                name
                for name in config.TARGETS
                if config.flipped_modes(flip)[name] != config.BASE_MODES[name]
            }
            self.assertEqual(changed, {target})

    def test_log_mixed_target_round_trip_for_every_flip(self):
        rows = 3
        shape = (rows, config.OUTPUT_STEPS, len(config.TARGETS))
        absolute = np.linspace(0.01, 10.0, np.prod(shape)).reshape(shape)
        current = np.linspace(
            0.02, 2.0, rows * len(config.TARGETS)
        ).reshape(rows, -1)
        split = {
            "current": current,
            "current_mask": np.ones_like(current, dtype=bool),
            "y_abs": absolute,
            "y_mask": np.ones(shape, dtype=bool),
        }
        for flip in config.FLIPS:
            transform = ModeTargetTransform.fit(
                split,
                config.flipped_modes(flip),
                log_enabled=True,
            )
            transformed, mask = transform.training_values(split)
            reconstructed = transform.to_absolute(transformed, current)
            np.testing.assert_allclose(
                reconstructed,
                absolute,
                rtol=1e-10,
                atol=1e-10,
            )
            self.assertTrue(mask.all())

    def test_report_uses_a_canonical_delta_vs_absolute_sign(self):
        rows: list[dict[str, object]] = []
        for target in config.TARGETS:
            target_flip = next(
                flip
                for flip, flip_target in config.FLIP_TARGETS.items()
                if flip_target == target
            )
            for variant in ("base", *config.FLIPS):
                is_target_alternative = variant == target_flip
                mode = (
                    config.flipped_modes(target_flip)[target]
                    if is_target_alternative
                    else config.BASE_MODES[target]
                )
                # Construct every target so absolute RMSE=1 and delta RMSE=2.
                rows.append(
                    {
                        "station": "测试站",
                        "seed": 42,
                        "preprocessing": "A",
                        "preprocessing_label": "A",
                        "target": target,
                        "horizon_hours": 4,
                        "mapping_variant": variant,
                        "rmse": 1.0 if mode == "absolute" else 2.0,
                        "nse": 0.8 if mode == "absolute" else 0.5,
                    }
                )
        detail = build_mode_detail(pd.DataFrame(rows))
        np.testing.assert_allclose(
            detail["delta_relative_absolute_rmse_pct"].to_numpy(),
            100.0,
        )
        np.testing.assert_allclose(
            detail["delta_minus_absolute_nse"].to_numpy(),
            -0.3,
        )

    def test_station_bootstrap_keeps_station_cluster_intact(self):
        frame = pd.DataFrame(
            {
                "station": ["甲", "甲", "乙", "乙"],
                "log_ratio": [np.log(1.03)] * 4,
            }
        )
        lower, upper = _station_bootstrap_geometric_percent_interval(
            frame, "log_ratio", seed=42, repeats=100
        )
        self.assertAlmostEqual(lower, 3.0)
        self.assertAlmostEqual(upper, 3.0)


if __name__ == "__main__":
    unittest.main()
