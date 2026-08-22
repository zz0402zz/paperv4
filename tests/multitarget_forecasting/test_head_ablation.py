from __future__ import annotations

import unittest

import numpy as np

from scripts.multitarget_forecasting import config as base_config
from scripts.multitarget_forecasting import head_ablation_config as config
from scripts.multitarget_forecasting.head_ablation_model import (
    build_model,
    mixed_targets,
    to_absolute_prediction,
)


class TargetSpecificHeadAblationTests(unittest.TestCase):
    def test_every_variant_outputs_eighteen_horizons_and_five_targets(self):
        import torch

        for variant in config.VARIANTS:
            network = build_model(
                torch, sequence_dim=36, context_dim=10, variant=variant
            )
            predicted = network(torch.zeros(3, 6, 36), torch.zeros(3, 10))
            self.assertEqual(tuple(predicted.shape), (3, 18, 5))

    def test_shared_mlp_and_target_heads_have_matched_head_capacity(self):
        import torch

        shared = build_model(
            torch, sequence_dim=36, context_dim=10, variant="mixed_shared_mlp"
        )
        target = build_model(
            torch, sequence_dim=36, context_dim=10, variant="mixed_target_heads"
        )
        shared_count = sum(p.numel() for p in shared.output_head.parameters())
        target_count = sum(p.numel() for p in target.output_head.parameters())
        self.assertLess(abs(shared_count - target_count) / shared_count, 0.01)

    def test_mixed_representation_and_absolute_reconstruction(self):
        absolute = np.arange(2 * 18 * 5, dtype=float).reshape(2, 18, 5)
        delta = absolute + 1_000.0
        split = {
            "y_abs": absolute,
            "y_delta": delta,
            "y_mask": np.ones_like(absolute, dtype=bool),
        }
        mixed, mask = mixed_targets(split)
        for target_index, target in enumerate(config.TARGETS):
            expected = (
                delta
                if config.TARGET_OUTPUT_MODES[target] == "delta"
                else absolute
            )
            np.testing.assert_array_equal(
                mixed[:, :, target_index], expected[:, :, target_index]
            )
        self.assertTrue(mask.all())

        current = np.full((2, 5), 10.0)
        reconstructed = to_absolute_prediction(mixed, current)
        for target_index, target in enumerate(config.TARGETS):
            expected = mixed[:, :, target_index]
            if config.TARGET_OUTPUT_MODES[target] == "delta":
                expected = expected + 10.0
            np.testing.assert_array_equal(
                reconstructed[:, :, target_index], expected
            )

    def test_paths_are_distinct_and_use_chinese_labels(self):
        paths = [config.prediction_path("测试站", variant, 42) for variant in config.VARIANTS]
        self.assertEqual(len(set(paths)), len(config.VARIANTS))
        self.assertTrue(all("测试站" in path.name for path in paths))

    def test_mixed_mode_mapping_covers_targets_exactly(self):
        self.assertEqual(set(config.TARGET_OUTPUT_MODES), set(base_config.TARGETS))
        self.assertEqual(
            set(config.TARGET_OUTPUT_MODES.values()), {"absolute", "delta"}
        )


if __name__ == "__main__":
    unittest.main()
