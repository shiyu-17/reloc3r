#!/usr/bin/env python3
"""Tests for bounded targeted-DAgger training weights."""

from __future__ import annotations

import unittest

import torch

from tools.train_stage3b_policy import (multi_optimal_policy_loss,
                                        temporal_split,
                                        training_weight,
                                        weighted_value_loss)


TARGETING = {"default_weight": 1.0, "maximum_weight": 4.0,
             "tag_weights": {"high_oracle_regret": 2.0,
                             "memory_optimal": 3.0,
                             "geometry_optimal": 4.0,
                             "premature_accept": 4.0}}


class TargetedDaggerTest(unittest.TestCase):
    def test_weight_uses_bounded_max_not_product(self):
        row = {"target_tags": ["high_oracle_regret", "geometry_optimal"]}
        self.assertEqual(training_weight(row, TARGETING), 4.0)
        self.assertEqual(training_weight({"target_tags": []}, TARGETING), 1.0)

    def test_weighted_losses_are_finite(self):
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        optimal = torch.tensor([[True, False], [True, False]])
        weights = torch.tensor([1.0, 4.0])
        policy = multi_optimal_policy_loss(logits, optimal, weights)
        predicted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        values = torch.zeros_like(predicted)
        mask = torch.ones_like(predicted, dtype=torch.bool)
        value = weighted_value_loss(predicted, values, mask, weights)
        self.assertTrue(torch.isfinite(policy))
        self.assertTrue(torch.isfinite(value))

    def test_dagger_states_are_training_not_temporal_validation(self):
        rows = [
            {"sequence": "seq", "frame_index": index, "dagger_diagnostics": []}
            for index in range(10)
        ]
        dagger = {"sequence": "seq", "frame_index": 20,
                  "dagger_diagnostics": [{"q_gap": 1.0}]}
        rows.append(dagger)
        train, validation, frames = temporal_split(rows, 0.2)
        self.assertIn(dagger, train)
        self.assertNotIn(dagger, validation)
        self.assertEqual(frames["seq"], [8, 9])


if __name__ == "__main__":
    unittest.main()
