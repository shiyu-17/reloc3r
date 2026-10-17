#!/usr/bin/env python3
"""Focused invariants for Stage 3E data targeting and score composition."""

import unittest

import torch

from tools.build_stage3e_recovery_ranking_dataset import geometry_margin, targeting_weight
from tools.stage3e_recovery_ranking_policy import RecoveryCostRanker


class Stage3ERecoveryRankingTest(unittest.TestCase):
    def test_targeting_keeps_positive_recovery_and_bounds_weight(self):
        reason, weight = targeting_weight({"recovery", "geometry_available"}, 0.2, 0.03)
        self.assertEqual(reason, "geometry_positive_recovery")
        self.assertEqual(weight, 4.0)
        reason, weight = targeting_weight({"root"}, -0.01, 0.03)
        self.assertEqual(reason, "geometry_boundary")
        self.assertEqual(weight, 2.0)

    def test_margin_compares_geometry_with_best_cheap_action(self):
        actions = [
            {"kind": "geometry", "oracle_value": 0.7},
            {"kind": "localize", "oracle_value": 0.2},
            {"kind": "memory", "oracle_value": 0.4},
        ]
        self.assertAlmostEqual(geometry_margin(actions), 0.3)

    def test_model_emits_one_score_per_legal_action(self):
        model = RecoveryCostRanker(state_dim=3, action_dim=2, hidden_dim=8)
        logits, costs = model(torch.zeros(3), torch.zeros(4, 2))
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(costs.shape), (4,))


if __name__ == "__main__":
    unittest.main()
