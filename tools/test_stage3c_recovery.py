#!/usr/bin/env python3
"""Tests for Stage 3C recovery/cost controller semantics."""

import unittest

import numpy as np
import torch

from tools.build_stage3c_recovery_dataset import state_weight
from tools.stage3c_recovery_policy import RecoveryCostMLP, RecoveryCostPolicy


def observation(*, verified=False, confidence=0.0, frame_action_count=1, source="reloc3r_online"):
    return {
        "frame": {"reference_groups": [{"reference_indices": [1, 2, 3], "retrieval_scores": [0.8, 0.7, 0.6],
                                           "mean_retrieval_score": 0.7, "map_spread_m": 1.0}]},
        "state": {"budget_s": 30.0, "remaining_budget_s": 30.0, "frame_tool_cost_s": 0.0,
                  "frame_action_count": frame_action_count, "memory_size": 0, "anchor_age": 0,
                  "reject_streak": 0, "accepted_pose_available": False},
        "candidate": ({"pose_available": True, "verified": True, "source": source,
                       "verification": {"confidence": confidence}, "evidence": {}} if verified else None),
        "memory": [],
    }


def policy(*, defer_threshold=0.0):
    model = RecoveryCostMLP(hidden_dim=8)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return RecoveryCostPolicy(model=model, state_mean=np.zeros(26, dtype=np.float32), state_std=np.ones(26, dtype=np.float32),
                              action_mean=np.zeros(12, dtype=np.float32), action_std=np.ones(12, dtype=np.float32),
                              lambda_cost=0.05, defer_threshold=defer_threshold,
                              terminal_continue_margin=0.05, device=torch.device("cpu"))


class RecoveryControllerTest(unittest.TestCase):
    def test_verified_candidate_is_terminal_gated_not_forced_by_action_order(self):
        actions = [{"kind": "localize", "references": [1, 2, 3]},
                   {"kind": "accept", "references": [], "is_terminal_action": True},
                   {"kind": "defer", "references": [], "is_terminal_action": True}]
        selected, metadata = policy().choose(observation(verified=True, confidence=0.9), actions)
        self.assertEqual(actions[selected]["kind"], "accept")
        self.assertEqual(metadata["controller"], "verifier_gated_terminal")

    def test_nonverified_state_can_choose_explicit_defer_baseline(self):
        actions = [{"kind": "localize", "references": [1, 2, 3]},
                   {"kind": "defer", "references": [], "is_terminal_action": True}]
        selected, metadata = policy(defer_threshold=0.6).choose(observation(), actions)
        self.assertEqual(actions[selected]["kind"], "defer")
        self.assertEqual(metadata["reason"], "defer_beats_acquisition")

    def test_memory_candidate_requires_fresh_terminal_evidence(self):
        actions = [{"kind": "accept", "references": [], "is_terminal_action": True},
                   {"kind": "defer", "references": [], "is_terminal_action": True}]
        selected, metadata = policy().choose(observation(verified=True, confidence=0.99, source="memory"), actions)
        self.assertEqual(actions[selected]["kind"], "defer")
        self.assertEqual(metadata["reason"], "memory_requires_fresh_evidence")

    def test_first_acquisition_is_not_blocked_by_defer_threshold(self):
        actions = [{"kind": "localize", "references": [1, 2, 3]},
                   {"kind": "geometry", "references": [1, 2, 3]}]
        selected, _ = policy(defer_threshold=0.6).choose(observation(frame_action_count=0), actions)
        self.assertIn(actions[selected]["kind"], {"localize", "geometry"})

    def test_targeting_is_bounded_and_not_generic_recovery_weighting(self):
        self.assertEqual(state_weight(["recovery:has_candidate"], 1.0, boundary_threshold=0.02), 1.0)
        self.assertEqual(state_weight(["candidate:memory"], 1.0, boundary_threshold=0.02), 1.5)
        self.assertEqual(state_weight([], 0.01, boundary_threshold=0.02), 2.0)


if __name__ == "__main__":
    unittest.main()
