#!/usr/bin/env python3
"""Regression tests for the utility-aligned Stage 3C.6 protocol."""

import unittest

import numpy as np

from tools.analyze_stage3c6_decisions import audit
from tools.agentic_localization_env import Observation
from tools.build_stage3c6_utility_dataset import component_targets
from tools.stage3c6_utility_policy import UtilityComponentPolicy, compose_utility, policy_state_features


UTILITY = {"success_reward": 1.0, "false_accept_penalty": 2.0,
           "cost_penalty_per_s": 0.05, "recovery_penalty_per_frame": 0.02,
           "memory_pollution_penalty": 1.0}


class Stage3C6Test(unittest.TestCase):
    def test_components_reconstruct_incremental_oracle_q(self):
        observation = {"state": {"frame_tool_cost_s": 0.5}}
        action = {"q_star": -3.075, "best_terminal_metrics": {
            "correct_accept": False, "false_accept": True, "measured_tool_cost_s": 1.5,
            "recovery_time_frames": 0, "memory_pollution": True}}
        targets = component_targets(action, observation, UTILITY)
        self.assertAlmostEqual(targets["component_utility"], -3.05)
        self.assertAlmostEqual(targets["oracle_incremental_utility"], -3.05)

    def test_same_scale_controller_can_continue_after_verified_candidate(self):
        policy = UtilityComponentPolicy(models={}, utility=dict(UTILITY))
        policy.score_actions = lambda observation, actions: {
            "correct_accept": np.array([0.8, 0.55, 0.0]),
            "false_accept": np.array([0.05, 0.20, 0.0]),
            "incremental_cost_s": np.array([1.0, 0.0, 0.0]),
            "recovery_time_frames": np.array([0.0, 0.0, 1.0]),
            "memory_pollution": np.array([0.0, 0.0, 0.0]),
            "expected_utility": np.array([0.65, 0.15, -0.02]),
        }
        observation = {"candidate": {"pose_available": True, "verified": True,
                                      "source": "reloc3r_online"}}
        actions = [{"kind": "geometry"}, {"kind": "accept", "is_terminal_action": True},
                   {"kind": "defer", "is_terminal_action": True}]
        choice, metadata = policy.choose(observation, actions)
        self.assertEqual(choice, 0)
        self.assertGreater(metadata["best_acquisition_utility"], metadata["best_terminal_utility"])

    def test_memory_candidate_cannot_be_accepted(self):
        policy = UtilityComponentPolicy(models={}, utility=dict(UTILITY))
        policy.score_actions = lambda observation, actions: {
            "correct_accept": np.array([0.99, 0.0]), "false_accept": np.zeros(2),
            "incremental_cost_s": np.zeros(2), "recovery_time_frames": np.array([0.0, 1.0]),
            "memory_pollution": np.zeros(2), "expected_utility": np.array([0.99, -0.02])}
        observation = {"candidate": {"pose_available": True, "verified": True, "source": "memory"}}
        choice, metadata = policy.choose(observation, [
            {"kind": "accept", "is_terminal_action": True},
            {"kind": "defer", "is_terminal_action": True}])
        self.assertEqual(choice, 1)
        self.assertEqual(metadata["masked_action_indices"], [0])

    def test_audit_separates_terminal_preemption_from_geometry_ranking(self):
        values = [
            {"action": {"kind": "geometry", "references": [1, 2]}, "action_value": {"q_star": 0.8}},
            {"action": {"kind": "localize", "references": [3, 4]}, "action_value": {"q_star": 0.2}},
            {"action": {"kind": "accept", "references": []}, "action_value": {"q_star": 0.1}},
        ]
        decision = {"action": {"kind": "accept", "references": []},
                    "oracle_label": {"oracle_action_values": values}}
        result = audit([decision])
        self.assertEqual(result["acquisition_admission"]["recall"], 0.0)
        self.assertEqual(result["end_to_end_geometry"]["recall"], 0.0)
        self.assertAlmostEqual(result["acquisition_admission"]["terminal_preemption_regret"]["mean"], 0.7)

    def test_post_geometry_observation_object_is_projected_before_feature_check(self):
        observation = Observation(
            sequence="test", frame_index=1, frame_count=2,
            frame={"reference_groups": []},
            state={"budget_s": 30.0, "remaining_budget_s": 28.0},
            candidate={"source": "roma_online_geometry", "pose_available": True,
                       "verified": False, "evidence": {
                           "geometry_backend": "RoMa", "inliers": 42,
                           "num_references": 3}},
            memory=[], last_event=None, done=False)
        features = policy_state_features(observation, "f3")
        self.assertEqual(features.shape, (47,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
