#!/usr/bin/env python3
"""Small invariants for the structured Stage 3B policy interface."""

from __future__ import annotations

import unittest

from tools.stage3b_policy import (ACTION_DIM, STATE_DIM, action_features,
                                  legal_actions, state_features)


def observation(*, frame_actions: int, memory: bool = False):
    return {
        "frame": {"reference_groups": [
            {"reference_indices": [1, 2, 3], "retrieval_scores": [0.8, 0.7, 0.6],
             "mean_retrieval_score": 0.7, "map_spread_m": 2.0},
            {"reference_indices": [4, 5, 6], "retrieval_scores": [0.5, 0.4, 0.3],
             "mean_retrieval_score": 0.4, "map_spread_m": 1.0},
        ]},
        "state": {"frame_action_count": frame_actions, "budget_s": 30.0,
                  "remaining_budget_s": 30.0, "anchor_age": 1,
                  "reject_streak": 0, "memory_size": int(memory)},
        "candidate": None,
        "memory": ([{"memory_id": 0}] if memory else []),
    }


class Stage3BPolicyTest(unittest.TestCase):
    def test_first_decision_masks_defer(self):
        actions = legal_actions(observation(frame_actions=0), [0, 1],
                                geometry_used=False, memory_used=False)
        self.assertNotIn("defer", [item["kind"] for item in actions])
        self.assertEqual([item["kind"] for item in actions],
                         ["localize", "localize", "geometry", "geometry"])
        self.assertEqual([item["reference_group"] for item in actions if item["kind"] == "geometry"], [0, 1])

    def test_later_decision_exposes_defer_but_not_memory_query(self):
        actions = legal_actions(observation(frame_actions=1, memory=True), [1],
                                geometry_used=True, memory_used=False)
        self.assertEqual([item["kind"] for item in actions], ["localize", "defer"])

    def test_acquisition_cap_still_exposes_terminal_decision(self):
        obs = observation(frame_actions=4)
        obs["candidate"] = {"pose_available": True, "verified": True}
        actions = legal_actions(obs, [1], geometry_used=False, memory_used=False,
                                allow_acquisition=False)
        self.assertEqual([item["kind"] for item in actions], ["accept", "defer"])

    def test_reject_is_a_legacy_alias_of_defer(self):
        obs = observation(frame_actions=1)
        reject = action_features(obs, {"kind": "reject", "references": []})
        defer = action_features(obs, {"kind": "defer", "references": []})
        self.assertTrue((reject == defer).all())

    def test_features_never_include_reference_ids(self):
        obs = observation(frame_actions=0)
        state = state_features(obs)
        action = action_features(obs, {"kind": "localize", "references": [1, 2, 3]})
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(action.shape, (ACTION_DIM,))


if __name__ == "__main__":
    unittest.main()
