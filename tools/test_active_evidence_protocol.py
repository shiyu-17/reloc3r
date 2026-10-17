"""Protocol-level parity checks shared by rules, policies, and the Oracle."""

from __future__ import annotations

import unittest

import numpy as np

from tools.active_evidence_protocol import action_key, legal_actions
from tools.agentic_localization_env import AgenticLocalizationEnv, CallableLocalizationBackend
from tools.eval_active_evidence_sequential import _oracle_expand


def pose() -> list:
    value = np.eye(4, dtype=float)
    return value.tolist()


class ActionProtocolTest(unittest.TestCase):
    def make_env(self):
        def frame(sequence, index, state):
            return {"reference_groups": [
                {"reference_indices": [1, 2], "mean_retrieval_score": 0.9, "map_spread_m": 1.0},
                {"reference_indices": [3, 4], "mean_retrieval_score": 0.8, "map_spread_m": 2.0},
            ]}
        def action(sequence, index, refs, state):
            return {"pose": pose(), "source": "reloc3r_online", "evidence": {},
                    "execution": {"executed": True, "cache_hit": False,
                                  "query_frame_index": index, "reference_indices": list(refs),
                                  "stages": ["test"]},
                    "causal": {"query_gt_read": False, "uses_future": False,
                               "selection_uses_gt": False}}
        def verify(sequence, index, candidate, state):
            return {"accepted": True, "evidence": {"confidence": 1.0},
                    "causal": {"query_gt_read": False, "uses_future": False,
                               "selection_uses_gt": False}}
        backend = CallableLocalizationBackend({"seq": 1}, frame, action, verify,
                                              strong_geometry_fn=action)
        env = AgenticLocalizationEnv(backend, allow_memory_query=False)
        return env, env.reset("seq")

    def test_oracle_root_actions_match_shared_contract(self):
        env, observation = self.make_env()
        expected = {action_key(action) for action in legal_actions(
            observation, [0, 1], geometry_used=False, allow_acquisition=True)}
        nodes = []
        _oracle_expand(env, observation.frame["reference_groups"], (0, 1), False, False,
                       0, 1, 0.0, 30.0, nodes, ())
        actual = {action_key(node["action"]) for node in nodes
                  if not node["parent_path_node_ids"]}
        self.assertEqual(actual, expected)
        self.assertNotIn(("memory", ()), actual)


if __name__ == "__main__":
    unittest.main()
