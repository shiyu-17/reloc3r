"""Regression checks for the retained LILocBench v1 action loop."""
from __future__ import annotations

import unittest

import numpy as np

from tools.active_evidence_protocol import LiveCapabilities
from tools.agentic_localization_env import AgenticLocalizationEnv, CallableLocalizationBackend
from tools.unified_localization_policy import run_fixed_order


CAUSAL = {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False}


class V1PolicyTest(unittest.TestCase):
    @staticmethod
    def make_env(geometry_succeeds: bool):
        def frame(_sequence, index, _state):
            return {"query_frame_id": index,
                    "reference_groups": [{"name": "A", "reference_indices": [1, 2, 3]}]}

        def pose(_sequence, index, refs, _state, source):
            return {"pose": np.eye(4).tolist(), "source": source,
                    "execution": {"executed": True, "cache_hit": False,
                                  "query_frame_index": index, "reference_indices": list(refs),
                                  "elapsed_s": 0.2, "stages": ["pose"]}, "causal": CAUSAL}

        def verify(_sequence, _index, candidate, _state):
            accepted = candidate.source == ("geometry" if geometry_succeeds else "reloc3r")
            return {"accepted": accepted,
                    "evidence": {"status": "supported" if accepted else "rejected",
                                 "elapsed_s": 0.1}, "causal": CAUSAL}

        backend = CallableLocalizationBackend(
            {"seq": 1}, frame,
            lambda s, i, r, st: pose(s, i, r, st, "reloc3r"), verify,
            strong_geometry_fn=lambda s, i, r, st: pose(s, i, r, st, "geometry"))
        return AgenticLocalizationEnv(backend, max_actions_per_frame=12,
                                      budget_s=20., enforce_tool_budget=True,
                                      allow_memory_query=False)

    def test_v1_accepts_reloc3r_without_retrieval_action(self):
        result = run_fixed_order(self.make_env(False), "seq", 0, 1,
                                 LiveCapabilities("precomputed", strong_geometry=True))
        actions = [row["action"]["kind"] for row in result["trace"] if row["type"] == "step"]
        self.assertEqual(actions, ["localize_with_references", "verify_candidate", "accept_candidate"])
        self.assertEqual(result["reference_retrieval_provenance"], "precomputed")
        self.assertTrue(result["environment_audit"]["passed"])

    def test_v1_uses_roma_only_after_rejection(self):
        result = run_fixed_order(self.make_env(True), "seq", 0, 1,
                                 LiveCapabilities("precomputed", strong_geometry=True),
                                 geometry_on_failure=True)
        actions = [row["action"]["kind"] for row in result["trace"] if row["type"] == "step"]
        self.assertEqual(actions, ["localize_with_references", "verify_candidate",
                                   "run_strong_geometry", "verify_candidate", "accept_candidate"])
        self.assertTrue(result["environment_audit"]["passed"])


if __name__ == "__main__":
    unittest.main()
