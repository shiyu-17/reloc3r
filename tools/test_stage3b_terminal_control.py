#!/usr/bin/env python3
"""End-to-end regression tests for Stage 3B terminal control."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from tools.agentic_localization_env import (AgenticLocalizationEnv,
                                            CallableLocalizationBackend,
                                            LocalizationAction)
from tools.eval_active_evidence_sequential import (DEFAULT_UTILITY,
                                                   label_oracle_state,
                                                   run_policy)


class ScriptedTerminalPolicy:
    """Acquire twice, then deliberately Accept the last verified candidate."""

    def __init__(self):
        self.candidate_kinds = []
        self.remaining_budgets = []

    def choose(self, observation, candidates):
        kinds = [item["kind"] for item in candidates]
        self.candidate_kinds.append(kinds)
        self.remaining_budgets.append(observation.state["remaining_budget_s"])
        if "localize" in kinds:
            return kinds.index("localize"), {"scripted": True}
        return kinds.index("accept"), {"scripted": True}


class AcceptOnlyTerminalPolicy:
    def choose(self, observation, candidates):
        kinds = [item["kind"] for item in candidates]
        if kinds != ["accept", "defer"]:
            raise AssertionError(f"expected terminal-only candidates, got {kinds}")
        return 0, {"terminal_aligned": True}


def pose(x: float):
    value = np.eye(4, dtype=float)
    value[0, 3] = x
    return value.tolist()


class Stage3BTerminalControlTest(unittest.TestCase):
    @staticmethod
    def backend():
        def frame_fn(sequence, index, state):
            return {"reference_groups": [
                {"reference_indices": [1, 2], "retrieval_scores": [0.9, 0.8],
                 "mean_retrieval_score": 0.85, "map_spread_m": 1.0},
                {"reference_indices": [3, 4], "retrieval_scores": [0.7, 0.6],
                 "mean_retrieval_score": 0.65, "map_spread_m": 1.0},
            ]}

        def localize_fn(sequence, index, refs, state):
            return {
                "pose": pose(float(refs[0])), "source": "test_online",
                "evidence": {},
                "execution": {"executed": True, "cache_hit": False,
                              "query_frame_index": index,
                              "reference_indices": list(refs), "elapsed_s": 0.25,
                              "stages": ["test_localization"]},
                "causal": {"query_gt_read": False, "uses_future": False,
                           "selection_uses_gt": False},
            }

        def verify_fn(sequence, index, candidate, state):
            return {
                "accepted": True,
                "evidence": {"confidence": 0.9, "elapsed_s": 0.05},
                "causal": {"query_gt_read": False, "uses_future": False,
                           "selection_uses_gt": False},
            }

        return CallableLocalizationBackend({"seq": 1}, frame_fn, localize_fn, verify_fn)

    def test_final_acquisition_can_still_be_accepted(self):
        args = SimpleNamespace(max_acquisitions=2, budget_ms=30_000, seed=777,
                               geometry_schedule_rate=0.0,
                               uncertainty_threshold=0.8, router_threshold=0.0)
        policy = ScriptedTerminalPolicy()
        result = run_policy(args, "seq", "learned", 0, 1,
                            backend=self.backend(), learned_policy=policy)
        record = result["records"][0]

        self.assertEqual(len(record["online"]["acquisitions"]), 2)
        self.assertTrue(record["evaluator"]["accepted"])
        self.assertEqual(record["online"]["terminal_action"], "accept")
        self.assertEqual(policy.candidate_kinds[-1], ["accept", "defer"])
        self.assertNotEqual(record["online"]["terminal_action"], "defer_fallback")
        self.assertAlmostEqual(policy.remaining_budgets[0], 30.0)
        self.assertLess(policy.remaining_budgets[-1], 30.0)

    def test_dagger_oracle_starts_from_live_spent_cost(self):
        env = AgenticLocalizationEnv(self.backend(), budget_s=30.0)
        obs = env.reset("seq")
        groups = obs.frame["reference_groups"]
        env.step(LocalizationAction.localize(1, 2))
        env.step(LocalizationAction.verify())
        args = SimpleNamespace(oracle_depth=2, budget_ms=30_000,
                               utility=dict(DEFAULT_UTILITY))
        label = label_oracle_state(args, env, groups, [1], True, False, 1,
                                   np.asarray(pose(1.0)))
        self.assertAlmostEqual(label["observation"]["state"]["frame_tool_cost_s"], 0.30)
        self.assertAlmostEqual(label["observation"]["state"]["remaining_budget_s"], 29.70)
        self.assertTrue(label["oracle_action_values"])
        for node in label["oracle_action_values"]:
            self.assertAlmostEqual(node["prior_cost_s"], 0.30)

    def test_feedback_acquisition_can_use_aligned_terminal_controller(self):
        args = SimpleNamespace(max_acquisitions=2, budget_ms=30_000, seed=777,
                               geometry_schedule_rate=0.0,
                               uncertainty_threshold=0.8, router_threshold=0.0)
        result = run_policy(args, "seq", "feedback_rule", 0, 1,
                            backend=self.backend(), terminal_policy=AcceptOnlyTerminalPolicy())
        record = result["records"][0]
        self.assertTrue(record["evaluator"]["accepted"])
        self.assertEqual(record["online"]["terminal_action"], "accept")
        self.assertEqual(record["online"]["decisions"][-1]["action"]["kind"], "accept")
        self.assertTrue(record["online"]["decisions"][-1]["model_decision"]["terminal_aligned"])


if __name__ == "__main__":
    unittest.main()
