#!/usr/bin/env python3
"""Small causal smoke tests for tools.agentic_localization_env."""

from __future__ import annotations

import unittest

import numpy as np

from tools.agentic_localization_env import (
    AgenticLocalizationEnv,
    CallableLocalizationBackend,
    CausalViolation,
    LocalizationAction,
    InvalidAction,
    MemoryRecord,
)
from tools.lilocbench_agent_backend import previous_state_risk, state_trust


def pose(x: float) -> list:
    value = np.eye(4, dtype=float)
    value[0, 3] = x
    return value.tolist()


class AgenticEnvironmentTest(unittest.TestCase):
    def make_backend(self, *, cache_hit: bool = False, accept: bool = True):
        calls = []

        def frame_fn(sequence, index, state):
            return {"image_id": f"{sequence}:{index}"}

        def localize_fn(sequence, index, refs, state):
            calls.append((index, tuple(refs)))
            return {
                "pose": pose(float(index)), "source": "reloc3r",
                "evidence": {"references_used": list(refs)},
                "execution": {"executed": not cache_hit, "cache_hit": cache_hit,
                               "query_frame_index": index, "reference_indices": list(refs),
                               "stages": ["pair_preprocessing", "relative_pose_estimation",
                                           "pose_aggregation"]},
                "causal": {"query_gt_read": False, "uses_future": False,
                            "selection_uses_gt": False},
            }

        def verify_fn(sequence, index, candidate, state):
            return {"accepted": accept, "evidence": {"confidence": 0.9},
                    "causal": {"query_gt_read": False, "uses_future": False,
                               "selection_uses_gt": False}}

        def geometry_fn(sequence, index, refs, state):
            calls.append(("geometry", index, tuple(refs)))
            return {
                "pose": pose(float(index) + 0.5), "source": "roma_online_geometry",
                "evidence": {"inliers": 42},
                "execution": {"executed": True, "cache_hit": False,
                               "query_frame_index": index, "reference_indices": list(refs),
                               "stages": ["roma_dense_matching", "triangulation", "pnp"]},
                "causal": {"query_gt_read": False, "uses_future": False,
                            "selection_uses_gt": False},
            }

        def retrieve_references_fn(sequence, index, top_k, state):
            return {
                "references": [{"reference_index": 7, "score": 0.9},
                               {"reference_index": 8, "score": 0.8}][:top_k],
                "groups": [{"name": "A", "reference_indices": [7, 8]}],
                "execution": {"executed": True, "cache_hit": False,
                              "query_frame_index": index, "elapsed_s": 0.12,
                              "stages": ["query_embedding", "map_search"]},
                "causal": {"query_gt_read": False, "uses_future": False,
                           "selection_uses_gt": False},
            }

        backend = CallableLocalizationBackend({"seq": 2}, frame_fn,
                                               localize_fn, verify_fn,
                                               strong_geometry_fn=geometry_fn,
                                               retrieve_references_fn=retrieve_references_fn)
        return backend, calls

    def test_action_changes_execution_and_only_verified_pose_enters_memory(self):
        backend, calls = self.make_backend()
        env = AgenticLocalizationEnv(backend)
        obs = env.reset("seq")
        self.assertEqual(obs.frame_index, 0)
        obs, _, _, _ = env.step(LocalizationAction.localize(7, 8))
        self.assertEqual(calls, [(0, (7, 8))])
        self.assertEqual(obs.memory, [])
        obs, _, _, _ = env.step(LocalizationAction.verify())
        self.assertTrue(obs.candidate["verified"])
        obs, _, _, _ = env.step(LocalizationAction.accept())
        self.assertEqual(obs.frame_index, 1)
        self.assertEqual(len(obs.memory), 1)
        self.assertEqual(obs.memory[0]["frame_index"], 0)
        obs, _, _, _ = env.step(LocalizationAction.retrieve())
        self.assertTrue(obs.candidate["pose_available"])
        env.step(LocalizationAction.verify())
        _, _, done, _ = env.step(LocalizationAction.accept())
        self.assertTrue(done)
        audit = env.audit_trace()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["query_gt_read"], False)

    def test_rejected_candidate_cannot_be_accepted_or_written(self):
        backend, _ = self.make_backend(accept=False)
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        env.step(LocalizationAction.localize(1))
        env.step(LocalizationAction.verify())
        with self.assertRaises(InvalidAction):
            env.step(LocalizationAction.accept())
        self.assertEqual(len(env.state.memory), 0)

    def test_protocol_can_disable_memory_pose_queries(self):
        backend, _ = self.make_backend()
        env = AgenticLocalizationEnv(backend, allow_memory_query=False)
        env.reset("seq")
        env.step(LocalizationAction.localize(1))
        env.step(LocalizationAction.verify())
        env.step(LocalizationAction.accept())
        with self.assertRaises(InvalidAction):
            env.step(LocalizationAction.retrieve())

    def test_new_reference_selection_invalidates_old_verification(self):
        backend, calls = self.make_backend()
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        env.step(LocalizationAction.localize(1, 2))
        env.step(LocalizationAction.verify())
        env.step(LocalizationAction.localize(3, 4))
        self.assertEqual(calls, [(0, (1, 2)), (0, (3, 4))])
        self.assertFalse(env.state.current_candidate.verified)
        with self.assertRaises(InvalidAction):
            env.step(LocalizationAction.accept())
        self.assertEqual(len(env.state.memory), 0)

    def test_cache_receipt_is_rejected(self):
        backend, _ = self.make_backend(cache_hit=True)
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        with self.assertRaises(CausalViolation):
            env.step(LocalizationAction.localize(1))

    def test_strong_geometry_is_a_live_reference_conditioned_action(self):
        backend, calls = self.make_backend()
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        obs, _, _, _ = env.step(LocalizationAction.geometry(4, 5))
        self.assertEqual(calls, [("geometry", 0, (4, 5))])
        self.assertEqual(obs.candidate["source"], "roma_online_geometry")
        env.step(LocalizationAction.verify())
        env.step(LocalizationAction.accept())
        self.assertTrue(env.audit_trace()["passed"], env.audit_trace())

    def test_reference_retrieval_is_live_visible_and_costed(self):
        backend, _ = self.make_backend()
        env = AgenticLocalizationEnv(backend, budget_s=1.0)
        obs = env.reset("seq")
        obs, _, _, _ = env.step(LocalizationAction.retrieve_references(top_k=2))
        self.assertEqual(obs.last_event["type"], "retrieve_references")
        self.assertEqual([x["reference_index"] for x in obs.last_event["references"]], [7, 8])
        self.assertAlmostEqual(obs.state["frame_tool_cost_s"], 0.12)
        self.assertAlmostEqual(obs.state["remaining_budget_s"], 0.88)
        env.step(LocalizationAction.localize(7, 8))
        self.assertTrue(env.audit_trace()["passed"], env.audit_trace())

    def test_reference_retrieval_rejects_cached_results(self):
        backend, _ = self.make_backend()
        original = backend._retrieve_references_fn
        backend._retrieve_references_fn = lambda sequence, index, top_k, state: dict(
            original(sequence, index, top_k, state),
            execution={"executed": False, "cache_hit": True,
                       "query_frame_index": index, "stages": ["cache_lookup"]})
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        with self.assertRaises(CausalViolation):
            env.step(LocalizationAction.retrieve_references())

    def test_policy_can_select_an_earlier_candidate_without_rerunning_it(self):
        backend, calls = self.make_backend()
        env = AgenticLocalizationEnv(backend)
        obs = env.reset("seq")
        obs, _, _, _ = env.step(LocalizationAction.localize(7))
        obs, _, _, _ = env.step(LocalizationAction.localize(8))
        self.assertEqual(len(obs.candidates), 2)
        self.assertEqual(calls, [(0, (7,)), (0, (8,))])
        obs, _, _, _ = env.step(LocalizationAction.select_candidate(0))
        self.assertEqual(obs.candidate["reference_indices"], [7])
        env.step(LocalizationAction.verify())
        obs, _, _, _ = env.step(LocalizationAction.accept())
        self.assertAlmostEqual(env.state.memory[0].pose[0, 3], 0.0)
        self.assertEqual(obs.frame_index, 1)
        self.assertEqual(obs.candidates, [])

    def test_mismatched_execution_receipt_is_rejected(self):
        backend, _ = self.make_backend()
        original = backend._localize_fn

        def localize(sequence, index, refs, state):
            result = dict(original(sequence, index, refs, state))
            result["execution"] = dict(result["execution"])
            result["execution"]["reference_indices"] = [999]
            return result

        backend._localize_fn = localize
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        with self.assertRaises(CausalViolation):
            env.step(LocalizationAction.localize(1))

    def test_evaluator_outcome_cannot_cross_online_boundary(self):
        backend, _ = self.make_backend()
        original = backend._localize_fn
        backend._localize_fn = lambda sequence, index, refs, state: dict(
            original(sequence, index, refs, state), success=True)
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        with self.assertRaises(CausalViolation):
            env.step(LocalizationAction.localize(1))

    def test_state_age_and_reject_streak_are_causal_backend_inputs(self):
        seen = []
        backend, _ = self.make_backend()
        original = backend._verify_fn

        def verify(sequence, index, candidate, state):
            seen.append((state["anchor_age"], state["reject_streak"]))
            return original(sequence, index, candidate, state)

        backend._verify_fn = verify
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        env.step(LocalizationAction.localize(1))
        env.step(LocalizationAction.verify())
        env.step(LocalizationAction.accept())
        env.step(LocalizationAction.retrieve())
        env.step(LocalizationAction.verify())
        self.assertEqual(seen[0], (1, 0))
        self.assertEqual(seen[1], (1, 0))

    def test_live_receipt_cost_is_policy_visible_and_resets_per_frame(self):
        backend, _ = self.make_backend()
        original_localize = backend._localize_fn
        original_verify = backend._verify_fn

        def localize(sequence, index, refs, state):
            result = dict(original_localize(sequence, index, refs, state))
            result["execution"] = dict(result["execution"], elapsed_s=0.25)
            return result

        def verify(sequence, index, candidate, state):
            result = dict(original_verify(sequence, index, candidate, state))
            result["evidence"] = dict(result["evidence"], elapsed_s=0.50)
            return result

        backend._localize_fn = localize
        backend._verify_fn = verify
        env = AgenticLocalizationEnv(backend, budget_s=1.0)
        obs = env.reset("seq")
        self.assertAlmostEqual(obs.state["remaining_budget_s"], 1.0)
        obs, _, _, _ = env.step(LocalizationAction.localize(1))
        self.assertAlmostEqual(obs.state["frame_tool_cost_s"], 0.25)
        self.assertAlmostEqual(obs.state["remaining_budget_s"], 0.75)
        obs, _, _, _ = env.step(LocalizationAction.verify())
        self.assertAlmostEqual(obs.state["frame_tool_cost_s"], 0.75)
        self.assertAlmostEqual(obs.state["remaining_budget_s"], 0.25)
        obs, _, _, _ = env.step(LocalizationAction.accept())
        self.assertEqual(obs.frame_index, 1)
        self.assertAlmostEqual(obs.state["frame_tool_cost_s"], 0.0)

    def test_soft_tool_budget_blocks_new_acquisition_but_allows_defer(self):
        backend, _ = self.make_backend()
        env = AgenticLocalizationEnv(backend, budget_s=0.1, enforce_tool_budget=True)
        env.reset("seq")
        env.step(LocalizationAction.retrieve_references(top_k=2))
        with self.assertRaises(InvalidAction):
            env.step(LocalizationAction.localize(7, 8))
        self.assertEqual(env.state.frame_action_count, 1)
        obs, _, _, _ = env.step(LocalizationAction.defer())
        self.assertEqual(obs.frame_index, 1)

    def test_state_retrust_decays_only_previous_state_risk(self):
        previous = np.eye(4)
        candidate = np.eye(4)
        candidate[0, 3] = 2.0
        raw = previous_state_risk(candidate, previous)
        fresh = state_trust(0, 0) * raw
        stale = state_trust(10000, 10000) * raw
        self.assertGreater(raw, 0.0)
        self.assertGreater(fresh, stale)

    def test_future_memory_record_is_rejected(self):
        backend, _ = self.make_backend()
        env = AgenticLocalizationEnv(backend)
        env.reset("seq")
        env.state.memory.append(MemoryRecord(
            memory_id=0, frame_index=1, pose=np.eye(4), source="test", confidence=1.0))
        with self.assertRaises(CausalViolation):
            env.step(LocalizationAction.retrieve())
        self.assertEqual(env.state.frame_action_count, 0)


if __name__ == "__main__":
    unittest.main()
