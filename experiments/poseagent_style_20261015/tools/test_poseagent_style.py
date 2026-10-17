#!/usr/bin/env python3
"""Unit tests for the lightweight PoseAgent-style skeleton."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor

from tools.poseagent_style import (DEFAULT_TOOLS, PairProfiler, PoseAgentFallbackController,
                                   RuleVerifier, StructuredVerifier, StructuredVerifierFeatures)


def observation(candidate=None):
    return {
        "sequence": "scene_a", "frame_index": 3, "done": False,
        "frame": {"reference_groups": [
            {"reference_indices": [1, 2, 3], "retrieval_scores": [0.9, 0.8, 0.7],
             "mean_retrieval_score": 0.8},
            {"reference_indices": [4, 5, 6], "retrieval_scores": [0.5, 0.4, 0.3],
             "mean_retrieval_score": 0.4},
        ]},
        "state": {"anchor_age": 2, "reject_streak": 0, "memory_size": 1,
                  "accepted_pose_available": True, "frame_action_count": 0,
                  "frame_tool_cost_s": 0.0, "budget_s": 30.0, "remaining_budget_s": 30.0},
        "candidate": candidate, "memory": [],
        "causal": {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False},
    }


class PoseAgentStyleTest(unittest.TestCase):
    def test_profiler_is_fixed_schema_and_reference_id_free(self):
        profiler = PairProfiler()
        group = observation()["frame"]["reference_groups"][0]
        profile = profiler.profile(observation(), DEFAULT_TOOLS[0], 0, group)
        self.assertEqual(set(profile), set(profiler.FEATURE_NAMES))
        self.assertAlmostEqual(profile["retrieval_mean"], 0.8)
        self.assertNotIn("reference_indices", profile)

    def test_controller_requires_verification_before_accept(self):
        controller = PoseAgentFallbackController(max_estimator_calls=2)
        first = controller.choose(observation())
        self.assertEqual(first.action.kind, "localize_with_references")
        unverified = {"source": "reloc3r_online", "pose_available": True,
                      "verified": False, "evidence": {}, "verification": {}}
        self.assertEqual(controller.choose(observation(unverified)).action.kind, "verify_candidate")
        accepted = {"source": "reloc3r_online", "pose_available": True,
                    "verified": True, "evidence": {},
                    "verification": {"confidence": 0.9}}
        self.assertEqual(controller.choose(observation(accepted)).action.kind, "accept_candidate")

    def test_rule_rejection_falls_back_or_defers(self):
        controller = PoseAgentFallbackController(max_estimator_calls=1)
        controller.choose(observation())
        rejected = {"source": "reloc3r_online", "pose_available": True,
                    "verified": False, "evidence": {},
                    "verification": {"confidence": 0.9, "total_risk": 0.1}}
        decision = controller.choose(observation(rejected))
        self.assertEqual(decision.action.kind, "defer_frame")

    def test_rule_verifier_is_only_an_adapter(self):
        rejected = {"source": "reloc3r_online", "pose_available": True,
                    "verified": False, "evidence": {}, "verification": {"confidence": 1.0}}
        self.assertFalse(RuleVerifier().evaluate(observation(rejected)).accept)

    def test_structured_features_reject_gt_leakage(self):
        leaking = observation({"source": "reloc3r_online", "pose_available": True,
                               "verified": False, "evidence": {"success": True}, "verification": {}})
        with self.assertRaises(ValueError):
            StructuredVerifierFeatures.vector(leaking)

    def test_structured_feature_dimension_is_stable(self):
        candidate = {"source": "roma_online_geometry", "pose_available": True,
                     "verified": False, "evidence": {"inliers": 32},
                     "verification": {"total_risk": 0.2}}
        features = StructuredVerifierFeatures.vector(observation(candidate))
        self.assertEqual(features.shape, (len(StructuredVerifierFeatures.NAMES),))

    def test_sklearn_structured_verifier_artifact_loads_safely(self):
        candidate = {"source": "reloc3r_online", "pose_available": True,
                     "verified": True, "evidence": {}, "verification": {"confidence": 0.8}}
        x = np.vstack([StructuredVerifierFeatures.vector(observation(candidate)),
                       StructuredVerifierFeatures.vector(observation(candidate)) + 0.01])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classifier = DummyClassifier(strategy="constant", constant=1).fit(x, [0, 1])
            translation = DummyRegressor(strategy="constant", constant=0.0).fit(x, [0.0, 0.0])
            rotation = DummyRegressor(strategy="constant", constant=0.0).fit(x, [0.0, 0.0])
            joblib.dump(classifier, root / "success_classifier.joblib")
            joblib.dump(translation, root / "translation_regressor.joblib")
            joblib.dump(rotation, root / "rotation_regressor.joblib")
            (root / "metadata.json").write_text(json.dumps({
                "feature_names": list(StructuredVerifierFeatures.NAMES),
                "model_backend": "sklearn_hist", "acceptance_threshold": 0.5,
                "require_rule_accept": True,
            }))
            result = StructuredVerifier.load(root).evaluate(observation(candidate))
        self.assertTrue(result.accept)
        self.assertEqual(result.verifier, "structured")


if __name__ == "__main__":
    unittest.main()
