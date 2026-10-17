#!/usr/bin/env python3
"""Tests for Stage 3D pre-action feature boundaries and targeted sampling."""

import unittest

import numpy as np

from tools.stage3d_features import feature_metadata, feature_vector, strip_post_geometry_fields
from tools.collect_stage3d_recovery_opportunities import classify, geometry_margin


class Stage3DFeatureTest(unittest.TestCase):
    def test_feature_ladder_dimensions_and_preaction_fields(self):
        observation = {"frame": {"retrieval_scores": [0.9, 0.5], "reference_groups": []},
                       "state": {"budget_s": 30.0, "remaining_budget_s": 30.0},
                       "candidate": {"source": "reloc3r_online", "evidence": {
                           "num_references": 2, "translation_dispersion_m": 1.2,
                           "rotation_dispersion_deg": 5.0,
                           "cheap_correspondence": {"coverage_ratio": 0.4, "spatial_entropy": 0.7},
                           "temporal_consistency": {"anchor_retrieval_consistency": 0.8}}}}
        self.assertEqual([len(feature_vector(observation, level)) for level in ("f0", "f1", "f2", "f3")],
                         [24, 34, 40, 47])
        self.assertTrue(np.isfinite(feature_vector(observation, "f3")).all())
        self.assertEqual(feature_metadata("f3")["schema_version"], 2)
        # F0 itself now contains legal cheap-localizer dispersion evidence,
        # rather than delegating every recovery signal to optional features.
        self.assertGreater(feature_vector(observation, "f0")[11], 0.0)

    def test_post_geometry_fields_fail_closed(self):
        observation = {"candidate": {"source": "reloc3r_online", "evidence": {"inliers": 12}}}
        with self.assertRaises(ValueError):
            feature_vector(observation, "f1")

    def test_online_projection_drops_post_geometry_receipt(self):
        observation = {"candidate": {"source": "roma_online_geometry", "evidence": {
            "geometry_backend": "RoMa", "inliers": 42, "num_references": 2}},
            "state": {"budget_s": 30.0}}
        projected = strip_post_geometry_fields(observation)
        self.assertNotIn("geometry_backend", projected["candidate"]["evidence"])
        self.assertNotIn("inliers", projected["candidate"]["evidence"])
        self.assertEqual(projected["candidate"]["evidence"]["num_references"], 2)
        self.assertEqual(len(feature_vector(projected, "f3")), 47)

    def test_targeting_keeps_positive_and_boundary_recovery(self):
        actions = [{"kind": "geometry", "oracle_advantage": 0.2},
                   {"kind": "localize", "oracle_advantage": 0.1}]
        row = {"observation": {"candidate": {"source": "reloc3r_online"}}, "actions": actions}
        self.assertAlmostEqual(geometry_margin(actions), 0.1)
        self.assertEqual(classify(row, 0.15), "geometry_positive_recovery")


if __name__ == "__main__":
    unittest.main()
