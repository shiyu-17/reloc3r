#!/usr/bin/env python3
"""Unit tests for the read-only Stage 3B Q-calibration audit helpers."""

import unittest

from tools.analyze_stage3b_q_calibration import calibration, margin_summary, strata


class QCalibrationAuditTest(unittest.TestCase):
    def test_calibration_reports_signed_bias(self):
        report = calibration([
            {"predicted": 2.0, "oracle": 1.0},
            {"predicted": 4.0, "oracle": 3.0},
        ])
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["bias_predicted_minus_oracle"], 1.0)
        self.assertAlmostEqual(report["mae"], 1.0)

    def test_margin_tracks_order_not_only_absolute_q(self):
        report = margin_summary([
            {"predicted": 0.2, "oracle": 0.1},
            {"predicted": -0.3, "oracle": -0.1},
        ])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["margin_sign_accuracy_non_ties"], 1.0)

    def test_strata_distinguish_regime_anchor_and_memory(self):
        output = strata({"sequence": "lt_changes_dynamics_0", "candidate": {"verified": True,
                         "pose_available": True, "source": "memory"},
                         "state": {"accepted_pose_available": True, "anchor_age": 6, "memory_size": 2}})
        self.assertIn("regime:long_term_change", output)
        self.assertIn("anchor:stale", output)
        self.assertIn("memory:available", output)
        self.assertIn("verifier:passed", output)


if __name__ == "__main__":
    unittest.main()
