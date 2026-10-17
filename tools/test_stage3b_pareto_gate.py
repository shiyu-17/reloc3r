#!/usr/bin/env python3
"""Unit tests for the Stage 3B three-objective gate."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from tools.evaluate_stage3b_pareto_gate import (evaluate_gate, load_policy_audits,
                                                pareto_relation, retain_audited)


CONFIG = {
    "success": {"noninferiority_margin": 0.02, "require_ci_noninferiority": False},
    "safety": {"max_false_accept_rate": 0.10},
    "efficiency": {"max_mean_tool_cost_s": 1.0,
                   "require_lower_cost_than_baseline": True},
    "coverage": {"required_sequences": ["a", "b"], "minimum_common_frames": 80},
}


def result(success, far, cost):
    return {"full_frame_success": success, "false_accept_rate": far,
            "mean_measured_tool_cost_s": cost}


class ParetoGateTest(unittest.TestCase):
    def test_passes_noninferior_safe_efficient_policy(self):
        learned = result(0.44, 0.08, 0.7)
        baseline = result(0.45, 0.10, 2.2)
        paired = {"success_difference": {"confidence_interval": [-0.08, 0.06]}}
        gate = evaluate_gate(learned, baseline, paired, CONFIG,
                             common_frames=80, common_sequences=["a", "b"])
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["pareto_relation"], "non_dominated_tradeoff")

    def test_each_objective_can_fail_independently(self):
        baseline = result(0.45, 0.10, 2.2)
        paired = {"success_difference": {"confidence_interval": [-0.20, 0.10]}}
        cases = [
            (result(0.42, 0.08, 0.7), "success_noninferiority"),
            (result(0.45, 0.11, 0.7), "false_accept_safety"),
            (result(0.45, 0.08, 1.0), "tool_cost_efficiency"),
        ]
        for learned, failed in cases:
            with self.subTest(failed=failed):
                gate = evaluate_gate(learned, baseline, paired, CONFIG,
                                     common_frames=80, common_sequences=["a", "b"])
                self.assertFalse(gate["pass"])
                self.assertFalse(gate["objectives"][failed]["pass"])

    def test_incomplete_pairing_blocks_gate(self):
        paired = {"success_difference": {"confidence_interval": [-0.01, 0.01]}}
        gate = evaluate_gate(result(0.45, 0.08, 0.7), result(0.45, 0.10, 2.2),
                             paired, CONFIG, common_frames=79, common_sequences=["a"])
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["coverage_pass"])

    def test_pareto_dominance_directions(self):
        self.assertEqual(
            pareto_relation(result(0.46, 0.08, 0.7), result(0.45, 0.10, 2.2)),
            "learned_dominates_baseline")

    def test_trace_requires_completed_passed_causal_audit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seq").mkdir()
            (root / "seq" / "summary.json").write_text(json.dumps({
                "policies": {"learned": {"seq": {"audit": {"passed": True}}}}
            }), encoding="utf-8")
            audits = load_policy_audits(root, "learned")
            retained, excluded = retain_audited(
                {("seq", 0): {}, ("missing", 0): {}}, audits)
            self.assertIn(("seq", 0), retained)
            self.assertEqual(excluded["missing"], "missing_completed_summary_audit")


if __name__ == "__main__":
    unittest.main()
