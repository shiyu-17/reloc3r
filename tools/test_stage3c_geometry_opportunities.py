#!/usr/bin/env python3
"""Unit tests for Stage 3C.5 live-state opportunity diagnostics."""

import unittest

from tools.analyze_stage3c_geometry_opportunities import audit_decisions
from tools.stage3c_xgboost_policy import XGBRecoveryPolicy


def node(kind, q, references=()):
    return {"action": {"kind": kind, "references": list(references)},
            "action_value": {"q_star": q}}


class GeometryOpportunityTest(unittest.TestCase):
    def test_missed_oracle_geometry_and_controller_decomposition(self):
        decision = {
            "sequence": "static_0", "state": {"candidate": {"source": "reloc3r_online"}},
            "action": {"kind": "localize", "references": [1, 2]},
            "oracle_label": {"oracle_action_values": [
                node("localize", 0.2, [1, 2]), node("geometry", 0.9, [1, 2]), node("defer", 0.1)]},
            "model_decision": {"recovery_probability": [0.4, 0.6],
                               "predicted_incremental_cost_s": [0.1, 5.0],
                               "explicit_advantage": [0.395, 0.35], "rank_score": [0.1, 0.9]},
        }
        report = audit_decisions([decision], lambda_cost=0.05)
        opportunity = report["geometry_opportunity"]
        self.assertEqual(opportunity["oracle_geometry_optimal"], 1)
        self.assertEqual(opportunity["hits"], 0)
        self.assertEqual(opportunity["recall"], 0.0)
        self.assertEqual(opportunity["missed_geometry_regret"]["count"], 1)
        decomposition = report["controller_decomposition"]
        self.assertEqual(decomposition["positive_probability_gain_but_cost_dominated"], 1)
        self.assertEqual(decomposition["ranker_prefers_geometry"], 1)

    def test_xgb_controller_rejects_direct_memory_accept(self):
        policy = XGBRecoveryPolicy(None, None, None, 0.05, 0.0, 0.05, "ranker")
        observation = {"candidate": {"pose_available": True, "verified": True, "source": "memory",
                                      "verification": {"confidence": 0.9}}}
        selected, receipt = policy.choose(observation, [{"kind": "accept"}, {"kind": "defer"}])
        self.assertEqual(selected, 1)
        self.assertEqual(receipt["reason"], "memory_requires_fresh_evidence")


if __name__ == "__main__":
    unittest.main()
