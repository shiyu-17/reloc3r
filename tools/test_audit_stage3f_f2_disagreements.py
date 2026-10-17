#!/usr/bin/env python3
"""Focused tests for F2-vs-rule paired disagreement attribution."""

import unittest

from tools.audit_stage3f_f2_disagreements import (
    _first_divergence, _oracle_decision_audit, _root_cause, _state_signature, report_from_rows,
)


def decision(kind, refs=(), q_geometry=0.9, q_localize=0.2):
    return {"action": {"kind": kind, "references": list(refs)}, "state": {"anchor_age": 1},
            "oracle_label": {"oracle_action_values": [
                {"action": {"kind": "geometry", "references": [1, 2]}, "action_value": {"q_star": q_geometry}},
                {"action": {"kind": "localize", "references": [3, 4]}, "action_value": {"q_star": q_localize}},
            ]}}


class Stage3FDisagreementAuditTest(unittest.TestCase):
    def test_attributes_rule_geometry_opportunity(self):
        f2 = {"decisions": [decision("localize", (3, 4))], "terminal_action": "accept"}
        rule = {"decisions": [decision("geometry", (1, 2))], "terminal_action": "accept_legacy"}
        divergence = _first_divergence(f2, rule)
        self.assertEqual(divergence["kind"], "acquisition_action")
        self.assertEqual(_root_cause(divergence, "rule_only_success"),
                         "f2_missed_oracle_geometry_opportunity")

    def test_legacy_accept_does_not_create_false_divergence(self):
        row = {"decisions": [decision("localize", (3, 4))], "terminal_action": "accept"}
        legacy = {"decisions": [decision("localize", (3, 4))], "terminal_action": "accept_legacy"}
        self.assertEqual(_first_divergence(row, legacy)["kind"], "none")

    def test_reads_policy_state_from_nested_observation(self):
        signature = _state_signature({"state": {"state": {
            "anchor_age": 4, "reject_streak": 2, "memory_size": 6,
            "accepted_pose_available": True, "frame_action_count": 1,
            "remaining_budget_s": 27.5,
        }}})
        self.assertEqual(signature["anchor_age"], 4)
        self.assertEqual(signature["memory_size"], 6)

    def test_terminal_audit_detects_eligible_missed_accept(self):
        item = decision("geometry", (1, 2), q_geometry=0.2, q_localize=0.1)
        item["oracle_label"]["oracle_action_values"].append(
            {"action": {"kind": "accept", "references": []}, "action_value": {"q_star": 0.9}})
        item["model_decision"] = {"terminal_eligible": True,
                                  "best_acquisition_utility": 0.7,
                                  "best_terminal_utility": 0.5}
        audit = _oracle_decision_audit(item)
        self.assertTrue(audit["oracle_accept_beats_selected"])
        self.assertEqual(audit["policy_terminal_eligible"], True)

    def test_report_counts_only_outcome_disagreements(self):
        report = report_from_rows([
            {"outcome": "both_success", "taxonomy": "reference_group_switch"},
            {"outcome": "rule_only_success", "taxonomy": "f2_missed_oracle_geometry_opportunity"},
        ])
        self.assertEqual(report["disagreement_frames"], 1)
        self.assertEqual(report["disagreement_taxonomy"],
                         {"f2_missed_oracle_geometry_opportunity": 1})


if __name__ == "__main__":
    unittest.main()
