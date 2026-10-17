from __future__ import annotations

import unittest

from tools.evaluate_active_evidence_value import action_dependency, evaluate


def trace(name, successes, costs):
    rows = {}
    for frame, (success, cost) in enumerate(zip(successes, costs)):
        rows[("seq", frame)] = {
            "sequence": "seq", "frame_index": frame, "accepted": success,
            "pose": "correct" if success else None, "tool_actions": 1,
            "measured_tool_cost_s": cost, "policy": name,
        }
    return rows


class ActiveEvidenceValueTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "scene_disjoint_ready": True, "claim_scope": "scene_disjoint",
            "episodes": [
                {"episode_id": "v", "sequence": "seq", "scene_id": "vscene",
                 "split": "validation", "start_frame": 0, "end_frame": 0},
                {"episode_id": "t", "sequence": "seq", "scene_id": "tscene",
                 "split": "test", "start_frame": 1, "end_frame": 1},
            ]}
        self.gate = {"minimum_oracle_success_gain_pp": 5.0,
                     "or_minimum_cost_reduction_fraction_at_matched_success": 0.2,
                     "minimum_action_dependent_state_fraction": 0.1}

    def test_gate_requires_predictive_signal_only_for_training_go(self):
        traces = {"fixed": trace("fixed", [False, False], [2.0, 2.0]),
                  "fixed_geometry": trace("fixed_geometry", [True, False], [4.0, 4.0]),
                  "finite_depth_oracle": trace("oracle", [True, True], [3.0, 3.0])}
        action_rows = [{"sequence": "seq", "frame_index": 1, "decision_index": 0,
                        "oracle_action_values": [
                            {"parent_path_node_ids": [], "action_value": {"best_terminal_success": 0, "q_star": 0}},
                            {"parent_path_node_ids": [], "action_value": {"best_terminal_success": 1, "q_star": 1}},
                        ]}]
        report = evaluate(self.manifest, traces,
                          lambda sequence, row: row.get("pose") == "correct",
                          oracle_name="finite_depth_oracle",
                          fixed_candidates=["fixed", "fixed_geometry"], action_rows=action_rows,
                          gate=self.gate, predictive_signal_passed=False)
        self.assertEqual(report["best_fixed_policy"], "fixed_geometry")
        self.assertTrue(report["gate"]["realizable_value_pass"])
        self.assertFalse(report["gate"]["learned_agent_go"])

    def test_action_dependency_ignores_non_root_nodes(self):
        rows = [{"sequence": "seq", "frame_index": 1, "oracle_action_values": [
            {"parent_path_node_ids": [], "action_value": {"best_terminal_success": 1, "q_star": 1}},
            {"parent_path_node_ids": ["parent"], "action_value": {"best_terminal_success": 0, "q_star": 0}},
        ]}]
        result = action_dependency(rows, [self.manifest["episodes"][1]])
        self.assertEqual(result["action_dependent_states"], 0)

    def test_missing_frames_are_explicitly_right_censored(self):
        manifest = {"scene_disjoint_ready": False, "claim_scope": "audit_only", "episodes": [
            {"episode_id": "m", "sequence": "seq", "scene_id": "scene", "split": "audit_only",
             "start_frame": 0, "end_frame": 1, "right_censored": False}]}
        traces = {
            "fixed": {("seq", 0): {"sequence": "seq", "frame_index": 0,
                       "online": {"tool_actions": 1, "measured_tool_cost_s": 1.0},
                       "evaluator": {"accepted": False, "pose": None, "memory_write": False}}},
            "finite_depth_oracle": {("seq", 0): {"sequence": "seq", "frame_index": 0,
                                      "online": {"tool_actions": 1, "measured_tool_cost_s": 1.0},
                                      "evaluator": {"accepted": False, "pose": None, "memory_write": False}}},
        }
        report = evaluate(manifest, traces, lambda *_: False, oracle_name="finite_depth_oracle",
                          fixed_candidates=["fixed"], action_rows=[], gate=self.gate)
        row = report["episode_results"]["fixed"][0]
        self.assertEqual(row["missing_frames"], [1])
        self.assertTrue(row["right_censored"])
        self.assertEqual(report["policy_summaries"]["fixed"]["audit_only"]["unrecovered_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
