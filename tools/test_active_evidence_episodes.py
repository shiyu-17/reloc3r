from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from tools.active_evidence_episodes import (
    TriggerConfig, assign_scene_splits, build_manifest, load_episode_spans,
    mine_sequence, trigger_reasons,
)


def row(frame, accepted, *, risk=0.0, anchor=0):
    return {
        "sequence": "seq", "frame_index": frame, "accepted": accepted,
        "terminal_action": "accept" if accepted else "defer_fallback",
        "acquisitions": [{
            "candidate": {"evidence": {}, "verification": {"total_risk": risk,
                                                              "anchor_age": anchor}}
        }],
    }


class ActiveEvidenceEpisodeTest(unittest.TestCase):
    def test_trigger_uses_only_online_receipts(self):
        reasons = trigger_reasons(row(3, False, risk=2.0, anchor=20), TriggerConfig())
        self.assertEqual(reasons, ["candidate_conflict", "stale_anchor", "verifier_failure"])

    def test_v2_miner_does_not_read_evaluator_acceptance(self):
        record = {"sequence": "seq", "frame_index": 0,
                  "online": {"terminal_action": "defer", "decisions": [], "acquisitions": []},
                  "evaluator": {"accepted": True, "pose": "must_not_be_read"}}
        self.assertEqual(trigger_reasons(record, TriggerConfig()), ["verifier_failure"])

    def test_episode_ends_on_clean_accept(self):
        episodes = mine_sequence("seq", [row(0, True), row(1, False), row(2, False),
                                                  row(3, True)], TriggerConfig())
        self.assertEqual(len(episodes), 1)
        self.assertEqual((episodes[0]["start_frame"], episodes[0]["end_frame"]), (1, 3))
        self.assertTrue(episodes[0]["observed_accept"])

    def test_trace_gap_censors_episode(self):
        episodes = mine_sequence("seq", [row(1, False), row(4, True)], TriggerConfig())
        self.assertEqual(episodes[0]["end_reason"], "trace_gap")
        self.assertTrue(episodes[0]["right_censored"])

    def test_one_scene_is_audit_only(self):
        manifest = build_manifest({"seq": [row(1, False), row(2, True)]},
                                  {"seq": {"scene_id": "same"}}, TriggerConfig(),
                                  seed=1, trace_paths=[])
        self.assertFalse(manifest["scene_disjoint_ready"])
        self.assertEqual(manifest["episodes"][0]["split"], "audit_only")

    def test_scene_split_has_no_scene_overlap(self):
        split = assign_scene_splits([f"s{i}" for i in range(10)], seed=7)
        self.assertEqual(set(split), {f"s{i}" for i in range(10)})
        self.assertTrue({"train", "validation", "test"}.issubset(set(split.values())))

    def test_episode_spans_include_causal_warmup(self):
        payload = {"episodes": [
            {"sequence": "a", "start_frame": 10, "end_frame": 12},
            {"sequence": "a", "start_frame": 20, "end_frame": 21},
            {"sequence": "b", "start_frame": 2, "end_frame": 3},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(load_episode_spans(path, 5), {"a": (5, 22), "b": (0, 4)})
            payload["episodes"][0]["episode_id"] = "only"
            path.write_text(json.dumps(payload))
            self.assertEqual(load_episode_spans(path, 2, ["only"]), {"a": (8, 13)})


if __name__ == "__main__":
    unittest.main()
