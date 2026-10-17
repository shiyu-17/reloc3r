#!/usr/bin/env python3
"""Assemble the four-condition, provenance-preserving AVL summary."""
import json
from pathlib import Path

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]


def main():
    out = Path("outputs/lilocbench/AVL-006/four_condition_summary.json")
    baseline = {}
    ablations = {x["sequence"]: x for x in json.loads(Path("outputs/lilocbench/AVL-003/summary_all.json").read_text())}
    manager = json.loads(Path("outputs/lilocbench/AVL-005/closed_loop_manager/route_eval.json").read_text())["sequences"]
    always = {s: json.loads(Path(f"outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0/{s}.json").read_text())["summary"]["route"] for s in SEQUENCES}
    early = json.loads(Path("outputs/lilocbench/AVL-006/early_router_eval.json").read_text())["policies"]
    for seq in SEQUENCES:
        avl2 = json.loads(Path(f"outputs/lilocbench/AVL-002/{seq}/summary.json").read_text())
        methods = ablations[seq]["methods"]
        ar = always[seq]
        baseline[seq] = {
            "frames": avl2["frames"],
            "reloc3r_top10": {
                "success_pct": 100 * avl2["success_rate"]["0.25m_5deg"],
                "median_translation_m": avl2["translation_error_m"]["median"],
                "median_rotation_deg": avl2["rotation_error_deg"]["median"],
                "max_failure_frames": avl2["failure_runs"]["max_frames"],
                "jump_rate_pct": 100 * avl2["trajectory_jump"]["rate_gt_1m_or_20deg"],
                "candidate_conflict_m": avl2["candidate_conflict"]["median_translation_m"],
                "candidate_conflict_deg": avl2["candidate_conflict"]["median_rotation_deg"],
            },
            "consensus3": {"success_pct": 100 * methods["consensus3"]["success_rate_0.25m_5deg"], "max_failure_frames": methods["consensus3"]["max_failure_frames"]},
            "memory3": {"success_pct": 100 * methods["memory3"]["success_rate_0.25m_5deg"], "max_failure_frames": methods["memory3"]["max_failure_frames"]},
            "always_roma_seed0": {
                "success_pct": 100 * ar["success_all_frames"],
                "success_count": round(ar["success_all_frames"] * ar["frames"]),
                "coverage_pct": 100 * ar["coverage"],
            },
            "closed_loop_manager": {
                "success_pct": 100 * manager[seq]["success_all_frames"],
                "coverage_pct": 100 * manager[seq]["coverage"],
                "geometry_calls": manager[seq]["geometry_calls"],
            },
            "early_router": {
                p: {
                    "roma_call_rate_pct": 100 * v["per_sequence"][seq]["roma_call_rate"],
                    "success_pct": 100 * v["per_sequence"][seq]["success"],
                    "success_count": v["per_sequence"][seq]["success_count"],
                } for p, v in early.items()
            },
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": "Four LILocBench query conditions; GT only in independent evaluator artifacts; no held-out claim",
        "sequence_order": SEQUENCES,
        "conditions": baseline,
        "sources": {
            "AVL-002": "outputs/lilocbench/AVL-002/*/summary.json",
            "AVL-003": "outputs/lilocbench/AVL-003/summary_all.json",
            "AVL-005_always_roma_seed0": "outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0/summary.json",
            "AVL-005_closed_loop": "outputs/lilocbench/AVL-005/closed_loop_manager/summary_all.json",
            "AVL-006_early_router": "outputs/lilocbench/AVL-006/early_router_eval.json",
        },
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
