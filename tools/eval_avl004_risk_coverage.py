#!/usr/bin/env python3
"""Evaluate the frozen no-GT Manager evidence as a selective predictor.

The trace is produced without query GT.  GT is loaded only here, after the
Manager decisions are frozen, to score the action output and the two stored
candidate streams.  The risk score and thresholds are fixed by the AVL-004
A.4 protocol; they are not fitted to these labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical,
    pose_error,
    read_manifest,
    tum_camera_poses,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
RISK_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))


def risk_score(evidence):
    """A fixed, dimensionless evidence risk; larger means less trustworthy."""
    terms = [
        float(evidence["top10_candidate_dispersion_m"]) / 1.5,
        float(evidence["top10_candidate_dispersion_deg"]) / 15.0,
        float(evidence["consensus_dispersion_m"]) / 1.5,
        float(evidence["consensus_dispersion_deg"]) / 15.0,
        float(evidence["predicted_jump_m"]) / 1.0,
        float(evidence["predicted_jump_deg"]) / 20.0,
        float(evidence["appearance_change"]) / 0.15,
        float(evidence["frame_change"]) / 0.15,
        1.0 if not evidence["memory_reliable_before"] else 0.0,
    ]
    return float(max(terms))


def score_pose(pose, gt):
    if pose is None:
        return {"available": False, "success": False,
                "translation_error_m": None, "rotation_error_deg": None}
    t, r = pose_error(np.asarray(pose, dtype=np.float64), gt)
    return {"available": True, "success": bool(t <= 0.25 and r <= 5.0),
            "translation_error_m": float(t), "rotation_error_deg": float(r)}


def summarize(rows, key, denominator=None):
    selected = [row for row in rows if row[key]["available"]]
    total = len(rows) if denominator is None else int(denominator)
    return {
        "frames_selected": len(selected),
        "coverage_rate": len(selected) / total if total else 0.0,
        "success_rate_all_frames": sum(row[key]["success"] for row in selected) / total if total else 0.0,
        "success_rate_selected": sum(row[key]["success"] for row in selected) / len(selected) if selected else 0.0,
        "median_translation_error_m": float(np.median([row[key]["translation_error_m"] for row in selected])) if selected else None,
        "median_rotation_error_deg": float(np.median([row[key]["rotation_error_deg"] for row in selected])) if selected else None,
        "false_accept_rate_selected": (sum(not row[key]["success"] for row in selected) / len(selected)) if selected else 0.0,
    }


def evaluate_sequence(args, sequence):
    trace_path = args.manager_dir / sequence / "manager_trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    _, _, query_times, _, query_root = read_manifest(args.data_root, sequence)
    gt, _ = tum_camera_poses(
        args.gt_dir / f"{sequence}_gt_poses.txt", query_times,
        base_to_front_rgb_optical(query_root), args.max_gt_delta,
    )
    rows = []
    for i, item in enumerate(trace):
        evidence = item["evidence"]
        risk = risk_score(evidence)
        row = {
            "query_index": i,
            "action": item["action"],
            "risk_score": risk,
            "evidence": evidence,
            "action_output": score_pose(item.get("candidate_pose"), gt[i]),
            "consensus_candidate": score_pose(item.get("consensus_pose"), gt[i]),
            "memory_candidate": score_pose(item.get("memory_pose"), gt[i]),
        }
        rows.append(row)

    thresholds = []
    for threshold in RISK_THRESHOLDS:
        selected = [row for row in rows if row["risk_score"] <= threshold]
        entry = {"threshold": threshold,
                 "selected": summarize(selected, "consensus_candidate", denominator=len(rows))}
        entry["selected"]["action_counts"] = {
            action: sum(row["action"] == action for row in selected)
            for action in ("accept_consensus", "use_memory", "call_geometry", "reject")
        }
        thresholds.append(entry)

    actual = summarize(rows, "action_output")
    action_stats = {}
    for action in ("accept_consensus", "use_memory", "call_geometry", "reject"):
        action_rows = [row for row in rows if row["action"] == action]
        action_stats[action] = {
            "frames": len(action_rows),
            "action_output": summarize(action_rows, "action_output"),
            "consensus_candidate": summarize(action_rows, "consensus_candidate"),
            "memory_candidate": summarize(action_rows, "memory_candidate"),
        }

    summary = {
        "sequence": sequence,
        "frames": len(rows),
        "actual_manager_output": actual,
        "action_stats": action_stats,
        "risk_coverage": thresholds,
        "risk_definition": {
            "score": "max(top10_t/1.5, top10_r/15, consensus_t/1.5, consensus_r/15, jump_t/1, jump_r/20, appearance/0.15, frame/0.15, 1-memory_unreliable_before)",
            "thresholds": list(RISK_THRESHOLDS),
            "fit_to_gt": False,
        },
        "selection_uses_gt": False,
        "gt_used_only_for_posthoc_evaluation": True,
    }
    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "risk_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--manager-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gt-delta", type=float, default=0.1)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [evaluate_sequence(args, sequence) for sequence in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
