#!/usr/bin/env python3
"""Post-hoc evaluator for the no-GT AVL-004 Manager trace.

This process is intentionally separate from ``run_avl004_manager.py``: query
GT is loaded here only to score already frozen actions and pose outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical,
    pose_error,
    read_manifest,
    tum_camera_poses,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def evaluate_sequence(args, sequence):
    trace_path = args.manager_dir / sequence / "manager_trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    _, _, query_times, _, query_root = read_manifest(args.data_root, sequence)
    gt, _ = tum_camera_poses(args.gt_dir / f"{sequence}_gt_poses.txt", query_times,
                             base_to_front_rgb_optical(query_root), args.max_gt_delta)
    rows = []
    for index, item in enumerate(trace):
        pose = None if item["candidate_pose"] is None else np.asarray(item["candidate_pose"], dtype=np.float64)
        if pose is None:
            metrics = {"available": False, "success": False,
                       "translation_error_m": None, "rotation_error_deg": None}
        else:
            translation, rotation = pose_error(pose, gt[index])
            metrics = {"available": True, "success": bool(translation <= 0.25 and rotation <= 5.0),
                       "translation_error_m": float(translation),
                       "rotation_error_deg": float(rotation)}
        rows.append({"query_index": index, "action": item["action"], **metrics})

    total = len(rows)
    covered = [row for row in rows if row["available"]]
    summary = {
        "sequence": sequence,
        "frames": total,
        "coverage_rate": len(covered) / total if total else 0.0,
        "success_rate_all_frames": sum(row["success"] for row in rows) / total if total else 0.0,
        "success_rate_covered": sum(row["success"] for row in covered) / len(covered) if covered else 0.0,
        "median_translation_error_covered_m": float(np.median([r["translation_error_m"] for r in covered])) if covered else None,
        "median_rotation_error_covered_deg": float(np.median([r["rotation_error_deg"] for r in covered])) if covered else None,
        "actions": {},
        "selection_uses_gt": False,
        "gt_used_only_for_posthoc_evaluation": True,
    }
    for action in ("accept_consensus", "use_memory", "call_geometry", "reject"):
        action_rows = [row for row in rows if row["action"] == action]
        action_covered = [row for row in action_rows if row["available"]]
        summary["actions"][action] = {
            "frames": len(action_rows),
            "coverage_rate": len(action_covered) / len(action_rows) if action_rows else 0.0,
            "success_rate_all_action_frames": sum(row["success"] for row in action_rows) / len(action_rows) if action_rows else 0.0,
            "success_rate_covered": sum(row["success"] for row in action_covered) / len(action_covered) if action_covered else 0.0,
        }
    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluated_rows.jsonl").write_text(
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
