#!/usr/bin/env python3
"""Replay a deterministic, no-GT AVL-004 Manager over frozen candidates.

This is deliberately a dry-run: ``call_geometry`` emits a trigger and no
pose until the HLoc/RoMa adapter is available.  The manager never loads query
GT.  Mapping poses are map assets, while all query-time evidence is computed
from the current image, frozen candidates, and state from earlier frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyse_lilocbench_failures import image_difference  # noqa: E402
from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical,
    read_manifest,
    tum_camera_poses,
)
from tools.run_avl003_ablations import (  # noqa: E402
    aggregate_selected,
    candidate_distances,
    load_relative,
    rotation_distance,
    select_methods,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def pose_jump(previous, current):
    if previous is None:
        return 0.0, 0.0
    return (float(np.linalg.norm(previous[:3, 3] - current[:3, 3])),
            float(rotation_distance(previous, current)))


def selected_dispersion(candidates, indices):
    chosen = [candidates[int(i)] for i in indices]
    if len(chosen) < 2:
        return 0.0, 0.0
    values = candidate_distances(chosen)
    # candidate_distances is normalized (m + deg/10), so calculate the two
    # interpretable components explicitly for the trace.
    trans, rots = [], []
    for first in range(len(chosen)):
        for second in range(first + 1, len(chosen)):
            trans.append(float(np.linalg.norm(chosen[first][:3, 3] - chosen[second][:3, 3])))
            rots.append(float(rotation_distance(chosen[first], chosen[second])))
    return float(np.median(trans)), float(np.median(rots))


def analyse_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, sequence)
    # Mapping poses are fixed map assets. No query GT is read in this script.
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times,
                                       base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.results_root / sequence, len(query_frames))
    previous_prediction = None
    previous_memory = None
    previous_memory_reliable = False
    trace = []
    for query_index, item in enumerate(relative):
        mapping_indices = item["indices"]
        rel_poses = item["relative"]
        candidates = [mapping_pose[int(idx)] @ rel for idx, rel in zip(mapping_indices, rel_poses)]
        selections = select_methods(candidates, mapping_indices, rel_poses, previous_memory)
        consensus_indices = selections["consensus3"]["indices"]
        memory_indices = selections["memory3"]["indices"]
        consensus_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, consensus_indices)
        memory_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, memory_indices)
        top10_t, top10_r = selected_dispersion(candidates, list(range(len(candidates))))
        consensus_t, consensus_r = selected_dispersion(candidates, consensus_indices)
        memory_t, memory_r = selected_dispersion(candidates, memory_indices)
        pred_jump_t, pred_jump_r = pose_jump(previous_prediction, consensus_pose)
        query_path = Path(query_frames[query_index]["path"])
        previous_path = Path(query_frames[query_index - 1]["path"]) if query_index else None
        frame_change = image_difference(previous_path, query_path) if previous_path else 0.0
        top1_path = Path(db_frames[int(mapping_indices[0])]["path"])
        appearance_change = image_difference(top1_path, query_path)

        candidate_conflict = top10_t > args.consensus_translation_m or top10_r > args.consensus_rotation_deg
        consensus_reliable = consensus_t <= args.consensus_translation_m and consensus_r <= args.consensus_rotation_deg
        high_jump = pred_jump_t > args.jump_translation_m or pred_jump_r > args.jump_rotation_deg
        appearance_risk = appearance_change > args.appearance_change or frame_change > args.frame_change
        high_risk = candidate_conflict or high_jump or appearance_risk
        memory_reliable_before = previous_memory_reliable

        # Priority is intentionally fixed before evaluation. A simultaneous
        # candidate conflict and appearance/motion alarm is escalated first;
        # otherwise a known reliable temporal state may bridge a milder risk.
        severe_geometry_trigger = candidate_conflict and (appearance_risk or high_jump)
        if severe_geometry_trigger:
            action = "call_geometry"
            action_pose = None
        elif previous_memory_reliable and high_risk:
            action = "use_memory"
            action_pose = memory_pose
        elif consensus_reliable and not high_jump:
            action = "accept_consensus"
            action_pose = consensus_pose
        elif high_risk:
            action = "call_geometry"
            action_pose = None
        else:
            action = "reject"
            action_pose = None

        current_memory_reliable = memory_t <= args.memory_translation_m and memory_r <= args.memory_rotation_deg
        # Only an internally coherent result can update memory. The state is
        # updated after the decision and is never filtered by GT.
        if current_memory_reliable:
            previous_memory = memory_pose
            previous_memory_reliable = True
        if action_pose is not None:
            previous_prediction = action_pose

        trace.append({
            "query_index": query_index,
            "timestamp": float(query_times[query_index]),
            "action": action,
            "evidence": {
                "top10_candidate_dispersion_m": top10_t,
                "top10_candidate_dispersion_deg": top10_r,
                "consensus_dispersion_m": consensus_t,
                "consensus_dispersion_deg": consensus_r,
                "memory_dispersion_m": memory_t,
                "memory_dispersion_deg": memory_r,
                "predicted_jump_m": pred_jump_t,
                "predicted_jump_deg": pred_jump_r,
                "appearance_change": float(appearance_change),
                "frame_change": float(frame_change),
                "memory_reliable_before": bool(memory_reliable_before),
                "memory_reliable_current": bool(current_memory_reliable),
                "candidate_conflict": bool(candidate_conflict),
                "consensus_reliable": bool(consensus_reliable),
                "high_risk": bool(high_risk),
                "severe_geometry_trigger": bool(severe_geometry_trigger),
            },
            "selected_candidate_positions": {
                "consensus3": [int(x) for x in consensus_indices],
                "memory3": [int(x) for x in memory_indices],
            },
            "candidate_pose": None if action_pose is None else action_pose.tolist(),
            # Keep the two no-GT candidate streams for post-hoc risk--coverage
            # auditing.  These are evidence-derived candidates, not GT inputs;
            # ``candidate_pose`` remains the action's actual output.
            "consensus_pose": consensus_pose.tolist(),
            "memory_pose": memory_pose.tolist(),
            "geometry_route": {
                "reference_policy": args.geometry_reference_policy,
                "reference_budget": 100 if args.geometry_reference_policy == "top100_lightglue_rerank" else 10,
                "tool_candidates": ["hloc_style_superpoint_lightglue", "roma_indoor"],
            } if action == "call_geometry" else None,
            "selection_uses_gt": False,
            "test_gt_read": False,
            "forbidden_query_fields": [],
        })
    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "manager_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace), encoding="utf-8")
    counts = {name: sum(row["action"] == name for row in trace)
              for name in ("accept_consensus", "use_memory", "call_geometry", "reject")}
    summary = {
        "sequence": sequence,
        "frames": len(trace),
        "action_counts": counts,
        "geometry_call_rate": counts["call_geometry"] / len(trace) if trace else 0.0,
        "coverage_rate": sum(row["candidate_pose"] is not None for row in trace) / len(trace) if trace else 0.0,
        "thresholds": {
            "consensus_translation_m": args.consensus_translation_m,
            "consensus_rotation_deg": args.consensus_rotation_deg,
            "memory_translation_m": args.memory_translation_m,
            "memory_rotation_deg": args.memory_rotation_deg,
            "jump_translation_m": args.jump_translation_m,
            "jump_rotation_deg": args.jump_rotation_deg,
            "appearance_change": args.appearance_change,
            "frame_change": args.frame_change,
            "geometry_reference_policy": args.geometry_reference_policy,
        },
        "selection_uses_gt": False,
        "test_gt_read": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mapping-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gt-delta", type=float, default=0.1)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--consensus-translation-m", type=float, default=1.5)
    parser.add_argument("--consensus-rotation-deg", type=float, default=15.0)
    parser.add_argument("--memory-translation-m", type=float, default=1.5)
    parser.add_argument("--memory-rotation-deg", type=float, default=15.0)
    parser.add_argument("--jump-translation-m", type=float, default=1.0)
    parser.add_argument("--jump-rotation-deg", type=float, default=20.0)
    parser.add_argument("--appearance-change", type=float, default=0.15)
    parser.add_argument("--frame-change", type=float, default=0.15)
    parser.add_argument("--geometry-reference-policy", default="top100_lightglue_rerank",
                        choices=("top10_frozen", "top100_lightglue_rerank"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analyse_sequence(args, sequence) for sequence in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
