#!/usr/bin/env python3
"""Causal deterministic recovery baseline for AVL-011.

This is a stronger, still non-Agent replay of
Reloc3r -> reliability -> Memory -> RoMa -> verification -> Reject.  The
geometry cache is only an input fixture; action selection and verification do
not inspect query GT or future frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
    load_relative,
    rotation_distance,
    select_methods,
)
from tools.run_avl004_manager import selected_dispersion  # noqa: E402


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def read_jsonl(path: Path):
    return {int(row["query_index"]): row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def pose_jump(previous, current):
    if previous is None:
        return 0.0, 0.0
    return (float(np.linalg.norm(previous[:3, 3] - current[:3, 3])),
            float(rotation_distance(previous, current)))


def verify_geometry(row, previous_prediction, min_inliers, max_reprojection_px,
                    max_jump_m, max_jump_deg, min_pair_support, min_pair_intersections):
    pose = row.get("pose")
    reproj = row.get("median_reprojection_px")
    basic = (pose is not None and int(row.get("inliers", 0)) >= min_inliers and
             reproj is not None and float(reproj) <= max_reprojection_px)
    pair_stats = row.get("pair_stats") or []
    pair_support = sum(int(x.get("intersections", 0)) >= min_pair_intersections for x in pair_stats)
    support_ok = pair_support >= min_pair_support if pair_stats else True
    jump_m = jump_deg = 0.0
    jump_ok = True
    if basic and previous_prediction is not None:
        candidate = np.asarray(pose, dtype=np.float64)
        jump_m, jump_deg = pose_jump(previous_prediction, candidate)
        jump_ok = jump_m <= max_jump_m and jump_deg <= max_jump_deg
    accepted = bool(basic and support_ok and jump_ok)
    return accepted, {
        "pose_available": pose is not None,
        "inliers": int(row.get("inliers", 0)),
        "median_reprojection_px": None if reproj is None else float(reproj),
        "pair_support": int(pair_support),
        "min_pair_support": int(min_pair_support),
        "min_pair_intersections": int(min_pair_intersections),
        "jump_from_previous_m": float(jump_m),
        "jump_from_previous_deg": float(jump_deg),
        "max_jump_m": float(max_jump_m),
        "max_jump_deg": float(max_jump_deg),
        "basic_ok": bool(basic),
        "support_ok": bool(support_ok),
        "jump_ok": bool(jump_ok),
    }


def run_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, sequence)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times,
                                       base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.results_root / sequence, len(query_frames))
    if args.limit_fraction < 1.0:
        limit = max(1, int(math.ceil(len(query_frames) * args.limit_fraction)))
        query_frames = query_frames[:limit]
        query_times = query_times[:limit]
        relative = relative[:limit]
    recovery = read_jsonl(args.recovery_root / f"{sequence}_top100_merged.jsonl")

    previous_prediction = None
    previous_memory = None
    previous_memory_reliable = False
    memory_hold_streak = 0
    rows = []
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

        severe_geometry_trigger = candidate_conflict and (args.geometry_on_conflict or appearance_risk or high_jump)
        if severe_geometry_trigger:
            action = "call_geometry"
            candidate_pose = None
            candidate_source = None
        elif previous_memory_reliable and high_risk:
            action = "use_memory"
            candidate_pose = memory_pose
            candidate_source = "memory"
        elif consensus_reliable and not high_jump:
            action = "accept_consensus"
            candidate_pose = consensus_pose
            candidate_source = "consensus"
        elif high_risk:
            action = "call_geometry"
            candidate_pose = None
            candidate_source = None
        else:
            action = "reject"
            candidate_pose = None
            candidate_source = None

        geometry_row = recovery.get(query_index, {})
        geometry_verified = False
        verification = None
        output_pose = candidate_pose
        output_source = candidate_source
        if action == "call_geometry":
            geometry_verified, verification = verify_geometry(
                geometry_row, previous_prediction, args.geometry_min_inliers,
                args.geometry_max_reprojection_px, args.geometry_jump_m,
                args.geometry_jump_deg, args.min_pair_support,
                args.min_pair_intersections)
            if geometry_verified:
                output_pose = geometry_row["pose"]
                output_source = "geometry"
            elif args.fallback_consensus and consensus_reliable and not high_jump:
                output_pose = consensus_pose
                output_source = "consensus_after_geometry_reject"
            elif memory_reliable_before and memory_hold_streak < args.memory_hold_max:
                output_pose = memory_pose if memory_t <= args.memory_translation_m and memory_r <= args.memory_rotation_deg else previous_memory
                output_source = "memory_after_geometry_reject" if output_pose is not None else "reject_geometry"
            else:
                output_pose = None
                output_source = "reject_geometry"

        accepted = output_pose is not None
        if accepted:
            previous_prediction = np.asarray(output_pose, dtype=np.float64)
        current_memory_reliable = memory_t <= args.memory_translation_m and memory_r <= args.memory_rotation_deg
        if output_source == "geometry" and geometry_verified:
            previous_memory = previous_prediction.copy()
            previous_memory_reliable = True
            memory_hold_streak = 0
        elif accepted and current_memory_reliable:
            previous_memory = memory_pose
            previous_memory_reliable = True
            memory_hold_streak = 0 if output_source not in ("memory", "memory_after_geometry_reject") else memory_hold_streak + 1
        elif output_source in ("memory", "memory_after_geometry_reject"):
            memory_hold_streak += 1
        else:
            memory_hold_streak = 0

        rows.append({
            "query_index": query_index,
            "timestamp": float(query_times[query_index]),
            "action": action,
            "accepted": bool(accepted),
            "pose_source": output_source,
            "pose": None if output_pose is None else np.asarray(output_pose).tolist(),
            "geometry_verified": bool(geometry_verified),
            "geometry_reference_indices": geometry_row.get("reference_indices"),
            "verification": verification,
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
                "memory_reliable_after": bool(previous_memory_reliable),
                "memory_hold_streak": int(memory_hold_streak),
                "candidate_conflict": bool(candidate_conflict),
                "consensus_reliable": bool(consensus_reliable),
                "high_risk": bool(high_risk),
                "severe_geometry_trigger": bool(severe_geometry_trigger),
            },
            "selection_uses_gt": False,
            "query_gt_read": False,
            "uses_future": False,
        })

    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "sequence": sequence,
        "frames": len(rows),
        "action_counts": {name: sum(row["action"] == name for row in rows)
                          for name in ("accept_consensus", "use_memory", "call_geometry", "reject")},
        "accepted_outputs": sum(row["accepted"] for row in rows),
        "geometry_verified": sum(row["geometry_verified"] for row in rows),
        "thresholds": vars(args),
        "query_gt_read": False,
        "selection_uses_gt": False,
        "uses_future": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--mapping-gt", type=Path, required=True)
    p.add_argument("--recovery-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    p.add_argument("--consensus-translation-m", type=float, default=1.5)
    p.add_argument("--consensus-rotation-deg", type=float, default=15.0)
    p.add_argument("--memory-translation-m", type=float, default=1.5)
    p.add_argument("--memory-rotation-deg", type=float, default=15.0)
    p.add_argument("--jump-translation-m", type=float, default=1.0)
    p.add_argument("--jump-rotation-deg", type=float, default=20.0)
    p.add_argument("--appearance-change", type=float, default=0.15)
    p.add_argument("--frame-change", type=float, default=0.15)
    p.add_argument("--geometry-min-inliers", type=int, default=50)
    p.add_argument("--geometry-max-reprojection-px", type=float, default=3.0)
    p.add_argument("--geometry-jump-m", type=float, default=2.5)
    p.add_argument("--geometry-jump-deg", type=float, default=60.0)
    p.add_argument("--min-pair-support", type=int, default=3)
    p.add_argument("--min-pair-intersections", type=int, default=500)
    p.add_argument("--memory-hold-max", type=int, default=3)
    p.add_argument("--fallback-consensus", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geometry-on-conflict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit-fraction", type=float, default=1.0,
                   help="run only the causal prefix; 0.4 is the frozen validation prefix")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_sequence(args, seq) for seq in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
