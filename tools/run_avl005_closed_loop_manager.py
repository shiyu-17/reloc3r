#!/usr/bin/env python3
"""Recovery-aware deterministic Manager replay for AVL-005.

The recovery JSONL is treated as a no-GT tool cache produced before this
replay.  At runtime the Manager only sees the current Reloc3r candidates,
past state, and the current recovery result.  A verified geometry pose is
written into previous_prediction and memory before the next frame, so later
decisions are genuinely state-dependent.  Query GT is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    candidate_distances,
    load_relative,
    rotation_distance,
    select_methods,
)
from tools.run_avl004_manager import selected_dispersion  # noqa: E402


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def pose_jump(previous, current):
    if previous is None:
        return 0.0, 0.0
    return (float(np.linalg.norm(previous[:3, 3] - current[:3, 3])),
            float(rotation_distance(previous, current)))


def read_jsonl(path: Path):
    return {int(row["query_index"]): row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def verify_geometry(row, min_inliers, max_reprojection_px):
    """GT-free acceptance gate for a geometry candidate."""
    pose = row.get("pose")
    reproj = row.get("median_reprojection_px")
    accepted = (pose is not None and int(row.get("inliers", 0)) >= min_inliers and
                reproj is not None and float(reproj) <= max_reprojection_px)
    return accepted, {
        "pose_available": pose is not None,
        "inliers": int(row.get("inliers", 0)),
        "median_reprojection_px": None if reproj is None else float(reproj),
        "min_inliers": min_inliers,
        "max_reprojection_px": max_reprojection_px,
    }


def run_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, sequence)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times,
                                       base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.results_root / sequence, len(query_frames))
    recovery = read_jsonl(args.recovery_root / f"{sequence}_top100_merged.jsonl")
    previous_prediction = None
    previous_memory = None
    previous_memory_reliable = False
    rows = []
    stage_totals = {name: 0.0 for name in
                    ("reloc3r_candidate_processing", "memory", "verification", "geometry")}

    for query_index, item in enumerate(relative):
        frame_start = time.perf_counter()
        t0 = time.perf_counter()
        mapping_indices = item["indices"]
        rel_poses = item["relative"]
        candidates = [mapping_pose[int(idx)] @ rel for idx, rel in zip(mapping_indices, rel_poses)]
        t_candidate = time.perf_counter() - t0

        t0 = time.perf_counter()
        selections = select_methods(candidates, mapping_indices, rel_poses, previous_memory)
        consensus_indices = selections["consensus3"]["indices"]
        memory_indices = selections["memory3"]["indices"]
        consensus_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, consensus_indices)
        memory_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, memory_indices)
        top10_t, top10_r = selected_dispersion(candidates, list(range(len(candidates))))
        consensus_t, consensus_r = selected_dispersion(candidates, consensus_indices)
        memory_t, memory_r = selected_dispersion(candidates, memory_indices)
        t_memory = time.perf_counter() - t0

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

        severe_geometry_trigger = candidate_conflict and (appearance_risk or high_jump)
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

        t_verify = time.perf_counter()
        geometry_row = recovery.get(query_index, {})
        geometry_verified = False
        verification = None
        output_pose = candidate_pose
        output_source = candidate_source
        if action == "call_geometry":
            geometry_verified, verification = verify_geometry(
                geometry_row, args.geometry_min_inliers, args.geometry_max_reprojection_px)
            if geometry_verified:
                output_pose = geometry_row["pose"]
                output_source = "geometry"
            else:
                output_pose = None
                output_source = "reject_geometry"
        t_verification = time.perf_counter() - t_verify

        # The state update is the defining difference from A.5.  Only an
        # accepted output can affect the next-frame prediction.  Memory is
        # updated from the internally coherent candidate or verified geometry.
        accepted = output_pose is not None
        if accepted:
            previous_prediction = np.asarray(output_pose, dtype=np.float64)
        current_memory_reliable = memory_t <= args.memory_translation_m and memory_r <= args.memory_rotation_deg
        if output_source == "geometry" and geometry_verified:
            previous_memory = previous_prediction.copy()
            previous_memory_reliable = True
        elif current_memory_reliable:
            previous_memory = memory_pose
            previous_memory_reliable = True

        geometry_elapsed = float(geometry_row.get("elapsed_s", 0.0)) if action == "call_geometry" else 0.0
        stage = {
            "reloc3r_candidate_processing_s": t_candidate,
            "memory_s": t_memory,
            "verification_s": t_verification,
            "geometry_s": geometry_elapsed,
            "manager_wall_s": time.perf_counter() - frame_start,
        }
        stage_totals["reloc3r_candidate_processing"] += t_candidate
        stage_totals["memory"] += t_memory
        stage_totals["verification"] += t_verification
        stage_totals["geometry"] += geometry_elapsed
        rows.append({
            "query_index": query_index,
            "timestamp": float(query_times[query_index]),
            "action": action,
            "accepted": accepted,
            "pose_source": output_source,
            "pose": None if output_pose is None else np.asarray(output_pose).tolist(),
            "geometry_verified": geometry_verified,
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
                "candidate_conflict": bool(candidate_conflict),
                "consensus_reliable": bool(consensus_reliable),
                "high_risk": bool(high_risk),
                "severe_geometry_trigger": bool(severe_geometry_trigger),
            },
            "state_after": {
                "previous_prediction_available": previous_prediction is not None,
                "memory_reliable": bool(previous_memory_reliable),
            },
            "timings": stage,
            "selection_uses_gt": False,
            "query_gt_read": False,
        })

    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    counts = {name: sum(row["action"] == name for row in rows)
              for name in ("accept_consensus", "use_memory", "call_geometry", "reject")}
    accepted = sum(row["accepted"] for row in rows)
    summary = {
        "sequence": sequence,
        "frames": len(rows),
        "action_counts": counts,
        "accepted_outputs": accepted,
        "coverage_rate": accepted / len(rows) if rows else 0.0,
        "stage_totals_s": stage_totals,
        "geometry_verified": sum(row["geometry_verified"] for row in rows),
        "thresholds": {
            "consensus_translation_m": args.consensus_translation_m,
            "consensus_rotation_deg": args.consensus_rotation_deg,
            "memory_translation_m": args.memory_translation_m,
            "memory_rotation_deg": args.memory_rotation_deg,
            "jump_translation_m": args.jump_translation_m,
            "jump_rotation_deg": args.jump_rotation_deg,
            "appearance_change": args.appearance_change,
            "frame_change": args.frame_change,
            "geometry_min_inliers": args.geometry_min_inliers,
            "geometry_max_reprojection_px": args.geometry_max_reprojection_px,
        },
        "query_gt_read": False,
        "selection_uses_gt": False,
        "recovery_cache_protocol": "no-GT recovery cache; pose is consumed only if this frame calls geometry",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--mapping-gt", type=Path, required=True)
    p.add_argument("--recovery-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    p.add_argument("--max-gt-delta", type=float, default=0.1,
                   help="mapping map-asset synchronization only; query GT is never read")
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
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_sequence(args, sequence) for sequence in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
