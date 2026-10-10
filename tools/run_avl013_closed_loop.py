#!/usr/bin/env python3
"""AVL-013 causal Reject -> recovery -> state update replay.

No query GT is opened here.  Existing Reloc3r/RoMa outputs are used as frozen
tool fixtures, while the state machine is rerun so a rejected pose cannot
update prediction or memory.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyse_lilocbench_failures import image_difference
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, read_manifest, tum_camera_poses
from tools.run_avl003_ablations import aggregate_selected, load_relative, rotation_distance, select_methods
from tools.run_avl004_manager import selected_dispersion
from tools.run_avl011_strong_recovery import pose_jump, verify_geometry, read_jsonl

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")

def ratio(a, b): return float(a) / float(b) if b else 0.0

def risk_score(cons_t, cons_r, top_t, top_r, pred_t, pred_r, appearance, frame_change,
               geometry_row=None, consensus_pose=None, memory_pose=None, previous=None):
    candidate = max(cons_t / 1.5, cons_r / 15.0, top_t / 3.0, top_r / 30.0)
    temporal = max(pred_t / 2.0, pred_r / 30.0, appearance / 0.20, frame_change / 0.20)
    geometry = 0.0
    if geometry_row and geometry_row.get("pose") is not None:
        inliers, corr = int(geometry_row.get("inliers", 0)), int(geometry_row.get("correspondences", 0))
        pairs = geometry_row.get("pair_stats") or []
        support = sum(int(p.get("intersections", 0)) >= 500 for p in pairs)
        gp = geometry_row["pose"]
        checks = []
        for other in (consensus_pose, memory_pose, previous):
            if other is not None:
                dt, dr = pose_jump(np.asarray(other), np.asarray(gp)); checks += [dt / 2.0, dr / 30.0]
        geometry = max(max(0.0, 1000.0 - inliers) / 1000.0,
                       float(geometry_row.get("median_reprojection_px") or 0.0) / 3.0,
                       1.0 - ratio(support, len(pairs)), *checks)
    return float(max(candidate, temporal, geometry)), float(candidate), float(temporal), float(geometry)

def run_sequence(args, sequence):
    t_sequence = time.perf_counter()
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, _ = read_manifest(args.data_root, sequence)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times, base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.relative_root / sequence, len(query_frames))
    recovery = read_jsonl(args.roma_root / f"{sequence}_top100_merged.jsonl")
    previous_prediction = None; previous_prediction_frame = None
    previous_memory = None; memory_reliable = False
    memory_hold = 0; reject_streak = 0; rows = []; stage_s = {"candidate_and_selection": 0.0, "verification_and_risk": 0.0, "state_update": 0.0, "cache_lookup": 0.0}
    for qi, item in enumerate(relative):
        t0 = time.perf_counter()
        mapping_indices, rel_poses = item["indices"], item["relative"]
        candidates = [mapping_pose[int(idx)] @ rel for idx, rel in zip(mapping_indices, rel_poses)]
        selections = select_methods(candidates, mapping_indices, rel_poses, previous_memory)
        ci, mi = selections["consensus3"]["indices"], selections["memory3"]["indices"]
        consensus_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, ci)
        memory_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, mi)
        top_t, top_r = selected_dispersion(candidates, list(range(len(candidates))))
        cons_t, cons_r = selected_dispersion(candidates, ci)
        mem_t, mem_r = selected_dispersion(candidates, mi)
        stale_state_recovery = reject_streak >= args.stale_state_after
        memory_reliable_before = memory_reliable
        state_before_frame = previous_prediction_frame
        pred_t, pred_r = (0.0, 0.0) if stale_state_recovery else pose_jump(previous_prediction, consensus_pose)
        qpath = Path(query_frames[qi]["path"]); ppath = Path(query_frames[qi-1]["path"]) if qi else None
        frame_change = image_difference(ppath, qpath) if ppath else 0.0
        appearance = image_difference(Path(db_frames[int(mapping_indices[0])]["path"]), qpath)
        stage_s["candidate_and_selection"] += time.perf_counter() - t0
        candidate_conflict = top_t > args.consensus_translation_m or top_r > args.consensus_rotation_deg
        consensus_reliable = cons_t <= args.consensus_translation_m and cons_r <= args.consensus_rotation_deg
        high_jump = pred_t > args.jump_translation_m or pred_r > args.jump_rotation_deg
        appearance_risk = appearance > args.appearance_change or frame_change > args.frame_change
        high_risk = candidate_conflict or high_jump or appearance_risk
        severe = candidate_conflict and args.geometry_on_conflict
        if severe or high_risk and not (memory_reliable and not candidate_conflict):
            action = "call_geometry"; candidate_pose = None; source = None
        elif memory_reliable and high_risk:
            action = "use_memory"; candidate_pose = memory_pose; source = "memory"
        elif consensus_reliable and not high_jump:
            action = "accept_consensus"; candidate_pose = consensus_pose; source = "consensus"
        elif high_risk:
            action = "call_geometry"; candidate_pose = None; source = None
        else:
            action = "reject"; candidate_pose = None; source = None
        t0 = time.perf_counter(); geometry_row = {}; verified = False; verification = None
        if action == "call_geometry":
            geometry_row = recovery.get(qi, {})
            verified, verification = verify_geometry(geometry_row, previous_prediction, args.geometry_min_inliers,
                args.geometry_max_reprojection_px, args.geometry_jump_m, args.geometry_jump_deg,
                args.min_pair_support, args.min_pair_intersections)
            if verified: candidate_pose, source = geometry_row["pose"], "geometry_recovery"
            elif consensus_reliable and not high_jump: candidate_pose, source = consensus_pose, "consensus_after_geometry_reject"
            elif memory_reliable and memory_hold < args.memory_hold_max:
                candidate_pose = memory_pose if mem_t <= args.memory_translation_m and mem_r <= args.memory_rotation_deg else previous_memory
                source = "memory_after_geometry_reject" if candidate_pose is not None else None
        lookup_elapsed = time.perf_counter() - t0
        stage_s["cache_lookup"] += lookup_elapsed
        score, candidate_risk, temporal_risk, geometry_risk = risk_score(cons_t, cons_r, top_t, top_r, pred_t, pred_r,
            appearance, frame_change, geometry_row if action == "call_geometry" else None,
            consensus_pose, memory_pose, None if stale_state_recovery else previous_prediction)
        accepted_before_reject = candidate_pose is not None
        if not accepted_before_reject or score > args.reliability_threshold:
            final_pose = None; final_source = "reject_risk" if accepted_before_reject else ("reject_geometry" if action == "call_geometry" else "reject")
            accepted = False; reject_streak += 1
        else:
            final_pose = np.asarray(candidate_pose, dtype=np.float64); final_source = source; accepted = True; reject_streak = 0
        stage_s["verification_and_risk"] += time.perf_counter() - t0
        state_before = previous_prediction is not None
        t0 = time.perf_counter()
        if accepted:
            previous_prediction = final_pose.copy()
            previous_prediction_frame = qi
            current_mem_reliable = mem_t <= args.memory_translation_m and mem_r <= args.memory_rotation_deg
            if final_source == "geometry_recovery": previous_memory, memory_reliable, memory_hold = final_pose.copy(), True, 0
            elif current_mem_reliable:
                previous_memory, memory_reliable = memory_pose, True
                memory_hold = memory_hold + 1 if final_source in ("memory", "memory_after_geometry_reject") else 0
            elif final_source in ("memory", "memory_after_geometry_reject"): memory_hold += 1
            else: memory_hold = 0
        stage_s["state_update"] += time.perf_counter() - t0
        rows.append({"query_index": qi, "timestamp": float(query_times[qi]), "action": action,
            "accepted": accepted, "pose": None if final_pose is None else final_pose.tolist(), "pose_source": final_source,
            "geometry_verified": bool(verified), "geometry_reference_indices": geometry_row.get("reference_indices"),
            "verification": verification, "risk_score": score, "candidate_risk": candidate_risk,
            "temporal_risk": temporal_risk, "geometry_risk": geometry_risk, "accepted_before_reject": accepted_before_reject,
            "state_before_available": state_before, "state_before_frame": state_before_frame,
            "state_updated": accepted, "state_updated_frame": qi if accepted else None,
            "reject_streak": reject_streak,
            "stale_state_recovery": stale_state_recovery,
            "evidence": {"top10_candidate_dispersion_m": top_t, "top10_candidate_dispersion_deg": top_r,
                "consensus_dispersion_m": cons_t, "consensus_dispersion_deg": cons_r,
                "memory_dispersion_m": mem_t, "memory_dispersion_deg": mem_r, "predicted_jump_m": pred_t,
                "predicted_jump_deg": pred_r, "appearance_change": appearance, "frame_change": frame_change,
                "memory_reliable_before": memory_reliable_before, "memory_reliable_after": memory_reliable,
                "candidate_conflict": candidate_conflict,
                "consensus_reliable": consensus_reliable, "high_risk": high_risk},
            "query_gt_read": False, "uses_future": False, "selection_uses_gt": False})
    out = args.output_root / sequence; out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_trace.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    summary = {"sequence": sequence, "frames": len(rows), "action_counts": {a: sum(x["action"] == a for x in rows) for a in ("accept_consensus","use_memory","call_geometry","reject")},
        "final_source_counts": {str(a): sum(x["pose_source"] == a for x in rows) for a in ("consensus","memory","geometry_recovery","consensus_after_geometry_reject","memory_after_geometry_reject","reject","reject_geometry","reject_risk")},
        "accepted_outputs": sum(x["accepted"] for x in rows), "max_reject_streak": max((x["reject_streak"] for x in rows), default=0),
        "reliability_threshold": args.reliability_threshold, "stage_wall_s": stage_s,
        "sequence_wall_s": time.perf_counter() - t_sequence, "query_gt_read": False, "uses_future": False, "selection_uses_gt": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, default=Path('data/lilocbench/prepared_1hz')); p.add_argument('--relative-root', type=Path, default=Path('outputs/lilocbench/AVL-002')); p.add_argument('--mapping-gt', type=Path, default=Path('data/lilocbench/gt/mapping_gt_poses.txt')); p.add_argument('--roma-root', type=Path, default=Path('outputs/lilocbench/AVL-004/always_geometry/roma')); p.add_argument('--output-root', type=Path, default=Path('outputs/lilocbench/AVL-013/closed_loop_v2_t1.5')); p.add_argument('--max-gt-delta', type=float, default=0.1); p.add_argument('--reliability-threshold', type=float, default=1.5)
    p.add_argument('--consensus-translation-m', type=float, default=1.5); p.add_argument('--consensus-rotation-deg', type=float, default=15.0); p.add_argument('--memory-translation-m', type=float, default=1.5); p.add_argument('--memory-rotation-deg', type=float, default=15.0); p.add_argument('--jump-translation-m', type=float, default=1.0); p.add_argument('--jump-rotation-deg', type=float, default=20.0); p.add_argument('--appearance-change', type=float, default=0.15); p.add_argument('--frame-change', type=float, default=0.15); p.add_argument('--geometry-min-inliers', type=int, default=50); p.add_argument('--geometry-max-reprojection-px', type=float, default=3.0); p.add_argument('--geometry-jump-m', type=float, default=100.0); p.add_argument('--geometry-jump-deg', type=float, default=180.0); p.add_argument('--min-pair-support', type=int, default=3); p.add_argument('--min-pair-intersections', type=int, default=500); p.add_argument('--memory-hold-max', type=int, default=3); p.add_argument('--stale-state-after', type=int, default=3); p.add_argument('--geometry-on-conflict', action='store_true', default=True); p.add_argument('--sequences', nargs='+', default=list(SEQUENCES))
    args = p.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True); reports = [run_sequence(args, s) for s in args.sequences]
    (args.output_root / 'run_summary.json').write_text(json.dumps({'experiment':'AVL-013','threshold':args.reliability_threshold,'sequences':reports,'query_gt_read':False,'uses_future':False,'selection_uses_gt':False}, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
if __name__ == '__main__': main()
