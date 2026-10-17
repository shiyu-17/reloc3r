#!/usr/bin/env python3
"""AVL-012: evidence-only reliability scores and reject calibration.

The score branch deliberately does not load query ground truth.  It replays
the already frozen AVL-011 trace and recomputes only map-side candidate and
geometry evidence.  A separate evaluator loads query GT after scores and
thresholds have been written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, read_manifest, tum_camera_poses  # noqa: E402
from tools.run_avl003_ablations import aggregate_selected, load_relative, rotation_distance, select_methods  # noqa: E402
from tools.run_avl004_manager import selected_dispersion  # noqa: E402

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def read_jsonl(path: Path):
    return {int(x["query_index"]): x for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def pose_distance(a, b):
    if a is None or b is None:
        return 0.0, 0.0
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa[:3, 3] - bb[:3, 3])), float(rotation_distance(aa, bb))


def safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def build_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, _, _, _ = read_manifest(args.data_root, sequence)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times,
                                       base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.relative_root / sequence, len(query_frames))
    trace = read_jsonl(args.trace_root / sequence / "closed_loop_trace.jsonl")
    roma = read_jsonl(args.roma_root / f"{sequence}_top100_merged.jsonl")
    previous_memory = None
    previous_prediction = None
    rows = []
    for qi, item in enumerate(relative):
        tr = trace[qi]
        mapping_indices = item["indices"]
        rel_poses = item["relative"]
        candidates = [mapping_pose[int(idx)] @ rel for idx, rel in zip(mapping_indices, rel_poses)]
        selections = select_methods(candidates, mapping_indices, rel_poses, previous_memory)
        ci = selections["consensus3"]["indices"]
        mi = selections["memory3"]["indices"]
        consensus_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, ci)
        memory_pose = aggregate_selected(mapping_pose[mapping_indices], rel_poses, mi)
        top10_t, top10_r = selected_dispersion(candidates, list(range(len(candidates))))
        cons_t, cons_r = selected_dispersion(candidates, ci)
        mem_t, mem_r = selected_dispersion(candidates, mi)
        ev = tr.get("evidence", {})
        # Geometry evidence is only available when the causal route actually
        # requested geometry.  The old AVL-012 replay read Always-RoMa rows for
        # every frame, which made its threshold non-deployable under a selective
        # geometry budget.
        geometry_requested = tr.get("action") == "call_geometry"
        geom = roma.get(qi, {}) if geometry_requested else {}
        gp = geom.get("pose")
        geom_t_cons, geom_r_cons = pose_distance(gp, consensus_pose)
        geom_t_mem, geom_r_mem = pose_distance(gp, memory_pose)
        geom_t_prev, geom_r_prev = pose_distance(gp, previous_prediction)
        pair_stats = geom.get("pair_stats") or []
        pair_support = sum(int(p.get("intersections", 0)) >= 500 for p in pair_stats)
        pair_support_ratio = safe_ratio(pair_support, len(pair_stats))
        inliers = int(geom.get("inliers", 0))
        correspondences = int(geom.get("correspondences", 0))
        reproj = geom.get("median_reprojection_px")
        # All terms are dimensionless risks.  Larger means less reliable.
        candidate_risk = max(
            float(cons_t) / 1.5, float(cons_r) / 15.0,
            float(top10_t) / 3.0, float(top10_r) / 30.0)
        temporal_risk = max(
            float(ev.get("predicted_jump_m", 0.0)) / 2.0,
            float(ev.get("predicted_jump_deg", 0.0)) / 30.0,
            float(ev.get("appearance_change", 0.0)) / 0.20,
            float(ev.get("frame_change", 0.0)) / 0.20)
        has_geometry = bool(gp is not None and inliers > 0)
        if has_geometry:
            # Keep state-independent geometry quality separate from evidence
            # that depends on the previous accepted state.  The old code only
            # exported the max-combined geometry_risk, which made it
            # impossible for dynamic re-trust to decay the previous-state
            # term without also retaining the same term inside the max.
            geometry_intrinsic_risk = max(
                max(0.0, 1000.0 - inliers) / 1000.0,
                float(reproj or 0.0) / 3.0,
                1.0 - pair_support_ratio)
            geometry_cross_source_risk = max(
                geom_t_cons / 2.0, geom_r_cons / 30.0,
                geom_t_mem / 2.0, geom_r_mem / 30.0)
            geometry_previous_state_risk = max(
                geom_t_prev / 2.0, geom_r_prev / 30.0)
            geometry_risk = max(geometry_intrinsic_risk,
                                geometry_cross_source_risk,
                                geometry_previous_state_risk)
        else:
            geometry_intrinsic_risk = geometry_cross_source_risk = geometry_previous_state_risk = 0.0
            geometry_risk = 0.0
        reliability_risk = float(max(candidate_risk, temporal_risk, geometry_risk))
        rows.append({
            "sequence": sequence, "query_index": qi,
            "action": tr.get("action"), "pose_source": tr.get("pose_source"),
            "route_pose": tr.get("pose"), "route_accepted": bool(tr.get("accepted", False)),
            "geometry_available": has_geometry,
            "geometry_requested": geometry_requested,
            "geometry_reference_indices": tr.get("geometry_reference_indices", []),
            "candidate_risk": float(candidate_risk), "temporal_risk": float(temporal_risk),
            "geometry_risk": float(geometry_risk),
            "geometry_intrinsic_risk": float(geometry_intrinsic_risk),
            "geometry_cross_source_risk": float(geometry_cross_source_risk),
            "geometry_previous_state_risk": float(geometry_previous_state_risk),
            "reliability_risk": reliability_risk,
            "top10_candidate_dispersion_m": float(top10_t),
            "top10_candidate_dispersion_deg": float(top10_r),
            "consensus_dispersion_m": float(cons_t), "consensus_dispersion_deg": float(cons_r),
            "memory_dispersion_m": float(mem_t), "memory_dispersion_deg": float(mem_r),
            "appearance_change": float(ev.get("appearance_change", 0.0)),
            "frame_change": float(ev.get("frame_change", 0.0)),
            "predicted_jump_m": float(ev.get("predicted_jump_m", 0.0)),
            "predicted_jump_deg": float(ev.get("predicted_jump_deg", 0.0)),
            "memory_reliable_before": bool(ev.get("memory_reliable_before", False)),
            "memory_reliable_after": bool(ev.get("memory_reliable_after", False)),
            "candidate_conflict": bool(ev.get("candidate_conflict", False)),
            "consensus_reliable": bool(ev.get("consensus_reliable", False)),
            "roma_inliers": inliers, "roma_correspondences": correspondences,
            "roma_inlier_ratio": safe_ratio(inliers, correspondences),
            "roma_median_reprojection_px": None if reproj is None else float(reproj),
            "roma_pair_support": int(pair_support), "roma_pair_count": len(pair_stats),
            "roma_pair_support_ratio": float(pair_support_ratio),
            "geometry_vs_consensus_m": geom_t_cons, "geometry_vs_consensus_deg": geom_r_cons,
            "geometry_vs_memory_m": geom_t_mem, "geometry_vs_memory_deg": geom_r_mem,
            "geometry_vs_previous_m": geom_t_prev, "geometry_vs_previous_deg": geom_r_prev,
            # Online audit fields are copied from the original causal trace.
            "query_gt_read": bool(tr.get("query_gt_read", False)),
            "uses_future": bool(tr.get("uses_future", False)),
            "selection_uses_gt": bool(tr.get("selection_uses_gt", False)),
        })
        # The replay state is historical route output only.  No query GT is used.
        if tr.get("accepted") and tr.get("pose") is not None:
            previous_memory = np.asarray(tr["pose"], dtype=np.float64)
            previous_prediction = previous_memory
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    p.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    p.add_argument("--relative-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    p.add_argument("--trace-root", type=Path, default=Path("outputs/lilocbench/AVL-011/final_v1"))
    p.add_argument("--roma-root", type=Path, default=Path("outputs/lilocbench/AVL-004/always_geometry/roma"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/lilocbench/AVL-012"))
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sequence in SEQUENCES:
        rows = build_sequence(args, sequence)
        out = args.output_root / sequence
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evidence.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        all_rows.extend(rows)
    manifest = {
        "experiment": "AVL-012", "score_version": "v1",
        "thresholds": list(THRESHOLDS),
        "formula": "max(candidate dispersion risk, temporal/appearance risk, geometry intrinsic risk, cross-source risk, previous-state risk)",
        "geometry_risk_components": {
            "intrinsic": "max(inlier deficit, reprojection, 1-pair-support-ratio)",
            "cross_source": "max(geometry-vs-consensus, geometry-vs-memory)",
            "previous_state": "max(geometry-vs-previous-state)",
            "dynamic_retrust": "only previous_state is state-decayed; intrinsic and cross_source remain active",
        },
        "query_gt_read": False, "uses_future": False, "selection_uses_gt": False,
        "num_frames": len(all_rows), "sequences": list(SEQUENCES),
    }
    (args.output_root / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frames": len(all_rows), "risk_min": min(x["reliability_risk"] for x in all_rows),
                      "risk_max": max(x["reliability_risk"] for x in all_rows),
                      "risk_median": float(np.median([x["reliability_risk"] for x in all_rows]))}, indent=2))


if __name__ == "__main__":
    main()
