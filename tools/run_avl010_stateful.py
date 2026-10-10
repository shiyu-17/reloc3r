#!/usr/bin/env python3
"""Causal AVL-010 stateful comparison using online DPVO trajectories.

The script is deliberately GT-free.  Query GT is read only by the independent
evaluator added separately.  Mapping poses are the existing map asset used by
AVL-009 to construct the frozen Reloc3r consensus candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, read_manifest, tum_camera_poses  # noqa: E402
from tools.run_avl003_ablations import load_relative  # noqa: E402
from tools.run_avl009_sequence_baseline import (  # noqa: E402
    GLOBAL_R_DISP,
    GLOBAL_T_DISP,
    JUMP_R,
    JUMP_T,
    ROMA_MAX_REPROJ,
    ROMA_MIN_INLIERS,
    average_from_relative,
    candidate_dispersion,
    global_info,
    load_roma_cache,
    pose_distance,
    pose_finite,
    select_consensus,
)

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
MEMORY_MAX_HOLD = 3


def serial_pose(pose):
    return np.asarray(pose, dtype=np.float64).tolist() if pose_finite(pose) else None


def load_dpvo(path: Path):
    data = np.load(path)
    poses = data["poses"].astype(np.float64)
    valid = data["valid"].astype(bool)
    return [poses[i] if valid[i] and pose_finite(poses[i]) else None for i in range(len(poses))], data


def load_avl009_memory(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[int(row["query_index"])] = row["methods"]["memory3"]
    return rows


def load_state():
    return {"last_pose": None, "failure_streak": 0, "memory_hold": 0, "last_frame": None}


def update_state(state, idx, action, pose):
    state["last_frame"] = int(idx)
    if action == "use_memory":
        state["failure_streak"] += 1
        state["memory_hold"] += 1
    elif action == "reject":
        state["failure_streak"] += 1
        state["memory_hold"] = 0
    else:
        state["last_pose"] = pose if pose_finite(pose) else state["last_pose"]
        state["failure_streak"] = 0
        state["memory_hold"] = 0


class CausalSim3:
    """Two-anchor causal metric alignment for a monocular trajectory."""

    def __init__(self):
        self.anchors = []
        self.scale_ready = False
        self.scale = None
        self.matrix = None
        self.last_residual = None

    def observe(self, idx, local, global_pose):
        if not (pose_finite(local) and pose_finite(global_pose)):
            return {"updated": False, "reason": "invalid_pose"}
        if self.anchors and self.anchors[-1][0] == int(idx):
            return {"updated": False, "reason": "duplicate_frame"}
        current = (int(idx), np.asarray(local, dtype=np.float64).copy(), np.asarray(global_pose, dtype=np.float64).copy())
        if not self.anchors:
            self.anchors.append(current)
            # One point fixes only a rigid placement; this is not yet a metric
            # Sim(3), so online evaluation must not score propagated poses.
            self.matrix = np.eye(4, dtype=np.float64)
            self.matrix[:3, :3] = global_pose[:3, :3] @ local[:3, :3].T
            self.matrix[:3, 3] = global_pose[:3, 3] - self.matrix[:3, :3] @ local[:3, 3]
            return {"updated": True, "ready": False, "reason": "first_anchor"}

        _, prev_local, prev_global = self.anchors[-1]
        local_delta = float(np.linalg.norm(local[:3, 3] - prev_local[:3, 3]))
        global_delta = float(np.linalg.norm(global_pose[:3, 3] - prev_global[:3, 3]))
        if local_delta < 0.05:
            self.anchors.append(current)
            return {"updated": False, "ready": self.scale_ready, "reason": "local_baseline_too_small",
                    "local_baseline_m": local_delta, "global_baseline_m": global_delta}
        scale = global_delta / local_delta
        if not (0.1 <= scale <= 10.0) or not np.isfinite(scale):
            self.anchors.append(current)
            return {"updated": False, "ready": self.scale_ready, "reason": "scale_out_of_range",
                    "local_baseline_m": local_delta, "global_baseline_m": global_delta, "scale": scale}

        # Use the current accepted anchor to define the left Sim(3) rotation
        # and translation.  This keeps the corrected trajectory attached to
        # the just-verified metric pose; the previous-anchor residual is still
        # exposed through the diagnostic below rather than hidden by a fit.
        R = global_pose[:3, :3] @ local[:3, :3].T
        S = np.eye(4, dtype=np.float64)
        S[:3, :3] = scale * R
        S[:3, 3] = global_pose[:3, 3] - scale * R @ local[:3, 3]
        mapped = np.eye(4, dtype=np.float64)
        mapped[:3, :3] = R @ prev_local[:3, :3]
        mapped[:3, 3] = scale * R @ prev_local[:3, 3] + S[:3, 3]
        self.matrix = S
        self.scale = float(scale)
        self.scale_ready = True
        self.last_residual = pose_distance(mapped, prev_global)
        self.anchors.append(current)
        return {"updated": True, "ready": True, "reason": "two_anchor_sim3", "scale": float(scale),
                "local_baseline_m": local_delta, "global_baseline_m": global_delta,
                "anchor_residual_m": self.last_residual[0], "anchor_residual_deg": self.last_residual[1]}

    def transform(self, local):
        if not self.scale_ready or self.matrix is None or not pose_finite(local):
            return None
        # ``self.matrix`` stores a Sim(3): its 3x3 block contains scale * R.
        # Left-multiplying an SE(3) pose directly would therefore scale the
        # output rotation and produce a matrix that is not a valid pose.  Apply
        # scale only to translation and keep the rotational part in SO(3).
        scale = float(self.scale)
        align_r = self.matrix[:3, :3] / scale
        align_t = self.matrix[:3, 3]
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = align_r @ local[:3, :3]
        output[:3, 3] = scale * align_r @ local[:3, 3] + align_t
        return output


def row(method, idx, action, pose, source, g, local, anchor_event, state, evidence=None):
    update_state(state, idx, action, pose)
    return {
        "method": method, "query_index": int(idx), "selected_action": action,
        "prediction_source": source, "pose": serial_pose(pose), "accepted": bool(pose_finite(pose)),
        "global_available": bool(g.get("available", False)),
        "global_reliable": bool(g.get("reliable", False)),
        "dpvo_available": bool(pose_finite(local)), "memory_available": bool(state["last_pose"] is not None),
        "failure_streak_after": int(state["failure_streak"]),
        "confidence": 1.0 if action in {"accept_global", "global_init", "call_roma"} else (0.8 if pose_finite(pose) else 0.0),
        "evidence": {**(evidence or {}), "anchor_event": anchor_event,
                     "global_dispersion_m": float(g.get("disp_t", math.inf)),
                     "global_dispersion_deg": float(g.get("disp_r", math.inf))},
        "uses_future": False, "query_gt_read": False, "selection_uses_gt": False,
    }


def compatible(state, candidate):
    if not pose_finite(candidate):
        return False, (math.inf, math.inf)
    if state["last_pose"] is None:
        return True, (math.inf, math.inf)
    d = pose_distance(state["last_pose"], candidate)
    return d[0] <= JUMP_T and d[1] <= JUMP_R, d


def run_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, sequence)
    db_pose, _ = tum_camera_poses(args.mapping_gt, db_times, base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relatives = load_relative(args.global_root / sequence, len(query_frames))
    local_poses, local_npz = load_dpvo(args.dpvo_root / sequence / args.trajectory_file_name)
    roma = load_roma_cache(args.roma_root / f"{sequence}_top100_merged.jsonl")
    old_memory = load_avl009_memory(args.memory_root / "sequence_memory" / sequence / "trace.jsonl")
    states = {m: load_state() for m in ("dpvo_temporal_only", "global_dpvo", "global_dpvo_memory", "global_dpvo_memory_roma")}
    anchors = {m: CausalSim3() for m in states}
    rows = []
    for idx, item in enumerate(relatives):
        g = global_info(db_pose, item)
        local = local_poses[idx] if idx < len(local_poses) else None
        out = {"query_index": int(idx), "timestamp": float(query_times[idx]), "methods": {}}

        # AVL-009 Memory-3 is copied unchanged as the causal historical baseline.
        out["methods"]["memory3"] = old_memory[idx]

        for method in ("dpvo_temporal_only", "global_dpvo", "global_dpvo_memory", "global_dpvo_memory_roma"):
            st, align = states[method], anchors[method]
            global_ok, global_jump = compatible(st, g.get("pose")) if g.get("reliable", False) else (False, (math.inf, math.inf))
            anchor_event = {"updated": False, "reason": "not_observed"}
            anchored = align.transform(local)

            # Global candidate is an accepted anchor only after the same jump
            # guard as AVL-009.  RoMa can create an anchor in the recovery arm.
            if global_ok:
                anchor_event = align.observe(idx, local, g["pose"])
            anchored = align.transform(local)
            temporal_ok, temporal_jump = compatible(st, anchored)
            action, pose, source = "reject", None, "none"
            evidence = {"global_jump": global_jump, "temporal_jump": temporal_jump,
                        "sim3_ready": bool(align.scale_ready), "sim3_scale": align.scale,
                        "anchor_count": len(align.anchors)}

            if method == "dpvo_temporal_only":
                if idx == 0 and g.get("reliable", False):
                    action, pose, source = "global_init", g["pose"], "global"
                elif temporal_ok:
                    action, pose, source = "accept_temporal", anchored, "dpvo"
                # DPVO temporal-only uses reliable global observations only to
                # establish/refresh causal Sim(3), never as a later pose output.
            else:
                if global_ok:
                    action, pose, source = "accept_global", g["pose"], "global"
                elif temporal_ok:
                    action, pose, source = "accept_temporal", anchored, "dpvo"
                elif method in {"global_dpvo_memory", "global_dpvo_memory_roma"} and st["last_pose"] is not None and st["memory_hold"] < MEMORY_MAX_HOLD:
                    action, pose, source = "use_memory", st["last_pose"], "memory"

            if method == "global_dpvo_memory_roma" and action in {"reject", "use_memory"}:
                rr = roma.get(idx)
                if rr is not None and rr.get("verified", False):
                    pose = rr["pose"]
                    action, source = "call_roma", "roma"
                    evidence.update({"roma_inliers": rr["inliers"], "roma_median_reprojection_px": rr["median_reprojection_px"],
                                     "roma_reference_indices": rr["reference_indices"]})
                    if pose_finite(local):
                        anchor_event = align.observe(idx, local, pose)
                        anchored = align.transform(local)
            out["methods"][method] = row(method, idx, action, pose, source, g, local, anchor_event, st, evidence)

        out.update({"query_gt_read": False, "uses_future": False, "selection_uses_gt": False})
        rows.append(out)

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    trace_path = out_root / f"{sequence}.jsonl"
    trace_path.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
    return {"sequence": sequence, "frames": len(rows), "trace": str(trace_path),
            "dpvo_manifest": str(args.dpvo_root / sequence / "manifest.json"),
            "query_gt_read": False, "future_frame_read": False,
            "local_valid": int(sum(pose_finite(x) for x in local_poses)),
            "methods": {m: {"anchor_count": len(anchors[m].anchors), "sim3_ready": anchors[m].scale_ready,
                             "scale": anchors[m].scale} for m in anchors}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--global-root", type=Path, required=True)
    p.add_argument("--dpvo-root", type=Path, required=True)
    p.add_argument("--memory-root", type=Path, required=True)
    p.add_argument("--roma-root", type=Path, required=True)
    p.add_argument("--mapping-gt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trajectory-file-name", default="trajectory_online.npz")
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_sequence(args, seq) for seq in args.sequences]
    (args.output_dir / "manifest.json").write_text(json.dumps({"experiment": "AVL-010", "sequences": summaries,
                                                                  "query_gt_read": False, "future_frame_read": False,
                                                                  "selection_uses_gt": False,
                                                                  "protocol": {"sim3_min_local_baseline_m": 0.05,
                                                                               "sim3_scale_range": [0.1, 10.0],
                                                                               "jump_guard_m": JUMP_T,
                                                                               "jump_guard_deg": JUMP_R}}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
