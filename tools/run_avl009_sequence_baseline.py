#!/usr/bin/env python3
"""Run the GT-free causal sequence baselines for AVL-009.

The script consumes mapping-only Reloc3r candidates from AVL-002 and the
causal adjacent-frame edges from ``run_avl009_temporal_edges.py``.  It never
loads query GT.  Each method owns an independent state dictionary so that
memory or a rejected frame in one method cannot leak into another method.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    average_from_relative,
    base_to_front_rgb_optical,
    candidate_dispersion,
    read_manifest,
    tum_camera_poses,
)
from tools.run_avl003_ablations import load_relative, select_consensus  # noqa: E402


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
GLOBAL_T_DISP = 1.5
GLOBAL_R_DISP = 15.0
AGREEMENT_T = 2.0
AGREEMENT_R = 30.0
JUMP_T = 3.0
JUMP_R = 45.0
MEMORY_MAX_HOLD = 3
ROMA_MIN_INLIERS = 100
ROMA_MAX_REPROJ = 3.0


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def pose_finite(pose) -> bool:
    if pose is None:
        return False
    pose = np.asarray(pose, dtype=np.float64)
    return pose.shape == (4, 4) and np.isfinite(pose).all() and np.linalg.norm(
        pose[3] - np.array([0.0, 0.0, 0.0, 1.0])
    ) < 1e-3


def pose_distance(first, second):
    if not (pose_finite(first) and pose_finite(second)):
        return float("inf"), float("inf")
    first, second = np.asarray(first), np.asarray(second)
    dt = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    dr = float(np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()))
    return dt, dr


def serialize_pose(pose):
    return np.asarray(pose, dtype=np.float64).tolist() if pose_finite(pose) else None


def state_template():
    return {
        "last_frame_idx": None,
        "last_accepted_pose": None,
        "last_accepted_frame": None,
        "previous_pose": None,
        "previous_confidence": 0.0,
        "failure_streak": 0,
        "recent_reliable_history": [],
        "last_global_anchor": None,
        "reference_reliability": {},
        "memory_hold_streak": 0,
    }


def state_snapshot(state):
    output = dict(state)
    output["last_accepted_pose"] = serialize_pose(state["last_accepted_pose"])
    output["previous_pose"] = serialize_pose(state["previous_pose"])
    output["last_global_anchor"] = serialize_pose(state["last_global_anchor"])
    output["reference_reliability"] = dict(state["reference_reliability"])
    output["recent_reliable_history"] = list(state["recent_reliable_history"])
    return output


def update_references(state, mapping_indices, reliable):
    delta = 0.05 if reliable else -0.05
    for idx in mapping_indices[:3]:
        key = str(int(idx))
        state["reference_reliability"][key] = float(
            np.clip(state["reference_reliability"].get(key, 0.5) + delta, 0.0, 1.0)
        )


def accept_state(state, frame_idx, pose, action, confidence, mapping_indices, reliable_global=False):
    state["last_frame_idx"] = int(frame_idx)
    state["previous_pose"] = pose if pose_finite(pose) else state["previous_pose"]
    if action == "use_memory":
        state["memory_hold_streak"] += 1
        state["failure_streak"] += 1
    elif action == "reject":
        state["failure_streak"] += 1
        state["memory_hold_streak"] = 0
    else:
        state["last_accepted_pose"] = pose if pose_finite(pose) else state["last_accepted_pose"]
        state["last_accepted_frame"] = int(frame_idx) if pose_finite(pose) else state["last_accepted_frame"]
        state["memory_hold_streak"] = 0
        state["failure_streak"] = 0
        if action in {"accept_global", "accept_temporal", "call_roma", "global_init"}:
            state["recent_reliable_history"] = (state["recent_reliable_history"] + [int(frame_idx)])[-3:]
        if action in {"accept_global", "global_init", "call_roma"} and pose_finite(pose):
            state["last_global_anchor"] = pose
    state["previous_confidence"] = float(confidence)
    if mapping_indices is not None:
        update_references(state, mapping_indices, reliable_global)


def global_info(db_pose, item):
    mapping_indices = item["indices"]
    relative = item["relative"]
    candidates = [db_pose[int(idx)] @ rel for idx, rel in zip(mapping_indices, relative)]
    if not candidates or not all(pose_finite(p) for p in candidates):
        return {"available": False, "pose": None, "top10": None, "candidates": candidates,
                "selected": [], "disp_t": float("inf"), "disp_r": float("inf"),
                "mapping_indices": mapping_indices}
    selected = select_consensus(candidates, min(3, len(candidates)))
    selected = list(selected)
    pose = average_from_relative(
        [db_pose[int(mapping_indices[i])] for i in selected],
        [relative[i] for i in selected],
    )[0]
    top10 = average_from_relative(
        [db_pose[int(idx)] for idx in mapping_indices], list(relative)
    )[0]
    disp_t, disp_r = candidate_dispersion([candidates[i] for i in selected])
    return {
        "available": pose_finite(pose),
        "pose": pose,
        "top10": top10,
        "candidates": candidates,
        "selected": selected,
        "disp_t": float(disp_t),
        "disp_r": float(disp_r),
        "mapping_indices": mapping_indices,
        "reliable": bool(pose_finite(pose) and disp_t <= GLOBAL_T_DISP and disp_r <= GLOBAL_R_DISP),
    }


def load_temporal_edges(edge_dir: Path):
    edges = {}
    for path in sorted(edge_dir.glob("edges_shard_*.npz")):
        data = np.load(path)
        for idx, pose, valid in zip(data["query_indices"], data["relative_poses"], data["valid"]):
            edges[int(idx)] = pose if bool(valid) and pose_finite(pose) else None
    return edges


def load_roma_cache(path: Path):
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pose = np.asarray(row.get("pose"), dtype=np.float64) if row.get("pose") is not None else None
        inliers = int(row.get("inliers", 0))
        reproj = float(row.get("median_reprojection_px", math.inf))
        verified = bool(pose_finite(pose) and inliers >= ROMA_MIN_INLIERS and reproj <= ROMA_MAX_REPROJ)
        cache[int(row["query_index"])] = {
            "pose": pose if pose_finite(pose) else None,
            "verified": verified,
            "inliers": inliers,
            "median_reprojection_px": reproj,
            "elapsed_s": float(row.get("elapsed_s", 0.0)),
            "reference_indices": row.get("reference_indices", []),
        }
    return cache


def decide_fusion(state, global_data, temporal_pose, use_memory):
    last = state["last_accepted_pose"]
    global_available = bool(global_data["available"])
    global_reliable = bool(global_data.get("reliable", False))
    temporal_available = pose_finite(temporal_pose)
    global_jump = pose_distance(last, global_data["pose"]) if last is not None and global_available else (math.inf, math.inf)
    temporal_jump = pose_distance(last, temporal_pose) if last is not None and temporal_available else (math.inf, math.inf)
    agreement = pose_distance(global_data["pose"], temporal_pose) if global_available and temporal_available else (math.inf, math.inf)
    temp_ok = temporal_available and (last is None or (temporal_jump[0] <= JUMP_T and temporal_jump[1] <= JUMP_R))
    glob_ok = global_available and (last is None or (global_jump[0] <= JUMP_T and global_jump[1] <= JUMP_R))

    if last is None:
        if global_reliable:
            return "accept_global", global_data["pose"], 1.0, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
                "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}
        return "reject", None, 0.0, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
            "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}

    # A reliable global candidate is authoritative when it is compatible with
    # the last accepted state.  Temporal propagation is preferred only when
    # the global candidate is itself unreliable or violates the jump guard;
    # this prevents a drifting temporal edge from overriding a stable global
    # anchor merely because the edge is numerically finite.
    if global_reliable and glob_ok:
        return "accept_global", global_data["pose"], 1.0, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
            "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}
    if temp_ok:
        return "accept_temporal", temporal_pose, 0.8, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
            "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}
    if use_memory and state["memory_hold_streak"] < MEMORY_MAX_HOLD:
        return "use_memory", last, 0.4, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
            "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}
    return "reject", None, 0.0, {"global_reliable": global_reliable, "temporal_ok": temp_ok,
        "global_jump": global_jump, "temporal_jump": temporal_jump, "agreement": agreement}


def method_row(state, frame_idx, method, action, pose, confidence, global_data, temporal_available,
               memory_available, evidence, mapping_indices):
    before = int(state["failure_streak"])
    reliable_global = bool(global_data.get("reliable", False))
    accept_state(state, frame_idx, pose, action, confidence, mapping_indices, reliable_global)
    after = int(state["failure_streak"])
    return {
        "method": method,
        "query_index": int(frame_idx),
        "selected_action": action,
        "prediction_source": {"accept_global": "global", "accept_temporal": "temporal",
                               "use_memory": "memory", "call_roma": "roma", "global_init": "global",
                               "reject": "none"}.get(action, action),
        "pose": serialize_pose(pose),
        "accepted": bool(pose_finite(pose)),
        "temporal_available": bool(temporal_available),
        "global_available": bool(global_data.get("available", False)),
        "global_reliable": reliable_global,
        "memory_available": bool(memory_available),
        "failure_streak_before": before,
        "failure_streak_after": after,
        "confidence": float(confidence),
        "evidence": evidence,
        "state_after": state_snapshot(state),
        "uses_future": False,
        "query_gt_read": False,
        "selection_uses_gt": False,
    }


def run_sequence(args, sequence):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, sequence)
    db_pose, _ = tum_camera_poses(args.mapping_gt, db_times, base_to_front_rgb_optical(db_root), args.max_gt_delta)
    relative = load_relative(args.global_root / sequence, len(query_frames))
    temporal = load_temporal_edges(args.temporal_root / sequence)
    roma = load_roma_cache(args.roma_root / f"{sequence}_top100_merged.jsonl")
    states = {name: state_template() for name in (
        "temporal_only", "global_temporal", "memory3", "global_temporal_memory", "recovery")}
    memory3_prev = None
    rows = []
    timings = {name: {"frames": 0, "roma_s": 0.0} for name in states}
    for frame_idx, item in enumerate(relative):
        g = global_info(db_pose, item)
        temporal_rel = temporal.get(frame_idx)
        out = {"query_index": frame_idx, "timestamp": float(query_times[frame_idx]), "methods": {}}

        # Independent-frame baselines.  They do not update a state.
        for name, pose in (("top10", g["top10"]), ("consensus3", g["pose"])):
            out["methods"][name] = {
                "method": name, "query_index": frame_idx, "selected_action": "accept_global",
                "prediction_source": "global", "pose": serialize_pose(pose), "accepted": pose_finite(pose),
                "temporal_available": bool(pose_finite(temporal_rel)), "global_available": bool(g["available"]),
                "global_reliable": bool(g.get("reliable", False)), "memory_available": False,
                "failure_streak_before": 0, "failure_streak_after": 0, "confidence": 1.0,
                "evidence": {"dispersion_m": g["disp_t"], "dispersion_deg": g["disp_r"]},
                "state_after": None, "uses_future": False, "query_gt_read": False,
                "selection_uses_gt": False,
            }

        # Exact causal memory-3 selector from AVL-003, retaining only internally reliable memory.
        candidates = g["candidates"]
        if candidates:
            if memory3_prev is None:
                selected = select_consensus(candidates, min(3, len(candidates)))
            else:
                distances = []
                for idx, candidate in enumerate(candidates):
                    dt, dr = pose_distance(candidate, memory3_prev)
                    distances.append((dt + dr / 10.0, dt, dr, idx))
                local = [x for x in sorted(distances) if x[1] <= 1.5 and x[2] <= 30.0]
                selected = [x[3] for x in local[:3]] if local else select_consensus(candidates, min(3, len(candidates)))
            memory_pose = average_from_relative(
                [db_pose[int(item["indices"][i])] for i in selected],
                [item["relative"][i] for i in selected],
            )[0]
            md_t, md_r = candidate_dispersion([candidates[i] for i in selected])
            memory_reliable = len(selected) >= 2 and md_t <= 1.5 and md_r <= 15.0
            if memory_reliable:
                memory3_prev = memory_pose
        else:
            selected, memory_pose, md_t, md_r, memory_reliable = [], None, math.inf, math.inf, False
        out["methods"]["memory3"] = {
            "method": "memory3", "query_index": frame_idx, "selected_action": "accept_global",
            "prediction_source": "memory3", "pose": serialize_pose(memory_pose), "accepted": pose_finite(memory_pose),
            "temporal_available": bool(pose_finite(temporal_rel)), "global_available": bool(g["available"]),
            "global_reliable": bool(memory_reliable), "memory_available": memory3_prev is not None,
            "failure_streak_before": 0, "failure_streak_after": 0, "confidence": 0.8,
            "evidence": {"dispersion_m": float(md_t), "dispersion_deg": float(md_r), "internal_reliable": memory_reliable},
            "state_after": None, "uses_future": False, "query_gt_read": False,
            "selection_uses_gt": False,
        }

        # Pure temporal propagation.  The first frame is a global initialization; every later frame uses only the previous output.
        ts = states["temporal_only"]
        if frame_idx == 0 and g["available"]:
            tp, ta, tc = g["pose"], "global_init", 1.0
        elif pose_finite(temporal_rel) and pose_finite(ts["last_accepted_pose"]):
            tp, ta, tc = ts["last_accepted_pose"] @ temporal_rel, "accept_temporal", 0.8
        else:
            tp, ta, tc = None, "reject", 0.0
        out["methods"]["temporal_only"] = method_row(
            ts, frame_idx, "temporal_only", ta, tp, tc, g, pose_finite(temporal_rel),
            ts["last_accepted_pose"] is not None, {"edge_available": pose_finite(temporal_rel)}, item["indices"])

        # Stateful deterministic fusion without and with memory hold.
        for name, use_memory in (("global_temporal", False), ("global_temporal_memory", True)):
            st = states[name]
            temporal_pose = st["last_accepted_pose"] @ temporal_rel if pose_finite(temporal_rel) and pose_finite(st["last_accepted_pose"]) else None
            action, pose, confidence, evidence = decide_fusion(st, g, temporal_pose, use_memory)
            out["methods"][name] = method_row(
                st, frame_idx, name, action, pose, confidence, g, pose_finite(temporal_pose),
                st["last_accepted_pose"] is not None and st["memory_hold_streak"] < MEMORY_MAX_HOLD,
                evidence, item["indices"])

        # Recovery ablation: replay an already computed, GT-free RoMa candidate only after the fusion rejects it.
        st = states["recovery"]
        temporal_pose = st["last_accepted_pose"] @ temporal_rel if pose_finite(temporal_rel) and pose_finite(st["last_accepted_pose"]) else None
        action, pose, confidence, evidence = decide_fusion(st, g, temporal_pose, True)
        roma_row = roma.get(frame_idx)
        low_confidence = not bool(evidence.get("global_reliable", False)) and not bool(evidence.get("temporal_ok", False))
        if action in {"reject", "use_memory"} and low_confidence and roma_row is not None and roma_row["verified"]:
            pose = roma_row["pose"]
            action, confidence = "call_roma", 0.9
            evidence = dict(evidence)
            evidence.update({"roma_verified": True, "roma_inliers": roma_row["inliers"],
                             "roma_median_reprojection_px": roma_row["median_reprojection_px"],
                             "roma_reference_indices": roma_row["reference_indices"],
                             "recovery_trigger": "reject_or_low_confidence"})
            timings["recovery"]["roma_s"] += roma_row["elapsed_s"]
        else:
            evidence = dict(evidence)
            evidence.update({"roma_verified": bool(roma_row and roma_row["verified"]),
                             "roma_available": roma_row is not None,
                             "recovery_trigger": "reject_or_low_confidence" if action in {"reject", "use_memory"} else "not_triggered"})
        out["methods"]["global_temporal_memory_roma_recovery"] = method_row(
            st, frame_idx, "global_temporal_memory_roma_recovery", action, pose, confidence, g,
            pose_finite(temporal_pose), st["last_accepted_pose"] is not None and st["memory_hold_streak"] < MEMORY_MAX_HOLD,
            evidence, item["indices"])
        for name in states:
            timings[name]["frames"] += 1
        out["query_gt_read"] = False
        out["uses_future"] = False
        out["selection_uses_gt"] = False
        rows.append(out)

    output = {
        "experiment": "AVL-009",
        "sequence": sequence,
        "frames": len(rows),
        "mapping_frames": len(db_frames),
        "temporal_edge_count": int(sum(1 for idx in range(1, len(rows)) if idx in temporal and temporal[idx] is not None)),
        "query_gt_read": False,
        "future_frame_read": False,
        "mapping_pose_asset": str(args.mapping_gt),
        "protocol_thresholds": {
            "global_dispersion_m": GLOBAL_T_DISP, "global_dispersion_deg": GLOBAL_R_DISP,
            "agreement_m": AGREEMENT_T, "agreement_deg": AGREEMENT_R,
            "jump_guard_m": JUMP_T, "jump_guard_deg": JUMP_R,
            "memory_max_hold": MEMORY_MAX_HOLD, "roma_min_inliers": ROMA_MIN_INLIERS,
            "roma_max_reprojection_px": ROMA_MAX_REPROJ,
        },
        "timings": timings,
        "rows": rows,
    }
    # Keep required subdirectories explicit, with one trace per relevant family.
    out_root = args.output_dir
    (out_root / "sequence_baseline" / sequence).mkdir(parents=True, exist_ok=True)
    (out_root / "sequence_memory" / sequence).mkdir(parents=True, exist_ok=True)
    (out_root / "sequence_recovery" / sequence).mkdir(parents=True, exist_ok=True)
    (out_root / "sequence_baseline" / sequence / "trace.jsonl").write_text(
        "".join(json.dumps({"query_index": r["query_index"], "timestamp": r["timestamp"],
                             "methods": {k: r["methods"][k] for k in ("top10", "consensus3", "temporal_only", "global_temporal")},
                             "query_gt_read": False, "uses_future": False, "selection_uses_gt": False}, default=json_default) + "\n" for r in rows), encoding="utf-8")
    (out_root / "sequence_memory" / sequence / "trace.jsonl").write_text(
        "".join(json.dumps({"query_index": r["query_index"], "timestamp": r["timestamp"],
                             "methods": {k: r["methods"][k] for k in ("memory3", "global_temporal_memory")},
                             "query_gt_read": False, "uses_future": False, "selection_uses_gt": False}, default=json_default) + "\n" for r in rows), encoding="utf-8")
    (out_root / "sequence_recovery" / sequence / "trace.jsonl").write_text(
        "".join(json.dumps({"query_index": r["query_index"], "timestamp": r["timestamp"],
                             "methods": {"global_temporal_memory_roma_recovery": r["methods"]["global_temporal_memory_roma_recovery"]},
                             "query_gt_read": False, "uses_future": False, "selection_uses_gt": False}, default=json_default) + "\n" for r in rows), encoding="utf-8")
    (out_root / "sequence_baseline" / sequence / "manifest.json").write_text(
        json.dumps({k: output[k] for k in ("experiment", "sequence", "frames", "mapping_frames", "temporal_edge_count",
                                            "query_gt_read", "future_frame_read", "protocol_thresholds", "timings")}, indent=2,
                   default=json_default) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True, help="AVL-002 per-sequence result root")
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--roma-root", type=Path, required=True)
    parser.add_argument("--mapping-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gt-delta", type=float, default=0.1, help="Only applies to mapping pose map asset")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation").mkdir(exist_ok=True)
    (args.output_dir / "test").mkdir(exist_ok=True)
    (args.output_dir / "failure_analysis").mkdir(exist_ok=True)
    (args.output_dir / "cost").mkdir(exist_ok=True)
    summaries = [run_sequence(args, sequence) for sequence in args.sequences]
    # The full rows are retained in separate traces; summary.json is a compact manifest.
    compact = [{k: s[k] for k in ("experiment", "sequence", "frames", "mapping_frames", "temporal_edge_count",
                                  "query_gt_read", "future_frame_read", "protocol_thresholds", "timings")} for s in summaries]
    (args.output_dir / "summary.json").write_text(json.dumps({"experiment": "AVL-009", "sequences": compact,
                                                                "query_gt_read": False, "future_frame_read": False}, indent=2,
                                                               default=json_default) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(compact, indent=2, default=json_default))


if __name__ == "__main__":
    main()
