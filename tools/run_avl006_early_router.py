#!/usr/bin/env python3
"""Generate GT-free pre-routing traces for AVL-006-B.

The router deliberately reads only mapping-only retrieval scores, current image
quality/change signals, and its own previous action.  It never opens pose or
ground-truth files.  The output is a policy trace; scoring is performed by the
separate evaluator so that the routing decision and the GT audit remain
separate artifacts.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
POLICIES = {
    # Thresholds were frozen before reading any evaluator output.  A vote is a
    # cheap risk cue; no pose/dispersion/GT value enters this rule.
    "conservative": {"top1": 0.20, "margin": 0.003, "change": 0.30, "lap": 80.0, "min_votes": 2},
    "balanced": {"top1": 0.30, "margin": 0.010, "change": 0.20, "lap": 120.0, "min_votes": 2},
    "context": {"top1": 0.30, "margin": 0.008, "change": 0.18, "lap": 100.0, "min_votes": 2},
    "sensitive": {"top1": 0.45, "margin": 0.020, "change": 0.18, "lap": 160.0, "min_votes": 1},
    "retrieval_sensitive": {"top1": 0.35, "margin": 0.015, "change": 0.25, "lap": 100.0, "min_votes": 1},
}


def feature_rows(data_root: Path, retrieval_root: Path, seq: str):
    manifest = json.loads((data_root / seq / "manifest.json").read_text())
    retrieval = np.load(retrieval_root / f"{seq}_top100.npz")
    scores = retrieval["mapping_scores"]
    query_indices = retrieval["query_indices"].astype(int)
    if len(manifest["frames"]) != len(query_indices):
        raise RuntimeError(f"frame/retrieval count mismatch for {seq}")
    rows = []
    prev_small = None
    for i, frame in enumerate(manifest["frames"]):
        image_path = data_root / seq / frame["path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        laplacian_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
        small = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        frame_change = 0.0 if prev_small is None else float(np.mean(cv2.absdiff(small, prev_small)) / 255.0)
        prev_small = small
        s = scores[i]
        rows.append({
            "query_index": int(query_indices[i]),
            "timestamp": frame["timestamp"],
            "top1_score": float(s[0]),
            "top1_margin": float(s[0] - s[1]),
            "top5_mean_score": float(np.mean(s[:5])),
            "laplacian_var": laplacian_var,
            "frame_change": frame_change,
            "feature_source": "mapping_only_retrieval+current_rgb+previous_rgb",
        })
    return rows


def route(rows, policy):
    prev_action = "start"
    frames_since_roma = 10**9
    out = []
    for row in rows:
        cues = {
            "low_top1": row["top1_score"] < policy["top1"],
            "low_margin": row["top1_margin"] < policy["margin"],
            "high_change": row["frame_change"] > policy["change"],
            "low_sharpness": row["laplacian_var"] < policy["lap"],
        }
        votes = int(sum(cues.values()))
        action = "roma" if votes >= policy["min_votes"] else "reloc3r_memory"
        frames_since_roma = 0 if action == "roma" else frames_since_roma + 1
        rec = dict(row)
        rec.update({
            "action": action,
            "previous_action": prev_action,
            "frames_since_previous_roma": int(frames_since_roma if prev_action != "roma" else 1),
            "risk_votes": votes,
            "risk_cues": cues,
            "query_gt_read": False,
            "selection_uses_gt": False,
        })
        out.append(rec)
        prev_action = action
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/lilocbench/prepared_1hz", type=Path)
    ap.add_argument("--retrieval-root", default="outputs/lilocbench/AVL-004/always_geometry/retrieval", type=Path)
    ap.add_argument("--output", default="outputs/lilocbench/AVL-006/early_router", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_timing = {}
    all_features = {}
    for seq in SEQUENCES:
        t0 = time.perf_counter()
        all_features[seq] = feature_rows(args.data_root, args.retrieval_root, seq)
        feature_timing[seq] = {"context_feature_elapsed_s": time.perf_counter() - t0, "frames": len(all_features[seq])}
    for policy_name, policy in POLICIES.items():
        policy_dir = args.output / policy_name
        policy_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for seq in SEQUENCES:
            rows = route(all_features[seq], policy)
            path = policy_dir / f"{seq}.jsonl"
            with path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            counts[seq] = {
                "frames": len(rows),
                "roma_calls": sum(r["action"] == "roma" for r in rows),
                "reloc3r_memory_calls": sum(r["action"] == "reloc3r_memory" for r in rows),
            }
        (policy_dir / "protocol.json").write_text(json.dumps({
            "policy_name": policy_name,
            "thresholds": policy,
            "inputs": ["top1_score", "top1_margin", "top5_mean_score", "laplacian_var", "frame_change", "previous_action"],
            "forbidden_inputs": ["Reloc3r_pose", "candidate_dispersion", "RoMa/HLoc_output", "query_GT", "future_frames"],
            "query_gt_read": False,
            "selection_uses_gt": False,
            "counts": counts,
        }, indent=2) + "\n")
    (args.output / "router_manifest.json").write_text(json.dumps({
        "sequences": SEQUENCES,
        "policies": POLICIES,
        "action_semantics": {
            "reloc3r_memory": "posthoc score from AVL-003 memory3; router does not inspect it",
            "roma": "posthoc score from AVL-005 seed0 Always-RoMa; router does not inspect it",
        },
        "query_gt_read": False,
        "selection_uses_gt": False,
    }, indent=2) + "\n")
    (args.output / "context_feature_timing.json").write_text(json.dumps({
        "timing_boundary": "image read + grayscale Laplacian + resized adjacent-frame difference; excludes retrieval",
        "sequences": feature_timing,
        "total_s": sum(x["context_feature_elapsed_s"] for x in feature_timing.values()),
        "query_gt_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
