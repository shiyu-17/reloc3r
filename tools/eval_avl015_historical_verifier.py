#!/usr/bin/env python3
"""AVL-015 Q2/Q3 source-specific verifier for historical-keyframe poses.

The candidate route is causal; query GT is loaded only in this independent
evaluator.  Threshold selection uses the validation prefix and is frozen for
the temporal test suffix.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import (base_to_front_rgb_optical,
                                           pose_error, read_manifest,
                                           tum_camera_poses)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def clamp(x):
    return float(max(0.0, min(1.0, x)))


def feature_risk(raw, source_trace):
    refs = raw.get("historical_keyframes", [])
    ev = raw.get("evidence", {})
    sims = [float(x.get("similarity", 0.0)) for x in refs]
    ages = [int(raw["query_index"]) - int(x["frame_index"]) for x in refs]
    rel = [float(source_trace[int(x["frame_index"])].get("risk_score", 1.0))
           for x in refs if int(x["frame_index"]) < len(source_trace)]
    pair_stats = ev.get("pair_stats", [])
    supported = sum(int(x.get("triangulated", 0)) >= 20 for x in pair_stats)
    pair_support = supported / max(len(pair_stats), 1)
    min_sim = min(sims) if sims else 0.0
    reproj = ev.get("median_reprojection_px")
    inliers = float(ev.get("inliers", 0))
    # All terms are evidence available before GT scoring.  The max operator is
    # deliberately conservative: a weak historical link cannot be hidden by
    # many matches from another pair.
    terms = {
        "similarity": clamp((0.995 - min_sim) / 0.12),
        "reprojection": clamp(float(reproj) / 3.0) if reproj is not None else 1.0,
        "inliers": clamp((80.0 - inliers) / 80.0),
        "pair_support": 1.0 - pair_support,
        "keyframe_reliability": max(rel) if rel else 1.0,
        "age": clamp((max(ages) if ages else 999) / 300.0),
    }
    return max(terms.values()), terms


def summarize(rows):
    accepted = [r for r in rows if r["accept"]]
    good = sum(r["success"] for r in accepted)
    return {
        "frames": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_correct": good,
        "accepted_accuracy": good / len(accepted) if accepted else 0.0,
        "false_accept_rate": (len(accepted) - good) / len(accepted) if accepted else 0.0,
        "full_frame_success": good / len(rows) if rows else 0.0,
    }


def choose_threshold(rows):
    eligible = [r for r in rows if r["pose_available"]]
    choices = []
    for t in THRESHOLDS:
        cur = [{**r, "accept": r["risk"] <= t} for r in eligible]
        s = summarize(cur)
        if s["coverage"] >= 0.20:
            choices.append((s["accepted_accuracy"], s["coverage"], -t, t, s))
    if not choices:
        return THRESHOLDS[-1]
    # Maximize validation selective accuracy, then coverage, then prefer the
    # lower threshold.  Test is never used for this choice.
    return max(choices)[3]


def curve(rows):
    return {str(t): summarize([{**r, "accept": r["pose_available"] and r["risk"] <= t}
                               for r in rows]) for t in THRESHOLDS}


def main():
    root = Path("outputs/lilocbench/AVL-015/historical_keyframe_memory")
    data_root = Path("data/lilocbench/prepared_1hz")
    gt_root = Path("data/lilocbench/gt")
    oracle = json.loads(Path("outputs/lilocbench/AVL-004/action_oracle/summary_always.json").read_text())
    report = {
        "experiment": "AVL-015",
        "component": "Q2_historical_keyframe_verifier",
        "protocol": {"threshold_grid": list(THRESHOLDS), "validation_fraction": 0.4,
                      "gt_used_only_for_evaluation": True},
        "sequences": {}, "aggregate": {}, "complementarity": {},
        "audit": {"query_gt_read": False, "uses_future": False,
                  "selection_uses_gt": False, "historical_future_access": False},
    }
    all_rows = {"all": [], "validation": [], "test": []}
    pending = {}
    validation_pool = []
    for seq in SEQUENCES:
        _, frames, times, _, seq_root = read_manifest(data_root, seq)
        gt, _ = tum_camera_poses(gt_root / f"{seq}_gt_poses.txt", times,
                                 base_to_front_rgb_optical(seq_root), 0.1)
        q3 = [json.loads(x) for x in (root / seq / "trace.jsonl").read_text().splitlines() if x.strip()]
        src = [json.loads(x) for x in (Path("outputs/lilocbench/AVL-014/AVL-013_v3") / seq / "closed_loop_trace.jsonl").read_text().splitlines() if x.strip()]
        rows = []
        for raw in q3:
            qi = int(raw["query_index"]); pose = raw.get("pose")
            t = rr = None; success = False
            if pose is not None:
                t, rr = pose_error(np.asarray(pose, dtype=np.float64), gt[qi])
                success = bool(t <= 0.25 and rr <= 5.0)
            risk, terms = feature_risk(raw, src)
            causal = raw.get("causal", {})
            max_hist = max((int(x["frame_index"]) for x in raw.get("historical_keyframes", [])), default=-1)
            row = {"sequence": seq, "query_index": qi, "pose_available": pose is not None,
                   "success": success, "risk": risk, "terms": terms, "accept": False,
                   "translation_error_m": t, "rotation_error_deg": rr,
                   "causal": causal, "max_historical_index": max_hist,
                   "other_actions": next((x["success"] for x in oracle["sequences"][seq]["rows"] if int(x["query_index"]) == qi), {})}
            rows.append(row)
            report["audit"]["query_gt_read"] |= bool(causal.get("query_gt_read", False))
            report["audit"]["uses_future"] |= bool(causal.get("uses_future", False))
            report["audit"]["selection_uses_gt"] |= bool(causal.get("selection_uses_gt", False))
            report["audit"]["historical_future_access"] |= not (max_hist < qi or max_hist < 0)
        cut = int(math.ceil(len(rows) * 0.4))
        pending[seq] = (rows, cut)
        validation_pool.extend(rows[:cut])
    # One Memory verifier is frozen globally on validation; sequence labels
    # are not available as an online action signal.
    threshold = choose_threshold(validation_pool)
    for seq, (rows, cut) in pending.items():
        for r in rows:
            r["threshold"] = threshold; r["accept"] = r["pose_available"] and r["risk"] <= threshold
        report["sequences"][seq] = {"threshold": threshold,
                                     "all": summarize(rows),
                                     "validation": summarize(rows[:cut]),
                                     "test": summarize(rows[cut:])}
        all_rows["all"].extend(rows); all_rows["validation"].extend(rows[:cut]); all_rows["test"].extend(rows[cut:])
    for split, rows in all_rows.items():
        report["aggregate"][split] = summarize(rows)
        groups = {"historical_only": 0, "map_or_roma_only": 0, "both": 0, "both_fail": 0}
        for r in rows:
            h = bool(r["accept"] and r["success"])
            other = r["other_actions"]
            m = any(bool(other.get(a, False)) for a in ("consensus", "memory", "hloc", "roma"))
            if h and not m: groups["historical_only"] += 1
            elif m and not h: groups["map_or_roma_only"] += 1
            elif h and m: groups["both"] += 1
            else: groups["both_fail"] += 1
        report["complementarity"][split] = groups
    report["curves"] = {split: curve(rows) for split, rows in all_rows.items()}
    out = root / "historical_verifier.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "sequences": report["sequences"],
                      "complementarity": report["complementarity"], "audit": report["audit"]}, indent=2))


if __name__ == "__main__":
    main()
