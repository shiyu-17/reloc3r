#!/usr/bin/env python3
"""Offline evaluator for AVL-015 Historical-keyframe candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import (base_to_front_rgb_optical, pose_error,
                                           read_manifest, tum_camera_poses)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def summarize(rows):
    n = len(rows)
    available = [r for r in rows if r["pose"] is not None]
    good = sum(bool(r["success"]) for r in rows)
    return {
        "frames": n,
        "pose_available": len(available),
        "coverage": len(available) / n if n else 0.0,
        "success_all_frames": good / n if n else 0.0,
        "success_on_covered": good / len(available) if available else 0.0,
        "correct_count": good,
        "inliers_median": float(np.median([r["inliers"] for r in available])) if available else 0.0,
        "reprojection_median": float(np.median([r["reprojection_px"] for r in available
                                                 if r["reprojection_px"] is not None])) if any(
                                                     r["reprojection_px"] is not None for r in available) else None,
        "elapsed_s": float(sum(r["elapsed_s"] for r in rows)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    p.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    p.add_argument("--trace-root", type=Path,
                   default=Path("outputs/lilocbench/AVL-015/historical_keyframe_memory"))
    p.add_argument("--oracle", type=Path,
                   default=Path("outputs/lilocbench/AVL-004/action_oracle/summary_always.json"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/lilocbench/AVL-015/historical_keyframe_memory/evaluation.json"))
    args = p.parse_args()
    oracle = json.loads(args.oracle.read_text())
    report = {
        "experiment": "AVL-015",
        "component": "Q3_historical_keyframe_relocalization",
        "protocol": {
            "gt_used_only_for_evaluation": True,
            "split": "per-sequence first ceil(40%) validation, remainder test",
            "success_threshold": "0.25m/5deg",
            "causal_requirements": ["max_historical_index < query_index", "query_gt_read=false", "uses_future=false", "selection_uses_gt=false"],
        },
        "sequences": {},
        "aggregate": {},
        "complementarity": {},
        "audit": {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False,
                  "historical_future_access": False},
    }
    all_rows = {"all": [], "validation": [], "test": []}
    action_lookup = {seq: {int(r["query_index"]): r["success"] for r in oracle["sequences"][seq]["rows"]}
                     for seq in SEQUENCES}
    for seq in SEQUENCES:
        _, _, times, _, seq_root = read_manifest(args.data_root, seq)
        gt, _ = tum_camera_poses(args.gt_root / f"{seq}_gt_poses.txt", times,
                                 base_to_front_rgb_optical(seq_root), 0.1)
        trace = [json.loads(x) for x in (args.trace_root / seq / "trace.jsonl").read_text().splitlines() if x.strip()]
        rows = []
        for raw in trace:
            qi = int(raw["query_index"])
            pose = raw.get("pose")
            t = r = None
            success = False
            if pose is not None:
                t, r = pose_error(np.asarray(pose, dtype=np.float64), gt[qi])
                success = bool(t <= 0.25 and r <= 5.0)
            hist = raw.get("historical_keyframes", [])
            max_hist = max((int(x["frame_index"]) for x in hist), default=-1)
            causal = raw.get("causal", {})
            row = {"sequence": seq, "query_index": qi, "pose": pose,
                   "success": success, "translation_error_m": t, "rotation_error_deg": r,
                   "inliers": int(raw.get("evidence", {}).get("inliers", 0)),
                   "reprojection_px": raw.get("evidence", {}).get("median_reprojection_px"),
                   "elapsed_s": float(raw.get("elapsed_s", 0.0)),
                   "historical_count": len(hist), "max_historical_index": max_hist,
                   "causal": causal,
                   "other_actions": action_lookup[seq].get(qi, {})}
            rows.append(row)
            report["audit"]["query_gt_read"] |= bool(causal.get("query_gt_read", False))
            report["audit"]["uses_future"] |= bool(causal.get("uses_future", False))
            report["audit"]["selection_uses_gt"] |= bool(causal.get("selection_uses_gt", False))
            report["audit"]["historical_future_access"] |= not (max_hist < qi or max_hist < 0)
        rows.sort(key=lambda x: x["query_index"])
        cut = int(math.ceil(len(rows) * 0.4))
        report["sequences"][seq] = {"all": summarize(rows), "validation": summarize(rows[:cut]),
                                     "test": summarize(rows[cut:]),
                                     "historical_unavailable": sum(r["pose"] is None for r in rows)}
        all_rows["all"].extend(rows); all_rows["validation"].extend(rows[:cut]); all_rows["test"].extend(rows[cut:])

    for split, rows in all_rows.items():
        report["aggregate"][split] = summarize(rows)
        groups = {"historical_only": 0, "map_or_roma_only": 0, "both": 0, "both_fail": 0}
        for r in rows:
            h = bool(r["success"])
            others = r["other_actions"]
            m = any(bool(others.get(a, False)) for a in ("consensus", "memory", "hloc", "roma"))
            if h and not m: groups["historical_only"] += 1
            elif m and not h: groups["map_or_roma_only"] += 1
            elif h and m: groups["both"] += 1
            else: groups["both_fail"] += 1
        report["complementarity"][split] = groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "complementarity": report["complementarity"], "audit": report["audit"]}, indent=2))


if __name__ == "__main__":
    main()
