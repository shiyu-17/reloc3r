#!/usr/bin/env python3
"""Evaluate a recovery result for every query frame (Always-RoMa/HLoc)."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    _, _, query_times, _, query_root = read_manifest(args.data_root, args.sequence)
    gt, _ = tum_camera_poses(args.gt_dir / f"{args.sequence}_gt_poses.txt", query_times, base_to_front_rgb_optical(args.data_root / args.sequence), 0.1)
    recovery = {int(x["query_index"]): x for x in (json.loads(line) for line in args.recovery.read_text().splitlines() if line.strip())}
    rows = []
    for qi in range(len(query_times)):
        rec = recovery.get(qi, {})
        pred = rec.get("pose")
        metric = None
        if pred is not None:
            t, r = pose_error(np.asarray(pred), gt[qi])
            metric = {"translation_error_m": t, "rotation_error_deg": r, "success_0.25m_5deg": bool(t <= 0.25 and r <= 5.0)}
        rows.append({"query_index": qi, "metric": metric, "pose_available": pred is not None, "query_gt_used_by_route": False, "evidence": {"tool": rec.get("tool", rec.get("model")), "elapsed_s": rec.get("elapsed_s")}})
    valid = [x["metric"] for x in rows if x["metric"] is not None]
    success = [bool(x["metric"] and x["metric"]["success_0.25m_5deg"]) for x in rows]
    summary = {"sequence": args.sequence, "route": {"frames": len(rows), "pose_available": len(valid), "coverage": len(valid) / len(rows), "success_all_frames": sum(success) / len(rows), "success_on_covered": sum(success) / len(valid) if valid else 0.0, "median_translation_error_m": float(np.median([x["translation_error_m"] for x in valid])) if valid else None, "median_rotation_error_deg": float(np.median([x["rotation_error_deg"] for x in valid])) if valid else None}, "query_gt_used_by_route": False, "memory_future_access": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
