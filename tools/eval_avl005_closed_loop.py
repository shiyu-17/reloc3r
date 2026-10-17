#!/usr/bin/env python3
"""Post-hoc evaluator for the AVL-005 closed-loop Manager trace."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"protocol": "post-hoc GT evaluator; closed-loop trace never reads query GT", "sequences": {}}
    for seq in SEQUENCES:
        _, _, qt, _, qr = read_manifest(args.data_root, seq)
        gt, _ = tum_camera_poses(args.gt_dir / f"{seq}_gt_poses.txt", qt,
                                 base_to_front_rgb_optical(args.data_root / seq), 0.1)
        traces = [json.loads(line) for line in
                  (args.trace_root / seq / "closed_loop_trace.jsonl").read_text().splitlines() if line.strip()]
        rows = []
        for row in traces:
            metric = None
            if row.get("pose") is not None:
                t, r = pose_error(np.asarray(row["pose"], dtype=np.float64), gt[int(row["query_index"])])
                metric = {"translation_error_m": t, "rotation_error_deg": r,
                          "success_0.25m_5deg": bool(t <= 0.25 and r <= 5.0)}
            rows.append({"query_index": row["query_index"], "action": row["action"],
                         "pose_source": row.get("pose_source"), "accepted": row.get("accepted"),
                         "metric": metric})
        valid = [row["metric"] for row in rows if row["metric"] is not None]
        good = sum(row["success_0.25m_5deg"] for row in valid)
        report["sequences"][seq] = {
            "frames": len(rows), "pose_available": len(valid),
            "coverage": len(valid) / len(rows),
            "success_all_frames": good / len(rows),
            "success_on_covered": good / len(valid) if valid else 0.0,
            "geometry_calls": sum(row["action"] == "call_geometry" for row in rows),
            "geometry_verified": sum(row["pose_source"] == "geometry" for row in rows),
            "sources": {source: sum(row["pose_source"] == source for row in rows)
                        for source in ("consensus", "memory", "geometry", "reject_geometry", None)},
            "rows": rows,
        }
    total = sum(x["frames"] for x in report["sequences"].values())
    report["weighted"] = {
        "frames": total,
        "pose_available": sum(x["pose_available"] for x in report["sequences"].values()),
        "success": sum(x["success_all_frames"] * x["frames"] for x in report["sequences"].values()) / total,
        "geometry_call_rate": sum(x["geometry_calls"] for x in report["sequences"].values()) / total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["weighted"], indent=2))


if __name__ == "__main__":
    main()
