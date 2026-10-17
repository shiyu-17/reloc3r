#!/usr/bin/env python3
"""Independent GT evaluator for the AVL-011 deterministic recovery trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical,
    pose_error,
    read_manifest,
    tum_camera_poses,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def summarize(rows):
    success = [bool(row["metric"] and row["metric"]["success_0.25m_5deg"]) for row in rows]
    runs, current = [], 0
    for ok in success:
        if ok:
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    recovery = []
    for i, ok in enumerate(success):
        if ok and i:
            j = i - 1
            while j >= 0 and not success[j]:
                j -= 1
            if j < i - 1:
                recovery.append(i - j - 1)
    available = sum(row["metric"] is not None for row in rows)
    good = sum(success)
    return {
        "frames": len(rows),
        "pose_available": available,
        "coverage": available / len(rows) if rows else 0.0,
        "success_all_frames": good / len(rows) if rows else 0.0,
        "success_on_covered": good / available if available else 0.0,
        "max_failure_frames": max(runs, default=0),
        "mean_failure_frames": float(np.mean(runs)) if runs else 0.0,
        "failure_run_count": len(runs),
        "recovery_latency_frames_mean": float(np.mean(recovery)) if recovery else 0.0,
        "recovery_latency_frames_median": float(np.median(recovery)) if recovery else 0.0,
        "recovery_latency_count": len(recovery),
        "geometry_calls": sum(row["action"] == "call_geometry" for row in rows),
        "geometry_verified": sum(row.get("geometry_verified", False) for row in rows),
        "geometry_call_rate": sum(row["action"] == "call_geometry" for row in rows) / len(rows) if rows else 0.0,
        "reject_rate": sum(row.get("pose_source") in (None, "reject_geometry") for row in rows) / len(rows) if rows else 0.0,
        "source_counts": {str(source): sum(row.get("pose_source") == source for row in rows)
                          for source in ("consensus", "consensus_after_geometry_reject", "memory",
                                         "memory_after_geometry_reject", "geometry", "reject_geometry", None)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = p.parse_args()

    report = {
        "experiment": "AVL-011",
        "protocol": "GT-only evaluator; online trace is causal and never reads query GT/future frames",
        "sequences": {},
        "validation": {},
        "test": {},
        "audit": {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False},
    }
    for sequence in args.sequences:
        _, _, query_times, _, query_root = read_manifest(args.data_root, sequence)
        gt, _ = tum_camera_poses(args.gt_dir / f"{sequence}_gt_poses.txt", query_times,
                                 base_to_front_rgb_optical(query_root), 0.1)
        trace = [json.loads(line) for line in
                 (args.trace_root / sequence / "closed_loop_trace.jsonl").read_text().splitlines()
                 if line.strip()]
        rows = []
        for row in trace:
            pose = row.get("pose")
            metric = None
            if pose is not None:
                t, r = pose_error(np.asarray(pose, dtype=np.float64), gt[int(row["query_index"])])
                metric = {"translation_error_m": float(t), "rotation_error_deg": float(r),
                          "success_0.25m_5deg": bool(t <= 0.25 and r <= 5.0)}
            rows.append({"query_index": int(row["query_index"]), "action": row["action"],
                         "pose_source": row.get("pose_source"), "geometry_verified": row.get("geometry_verified", False),
                         "metric": metric})
        n = len(rows)
        cut = min(n, int(math.ceil(len(query_times) * 0.4)))
        report["sequences"][sequence] = summarize(rows)
        report["validation"][sequence] = summarize(rows[:cut])
        report["test"][sequence] = summarize(rows[cut:]) if cut < n else None
        report["audit"][sequence] = {
            "trace_rows": n,
            "query_gt_read": any(row.get("query_gt_read", False) for row in trace),
            "uses_future": any(row.get("uses_future", False) for row in trace),
            "selection_uses_gt": any(row.get("selection_uses_gt", False) for row in trace),
        }

    def aggregate(split):
        parts = [x for x in report[split].values() if x is not None]
        total = sum(x["frames"] for x in parts)
        return {
            "frames": total,
            "pose_available": sum(x["pose_available"] for x in parts),
            "coverage": sum(x["pose_available"] for x in parts) / total if total else 0.0,
            "success_all_frames": sum(x["success_all_frames"] * x["frames"] for x in parts) / total if total else 0.0,
            "success_on_covered": sum(x["success_all_frames"] * x["frames"] for x in parts) / sum(x["pose_available"] for x in parts) if parts and sum(x["pose_available"] for x in parts) else 0.0,
            "geometry_call_rate": sum(x["geometry_calls"] for x in parts) / total if total else 0.0,
            "max_failure_frames": max((x["max_failure_frames"] for x in parts), default=0),
            "recovery_latency_frames_mean": float(np.mean([x["recovery_latency_frames_mean"] for x in parts])) if parts else 0.0,
        }

    report["validation_aggregate"] = aggregate("validation")
    report["test_aggregate"] = aggregate("test")
    report["all_aggregate"] = aggregate("sequences")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": report["validation_aggregate"], "test": report["test_aggregate"],
                      "all": report["all_aggregate"], "audit": report["audit"]}, indent=2))


if __name__ == "__main__":
    main()
