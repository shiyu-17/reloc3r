#!/usr/bin/env python3
"""Independent GT-only evaluator for AVL-012 reliability/reject curves."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
SUCCESS_T, SUCCESS_R = 0.25, 5.0


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def runs(bits):
    best = cur = 0
    for b in bits:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def metrics(rows, threshold, start, end):
    part = rows[start:end]
    accepted = [r for r in part if r["route_accepted"] and r["reliability_risk"] <= threshold]
    correct = [r for r in accepted if r.get("success", False)]
    wrong = len(accepted) - len(correct)
    bits = [bool(r.get("success", False)) if r["route_accepted"] and r["reliability_risk"] <= threshold else False for r in part]
    return {
        "frames": len(part), "accepted": len(accepted), "correct_accepted": len(correct),
        "wrong_accepted": wrong, "coverage": len(accepted) / len(part) if part else 0.0,
        "selective_accuracy": len(correct) / len(accepted) if accepted else None,
        "success_all_frames": len(correct) / len(part) if part else 0.0,
        "reject_rate": 1.0 - (len(accepted) / len(part) if part else 0.0),
        "false_accept_rate": wrong / len(accepted) if accepted else None,
        "max_failure": runs(bits),
        "actions_accepted": {a: sum(r["action"] == a and r["route_accepted"] and r["reliability_risk"] <= threshold for r in part)
                             for a in sorted(set(r["action"] for r in part))},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    p.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    p.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    p.add_argument("--evidence-root", type=Path, default=Path("outputs/lilocbench/AVL-012"))
    p.add_argument("--output", type=Path, default=Path("outputs/lilocbench/AVL-012/evaluation.json"))
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    args = p.parse_args()
    curves = {"validation": {}, "test": {}}
    per_sequence = {"validation": {}, "test": {}}
    all_rows = {"validation": [], "test": []}
    audit = {"query_gt_read_in_score": False, "score_uses_future": False,
             "score_uses_selection_gt": False, "trace_flags_clear": True}
    for sequence in SEQUENCES:
        _, _, _, _, query_root = read_manifest(args.data_root, sequence)
        _, _, query_times, _, _ = read_manifest(args.data_root, sequence)
        gt_path = args.gt_root / f"{sequence}_gt_poses.txt"
        query_gt, _ = tum_camera_poses(gt_path, query_times,
                                       base_to_front_rgb_optical(query_root), args.max_gt_delta)
        rows = read_rows(args.evidence_root / sequence / "evidence.jsonl")
        if len(rows) != len(query_gt):
            raise RuntimeError(f"row/GT mismatch for {sequence}: {len(rows)} vs {len(query_gt)}")
        for row, gt in zip(rows, query_gt):
            if row["route_pose"] is None:
                row["success"] = False
                row["translation_error_m"] = None
                row["rotation_error_deg"] = None
            else:
                t, r = pose_error(np.asarray(row["route_pose"], dtype=np.float64), gt)
                row["translation_error_m"] = float(t)
                row["rotation_error_deg"] = float(r)
                row["success"] = bool(t <= SUCCESS_T and r <= SUCCESS_R)
            audit["trace_flags_clear"] &= not (row["query_gt_read"] or row["uses_future"] or row["selection_uses_gt"])
        # This file is explicitly evaluator-only; it is never read by the score branch.
        with (args.evidence_root / sequence / "evaluator_rows.jsonl").open("w") as ef:
            for row in rows:
                ef.write(json.dumps(row, sort_keys=True) + "\n")
        split = int(math.ceil(len(rows) * 0.4))
        for name, start, end in (("validation", 0, split), ("test", split, len(rows))):
            chunk = rows[start:end]
            all_rows[name].extend(chunk)
            per_sequence[name][sequence] = {str(t): metrics(rows, t, start, end) for t in THRESHOLDS}
    for name in ("validation", "test"):
        # Global rows are concatenated by sequence, preserving only fixed split boundaries.
        curves[name] = {str(t): metrics(all_rows[name], t, 0, len(all_rows[name])) for t in THRESHOLDS}
    # Freeze one deployable threshold on validation only: highest selective accuracy
    # with at least 50% coverage; ties prefer greater coverage, then lower threshold.
    eligible = [(float(t), m) for t, m in curves["validation"].items() if m["coverage"] >= 0.5 and m["selective_accuracy"] is not None]
    chosen_t, chosen_m = max(eligible, key=lambda x: (x[1]["selective_accuracy"], x[1]["coverage"], -x[0])) if eligible else (None, None)
    result = {
        "experiment": "AVL-012", "success_threshold": {"translation_m": SUCCESS_T, "rotation_deg": SUCCESS_R},
        "threshold_grid": list(THRESHOLDS), "selection_rule": "validation selective_accuracy max subject to coverage >= 0.50; tie higher coverage then lower threshold",
        "chosen_threshold_validation": chosen_t, "chosen_validation_metrics": chosen_m,
        "validation": curves["validation"], "test": curves["test"], "per_sequence": per_sequence,
        "audit": audit,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chosen_threshold_validation": chosen_t, "validation": curves["validation"], "test": curves["test"], "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
