#!/usr/bin/env python3
"""Post-hoc GT-only evaluator for AVL-006-B routing traces."""
import argparse
import json
from pathlib import Path

import numpy as np

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]


def finite_pose(metric):
    return metric is not None and np.isfinite(metric.get("translation_error_m", np.nan)) and np.isfinite(metric.get("rotation_error_deg", np.nan))


def longest_false(values):
    best = cur = 0
    for v in values:
        if not v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def memory_rows(root, seq):
    rows = []
    with (root / seq / "frame_results.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            m = d["methods"]["memory3"]
            rows.append({
                "success": bool(m.get("success_0.25m_5deg", False)),
                "coverage": finite_pose(m),
                "translation_error_m": m.get("translation_error_m"),
                "rotation_error_deg": m.get("rotation_error_deg"),
            })
    return rows


def roma_rows(root, seq):
    d = json.loads((root / f"{seq}.json").read_text())
    rows = []
    for f in d["frames"]:
        m = f.get("metric")
        rows.append({
            "success": bool(m and m.get("success_0.25m_5deg", False)),
            "coverage": bool(f.get("pose_available", False)) and finite_pose(m),
            "translation_error_m": m.get("translation_error_m") if m else None,
            "rotation_error_deg": m.get("rotation_error_deg") if m else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-root", default="outputs/lilocbench/AVL-006/early_router", type=Path)
    ap.add_argument("--memory-root", default="outputs/lilocbench/AVL-003", type=Path)
    ap.add_argument("--roma-root", default="outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0", type=Path)
    ap.add_argument("--output", default="outputs/lilocbench/AVL-006/early_router_eval.json", type=Path)
    args = ap.parse_args()
    policies = [p.name for p in args.router_root.iterdir() if p.is_dir()]
    all_results = {}
    for policy in sorted(policies):
        per_seq = {}
        all_success = []
        all_coverage = []
        total_roma = 0
        for seq in SEQUENCES:
            trace = [json.loads(x) for x in (args.router_root / policy / f"{seq}.jsonl").read_text().splitlines()]
            mem = memory_rows(args.memory_root, seq)
            roma = roma_rows(args.roma_root, seq)
            if not (len(trace) == len(mem) == len(roma)):
                raise RuntimeError(f"length mismatch {policy} {seq}")
            success, coverage, actions = [], [], []
            for i, r in enumerate(trace):
                outcome = roma[i] if r["action"] == "roma" else mem[i]
                success.append(outcome["success"])
                coverage.append(outcome["coverage"])
                actions.append(r["action"])
            total_roma += sum(a == "roma" for a in actions)
            all_success.extend(success); all_coverage.extend(coverage)
            per_seq[seq] = {
                "frames": len(trace),
                "roma_calls": sum(a == "roma" for a in actions),
                "roma_call_rate": sum(a == "roma" for a in actions) / len(actions),
                "coverage": sum(coverage) / len(coverage),
                "success": sum(success) / len(success),
                "success_count": int(sum(success)),
                "coverage_count": int(sum(coverage)),
                "longest_failure_frames": longest_false(success),
                "query_gt_read_by_router": all(not r.get("query_gt_read", True) for r in trace),
                "selection_uses_gt_by_router": any(r.get("selection_uses_gt", True) for r in trace),
            }
        all_results[policy] = {
            "frames": len(all_success),
            "roma_calls": total_roma,
            "roma_call_rate": total_roma / len(all_success),
            "coverage": sum(all_coverage) / len(all_coverage),
            "success": sum(all_success) / len(all_success),
            "success_count": int(sum(all_success)),
            "coverage_count": int(sum(all_coverage)),
            "longest_failure_frames": longest_false(all_success),
            "per_sequence": per_seq,
            "router_gt_audit_pass": all(not v["selection_uses_gt_by_router"] and v["query_gt_read_by_router"] for v in per_seq.values()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "GT only read here after frozen route traces; memory3 vs fixed seed0 Always-RoMa",
        "policies": all_results,
        "sequence_order": SEQUENCES,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
