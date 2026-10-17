#!/usr/bin/env python3
"""Independent GT evaluator and cost ledger for AVL-010 DPVO traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_avl009_sequence import (  # noqa: E402
    METHODS as OLD_METHODS,
    aggregate_metrics,
    load_always_roma,
    metrics_from_always_roma,
    metrics_from_predictions,
)
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, read_manifest, tum_camera_poses  # noqa: E402

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
METHODS = ("memory3", "dpvo_temporal_only", "global_dpvo", "global_dpvo_memory",
           "global_dpvo_memory_roma", "always_roma")


def load_trace(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def method_data(rows, method):
    entries = [r["methods"][method] for r in rows]
    predictions = [np.asarray(e["pose"], dtype=np.float64) if e.get("pose") is not None else None for e in entries]
    actions = [e.get("selected_action") for e in entries]
    sources = [e.get("prediction_source") for e in entries]
    return predictions, actions, sources


def aggregate_any(per_sequence):
    output = {}
    for method in METHODS:
        entries = [v[method] for v in per_sequence.values() if method in v]
        if not entries:
            continue
        frames = sum(e["frames"] for e in entries)
        means = [e["recovery_latency_frames"]["mean"] for e in entries
                 if e["recovery_latency_frames"]["mean"] is not None]
        output[method] = {
            "frames": frames,
            "pose_available": sum(e["pose_available"] for e in entries),
            "coverage": sum(e["pose_available"] for e in entries) / frames,
            "success_count": sum(e["success_count"] for e in entries),
            "success_rate_all_frames": sum(e["success_count"] for e in entries) / frames,
            "max_failure_frames": max(e["max_failure_frames"] for e in entries),
            "mean_failure_frames_unweighted": float(np.mean([e["mean_failure_frames"] for e in entries])),
            "median_failure_frames_unweighted": float(np.median([e["median_failure_frames"] for e in entries])),
            "failure_run_count": sum(e["failure_run_count"] for e in entries),
            "recovery_latency_mean_unweighted": float(np.mean(means)) if means else None,
            "reject_rate_frame_weighted": sum(e["reject_rate"] * e["frames"] for e in entries) / frames,
            "global_call_rate_frame_weighted": sum(e["global_call_rate"] * e["frames"] for e in entries) / frames,
            "temporal_call_rate_frame_weighted": sum(e["temporal_call_rate"] * e["frames"] for e in entries) / frames,
            "roma_call_rate_frame_weighted": sum(e["roma_call_rate"] * e["frames"] for e in entries) / frames,
        }
    return output


def cost_ledger(args, rows_by_sequence, seq_frames):
    tracker = {}
    tracker_total = 0.0
    for seq in SEQUENCES:
        manifest = json.loads((args.dpvo_root / seq / "manifest.json").read_text())
        tracker[seq] = {"sum_s": float(manifest["total_tracking_elapsed_s"]),
                        "mean_s": float(manifest["mean_tracking_elapsed_s"]),
                        "measurement": "measured online DPVO call wall time with CUDA synchronization",
                        "terminate_called": False}
        tracker_total += tracker[seq]["sum_s"]
    old = json.loads((args.cost_root / "summary.json").read_text())
    global_est = float(old.get("comparison", {}).get("manager", {}).get("reloc3r_estimated_s", 0.0))
    roma_always = float(old.get("comparison", {}).get("always_roma", {}).get("total_s", 0.0))
    # Use the recorded fixed-seed recovery rows to count actual selected calls.
    roma_calls = 0
    roma_recovery_s = 0.0
    for seq, rows in rows_by_sequence.items():
        raw = {}
        p = args.roma_root / f"{seq}_top100_merged.jsonl"
        for line in p.read_text().splitlines():
            x = json.loads(line); raw[int(x["query_index"])] = x
        for r in rows:
            e = r["methods"]["global_dpvo_memory_roma"]
            if e.get("selected_action") == "call_roma":
                roma_calls += 1
                roma_recovery_s += float(raw.get(int(r["query_index"]), {}).get("elapsed_s", 0.0))
    return {
        "protocol": "AVL-010 serial accounting; tracker is measured, inherited Reloc3r is AVL-005 transparent estimate",
        "dpvo": {"sum_s": tracker_total, "by_sequence": tracker, "gpu_hours": tracker_total / 3600.0},
        "global_reloc3r_inherited_estimate_s": global_est,
        "retrieval_inherited_s": float(old.get("comparison", {}).get("retrieval_total_s", 0.0)),
        "roma_recovery": {"calls": roma_calls, "cached_elapsed_s": roma_recovery_s,
                          "measurement": "fixed-seed no-GT cache selected by causal trace"},
        "always_roma_reference_s": roma_always,
        "known_components_with_dpvo_s": tracker_total + (global_est or 0.0) + roma_recovery_s,
        "warning": "Global/RoMa inherited components remain mixed measured/estimated; no speedup claim from this ledger alone.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--cost-root", type=Path, required=True)
    p.add_argument("--roma-root", type=Path, required=True)
    p.add_argument("--roma-timed-root", type=Path, required=True)
    p.add_argument("--dpvo-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "test", "all", "failure_analysis", "cost"):
        (args.output_dir / name).mkdir(exist_ok=True)
    (args.output_dir / "audit").mkdir(exist_ok=True)

    results = {"validation": {}, "test": {}, "all": {}}
    rows_by_sequence = {}
    sequence_info = {}
    for seq in SEQUENCES:
        _, _, times, _, query_root = read_manifest(args.data_root, seq)
        gt, delta = tum_camera_poses(args.gt_dir / f"{seq}_gt_poses.txt", times,
                                     base_to_front_rgb_optical(query_root), args.max_gt_delta)
        rows = load_trace(args.trace_root / f"{seq}.jsonl")
        rows_by_sequence[seq] = rows
        n = len(rows); split = int(math.ceil(0.4 * n))
        sequence_info[seq] = {"frames": n, "validation_frames": split, "test_frames": n - split,
                              "gt_match_delta_max_s": float(delta.max())}
        per = {"validation": {}, "test": {}, "all": {}}
        for method in METHODS[:-1]:
            preds, actions, sources = method_data(rows, method)
            per["validation"][method] = metrics_from_predictions(preds, gt, times, 0, split, actions, sources)
            per["test"][method] = metrics_from_predictions(preds, gt, times, split, n, actions, sources)
            per["all"][method] = metrics_from_predictions(preds, gt, times, 0, n, actions, sources)
        roma = load_always_roma(args.cost_root / "eval_roma_seed0" / f"{seq}.json",
                                args.roma_timed_root / f"{seq}.jsonl")
        for name, a, b in (("validation", 0, split), ("test", split, n), ("all", 0, n)):
            per[name]["always_roma"] = metrics_from_always_roma(roma, times, a, b)
        for name in per:
            entry = {"experiment": "AVL-010", "sequence": seq, "split": name,
                     "thresholds_frozen": True, "same_scene_temporal_holdout": True,
                     "query_gt_read_by_online_trace": False, "methods": per[name]}
            results[name][seq] = entry
            (args.output_dir / name / f"{seq}.json").write_text(json.dumps(entry, indent=2, allow_nan=True) + "\n")

    for name in results:
        results[name]["aggregate"] = aggregate_any({s: results[name][s]["methods"] for s in SEQUENCES})
    failure = {label: {m: results["test"][seq]["methods"].get(m) for m in METHODS}
               for label, seq in (("static", "static_0"), ("dynamic", "dynamics_0"),
                                  ("long_term_change", "lt_changes_0"), ("dynamic_plus_long_term", "lt_changes_dynamics_0"))}
    (args.output_dir / "failure_analysis" / "condition_test_metrics.json").write_text(
        json.dumps({"experiment": "AVL-010", "conditions": failure,
                    "interpretation": "sequence labels are operational condition groups, not causal semantic GT"}, indent=2, allow_nan=True) + "\n")
    ledger = cost_ledger(args, rows_by_sequence, sequence_info)
    (args.output_dir / "cost" / "summary.json").write_text(json.dumps(ledger, indent=2, allow_nan=True) + "\n")
    summary = {"experiment": "AVL-010", "protocol": {"validation_fraction": 0.4, "test_fraction": 0.6,
                                                           "thresholds_frozen_before_test": True, "query_gt_online": False,
                                                           "future_frame_online": False, "no_agent": True,
                                                           "dpvo_causal_online_output": True},
               "sequences": sequence_info, "validation": results["validation"], "test": results["test"],
               "all": results["all"], "cost": ledger}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    audit_rows = []
    for seq, rows in rows_by_sequence.items():
        flags_ok = all((r.get("query_gt_read") is False and r.get("uses_future") is False and
                        r.get("selection_uses_gt") is False and
                        all(e.get("query_gt_read") is False and e.get("uses_future") is False and
                            e.get("selection_uses_gt") is False for e in r.get("methods", {}).values()))
                       for r in rows)
        dpvo_manifest = json.loads((args.dpvo_root / seq / "manifest.json").read_text())
        audit_rows.append({"sequence": seq, "trace_rows": len(rows), "trace_flags_ok": bool(flags_ok),
                           "dpvo_full_frame_count": dpvo_manifest["frame_count"],
                           "dpvo_valid_online_pose_count": dpvo_manifest["valid_online_pose_count"],
                           "dpvo_terminate_called": dpvo_manifest["terminate_called"],
                           "dpvo_query_gt_read": dpvo_manifest["query_gt_read"],
                           "dpvo_future_frame_read": dpvo_manifest["future_frame_read"]})
    (args.output_dir / "audit" / "causal_audit.json").write_text(json.dumps({
        "experiment": "AVL-010", "all_trace_flags_ok": all(x["trace_flags_ok"] for x in audit_rows),
        "all_dpvo_terminate_false": all(not x["dpvo_terminate_called"] for x in audit_rows),
        "rows": audit_rows}, indent=2) + "\n")
    print(json.dumps({"validation": results["validation"]["aggregate"], "test": results["test"]["aggregate"],
                      "cost": ledger}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
