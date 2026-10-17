#!/usr/bin/env python3
"""Post-hoc analysis of the fixed AVL-004 recovery route.

The route and recovery tools are no-GT.  This script is an evaluator only: it
reads their saved outputs, computes failure/latency/tool-agreement statistics,
and writes a compact JSON artifact for the pre-Agent decision.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def longest_false_run(values):
    best = cur = 0
    for value in values:
        if not value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def rotation_angle_deg(r):
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return {int(row["query_index"]): row for row in (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())}


def route_stats(route):
    rows = route["frames"]
    success = [bool(r["metric"] and r["metric"]["success_0.25m_5deg"]) for r in rows]
    covered = [r for r in rows if r["metric"] is not None]
    errors_t = [r["metric"]["translation_error_m"] for r in covered]
    errors_r = [r["metric"]["rotation_error_deg"] for r in covered]
    return {
        "frames": len(rows),
        "pose_available": len(covered),
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "success_all_frames": sum(success) / len(rows) if rows else 0.0,
        "success_on_covered": sum(success) / len(covered) if covered else 0.0,
        "median_translation_error_m": float(np.median(errors_t)) if errors_t else None,
        "median_rotation_error_deg": float(np.median(errors_r)) if errors_r else None,
        "longest_failure_run_frames": longest_false_run(success),
        "false_accept_rate_on_covered": 1.0 - (sum(success) / len(covered) if covered else 0.0),
    }


def action_stats(route, action):
    rows = [r for r in route["frames"] if r["action"] == action]
    return route_stats({"frames": rows})


def recovery_stats(recovery, route):
    rows = list(recovery.values())
    geometry_metrics = {int(r["query_index"]): r for r in route["frames"] if r["action"] == "call_geometry"}
    success = [bool(geometry_metrics.get(int(r["query_index"]), {}).get("metric", {}) and geometry_metrics[int(r["query_index"])] ["metric"]["success_0.25m_5deg"]) for r in rows]
    elapsed = [float(r["elapsed_s"]) for r in rows if r.get("elapsed_s") is not None]
    return {
        "frames": len(rows),
        "pose_available": sum(r.get("pose") is not None for r in rows),
        "success_on_geometry_calls": sum(success) / len(success) if success else 0.0,
        "longest_failure_run_in_query_order": longest_false_run(success),
        "elapsed_s_median": percentile(elapsed, 50),
        "elapsed_s_p95": percentile(elapsed, 95),
        "elapsed_s_total": float(sum(elapsed)),
        "inliers_median": percentile([float(r["inliers"]) for r in rows if r.get("inliers") is not None], 50),
        "reprojection_px_median": percentile([float(r["median_reprojection_px"]) for r in rows if r.get("median_reprojection_px") is not None], 50),
        "query_gt_read_all_false": all(r.get("query_gt_read") is False for r in rows),
        "selection_uses_gt_all_false": all(r.get("selection_uses_gt") is False for r in rows),
    }


def tool_agreement(hloc, roma, hloc_route, roma_route):
    common = sorted(set(hloc) & set(roma))
    pose_pairs = []
    success_confusion = {"both_success": 0, "hloc_only": 0, "roma_only": 0, "neither": 0}
    same_refs = 0
    for qi in common:
        hp, rp = hloc[qi].get("pose"), roma[qi].get("pose")
        if hp is not None and rp is not None:
            h = np.asarray(hp, dtype=float)
            r = np.asarray(rp, dtype=float)
            pose_pairs.append({"query_index": qi, "translation_delta_m": float(np.linalg.norm(h[:3, 3] - r[:3, 3])), "rotation_delta_deg": rotation_angle_deg(h[:3, :3] @ r[:3, :3].T)})
        if hloc[qi].get("selected_refs") == roma[qi].get("reference_indices"):
            same_refs += 1
        hs = next((x for x in hloc_route["frames"] if int(x["query_index"]) == qi), {}).get("metric")
        rs = next((x for x in roma_route["frames"] if int(x["query_index"]) == qi), {}).get("metric")
        hb = bool(hs and hs["success_0.25m_5deg"])
        rb = bool(rs and rs["success_0.25m_5deg"])
        key = "both_success" if hb and rb else "hloc_only" if hb else "roma_only" if rb else "neither"
        success_confusion[key] += 1
    return {
        "common_geometry_frames": len(common),
        "same_selected_references": same_refs,
        "same_reference_fraction": same_refs / len(common) if common else 0.0,
        "success_confusion": success_confusion,
        "pose_delta_translation_m_median": percentile([x["translation_delta_m"] for x in pose_pairs], 50),
        "pose_delta_translation_m_p95": percentile([x["translation_delta_m"] for x in pose_pairs], 95),
        "pose_delta_rotation_deg_median": percentile([x["rotation_delta_deg"] for x in pose_pairs], 50),
        "pose_delta_rotation_deg_p95": percentile([x["rotation_delta_deg"] for x in pose_pairs], 95),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route-dir", type=Path, required=True)
    p.add_argument("--hloc-dir", type=Path, required=True)
    p.add_argument("--roma-dir", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"protocol": "AVL-004 A.5 full no-GT Manager route; GT is evaluator-only", "sequences": {}}
    for seq in args.sequences:
        hroute = load_json(args.route_dir / f"hloc_{seq}.json")
        rroute = load_json(args.route_dir / f"roma_{seq}.json")
        hrec = load_jsonl(args.hloc_dir / f"{seq}_top100_merged.jsonl")
        rrec = load_jsonl(args.roma_dir / f"{seq}_top100_merged.jsonl")
        result["sequences"][seq] = {
            "hloc": {"route": route_stats(hroute), "actions": {a: action_stats(hroute, a) for a in ("accept_consensus", "use_memory", "call_geometry", "reject")}, "recovery": recovery_stats(hrec, hroute)},
            "roma": {"route": route_stats(rroute), "actions": {a: action_stats(rroute, a) for a in ("accept_consensus", "use_memory", "call_geometry", "reject")}, "recovery": recovery_stats(rrec, rroute)},
            "tool_agreement": tool_agreement(hrec, rrec, hroute, rroute),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
