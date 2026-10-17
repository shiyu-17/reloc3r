#!/usr/bin/env python3
"""Aggregate AVL-005 stage timings without reading query GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"sum_s": float(values.sum()), "mean_s": float(values.mean()),
            "median_s": float(np.median(values)), "p95_s": float(np.percentile(values, 95)),
            "calls": int(len(values))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--retrieval-profile", type=Path, required=True)
    p.add_argument("--manager-trace-root", type=Path, required=True)
    p.add_argument("--reloc-timing", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    retrieval = json.loads(args.retrieval_profile.read_text())
    report = {"protocol": {
        "query_gt_read": False,
        "note": "same mapping-only database; HLoc/RoMa full-frame timing rows; manager stage rows are no-GT replay",
        "reloc3r": "candidate aggregation timing only; neural Reloc3r inference is reported separately when available",
    }, "sequences": {}}
    weighted = {"frames": 0, "manager_geometry_calls": 0, "manager": {}, "always_roma": {}}
    for seq in SEQUENCES:
        hloc = rows(args.root / "hloc_timed" / f"{seq}.jsonl")
        roma = rows(args.root / "roma_timed" / f"{seq}.jsonl")
        trace = rows(args.manager_trace_root / seq / "closed_loop_trace.jsonl")
        h = {key: stats([r[key] for r in hloc]) for key in
             ("query_extract_s", "rerank_s", "geometry_s", "elapsed_s")}
        r = {key: stats([r[key] for r in roma]) for key in
             ("matching_s", "triangulation_s", "pnp_s", "elapsed_s")}
        call_indices = {int(x["query_index"]) for x in trace if x["action"] == "call_geometry"}
        roma_by_q = {int(x["query_index"]): x for x in roma}
        hloc_by_q = {int(x["query_index"]): x for x in hloc}
        manager_geometry = [roma_by_q[i]["elapsed_s"] for i in sorted(call_indices) if i in roma_by_q]
        manager_query_extract = [hloc_by_q[i]["query_extract_s"] for i in sorted(call_indices) if i in hloc_by_q]
        manager_rerank = [hloc_by_q[i]["rerank_s"] for i in sorted(call_indices) if i in hloc_by_q]
        manager = {
            "frames": len(trace),
            "geometry_calls": len(call_indices),
            "geometry_call_rate": len(call_indices) / len(trace),
            "geometry": stats(manager_geometry),
            "top100_rerank": stats(manager_rerank),
            "top100_query_extract": stats(manager_query_extract),
            "manager_candidate_processing": stats([x["timings"]["reloc3r_candidate_processing_s"] for x in trace]),
            "memory": stats([x["timings"]["memory_s"] for x in trace]),
            "verification": stats([x["timings"]["verification_s"] for x in trace]),
            "geometry_missing_rows": len(call_indices) - len(manager_geometry),
            "rerank_missing_rows": len(call_indices) - len(manager_rerank),
        }
        report["sequences"][seq] = {"frames": len(roma), "hloc": h, "roma": r, "manager": manager,
            "retrieval": retrieval["sequences"][seq], "query_gt_read": False}
        weighted["frames"] += len(roma); weighted["manager_geometry_calls"] += len(call_indices)
        for name, source in (("always_roma", r),):
            for stage, s in source.items(): weighted[name].setdefault(stage, []).extend(
                [s["sum_s"]])
        for key in ("geometry", "top100_rerank", "top100_query_extract", "manager_candidate_processing", "memory", "verification"):
            weighted["manager"].setdefault(key, 0.0)
            weighted["manager"][key] += manager[key]["sum_s"]
    weighted["manager"]["geometry_call_rate"] = weighted["manager_geometry_calls"] / weighted["frames"]
    weighted["manager"]["geometry_missing_rows"] = sum(report["sequences"][s]["manager"]["geometry_missing_rows"] for s in SEQUENCES)
    weighted["manager"]["rerank_missing_rows"] = sum(report["sequences"][s]["manager"]["rerank_missing_rows"] for s in SEQUENCES)
    for key, vals in weighted["always_roma"].items():
        weighted["always_roma"][key] = {"sum_s": float(sum(vals))}
    retrieval_total = float(retrieval["mapping_descriptor_s"] +
                            sum(x["total_s"] for x in retrieval["sequences"].values()))
    comparison = {"retrieval_total_s": retrieval_total,
                  "always_roma": {"retrieval_s": retrieval_total,
                    "top100_query_extract_s": sum(report["sequences"][s]["hloc"]["query_extract_s"]["sum_s"] for s in SEQUENCES),
                    "top100_rerank_s": sum(report["sequences"][s]["hloc"]["rerank_s"]["sum_s"] for s in SEQUENCES),
                    "geometry_s": weighted["always_roma"]["elapsed_s"]["sum_s"]},
                  "manager": {"retrieval_s": retrieval_total,
                    "top100_query_extract_s": weighted["manager"]["top100_query_extract"],
                    "top100_rerank_s": weighted["manager"]["top100_rerank"],
                    "geometry_s": weighted["manager"]["geometry"],
                    "candidate_processing_s": weighted["manager"]["manager_candidate_processing"],
                    "memory_s": weighted["manager"]["memory"],
                    "verification_s": weighted["manager"]["verification"]}}
    if args.reloc_timing:
        reloc = json.loads(args.reloc_timing.read_text())
        comparison["manager"]["reloc3r_estimated_s"] = sum(
            (reloc["sequences"][s]["mean_preprocess_s"] + reloc["sequences"][s]["mean_inference_s"])
            * report["sequences"][s]["frames"] for s in SEQUENCES)
        comparison["manager"]["reloc3r_timing_protocol"] = "mean of 8 fixed temporal samples per sequence, scaled to frame count"
    comparison["always_roma"]["total_s"] = sum(v for k,v in comparison["always_roma"].items() if k.endswith("_s"))
    comparison["manager"]["total_s"] = sum(v for k,v in comparison["manager"].items() if k.endswith("_s") and k != "reloc3r_timing_protocol")
    comparison["manager"]["geometry_call_rate"] = weighted["manager"]["geometry_call_rate"]
    comparison["always_roma"]["geometry_call_rate"] = 1.0
    report["comparison"] = comparison
    report["weighted"] = weighted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"weighted": weighted, "sequences": {
        s: {"frames": report["sequences"][s]["frames"],
            "manager_geometry_calls": report["sequences"][s]["manager"]["geometry_calls"],
            "roma_sum_s": report["sequences"][s]["roma"]["elapsed_s"]["sum_s"],
            "hloc_sum_s": report["sequences"][s]["hloc"]["elapsed_s"]["sum_s"]}
        for s in SEQUENCES}}, indent=2))


if __name__ == "__main__":
    main()
