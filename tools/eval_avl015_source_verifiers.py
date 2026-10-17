#!/usr/bin/env python3
"""AVL-015 Q2 exploratory source-specific verifier calibration.

The score is computed only from evidence fields that were available to the
causal AVL-012/013 trace.  GT is joined afterwards by this evaluator solely to
measure selective accuracy, coverage, and false acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
SOURCES = ("consensus", "memory", "geometry")


def source_name(row):
    src = row.get("pose_source")
    if src in ("consensus", "consensus_after_geometry_reject"):
        return "consensus"
    if src in ("memory", "memory_after_geometry_reject"):
        return "memory"
    if src in ("geometry", "geometry_recovery"):
        return "geometry"
    return None


def direct_score(e):
    """Return source-specific evidence risk; no outcome fields are used."""
    src = source_name(e)
    if src == "consensus":
        return max(
            float(e.get("consensus_dispersion_m", 0.0)) / 1.5,
            float(e.get("consensus_dispersion_deg", 0.0)) / 15.0,
            float(e.get("top10_candidate_dispersion_m", 0.0)) / 1.5,
            float(e.get("top10_candidate_dispersion_deg", 0.0)) / 15.0,
            float(e.get("temporal_risk", 0.0)),
        )
    if src == "memory":
        return max(
            float(e.get("memory_dispersion_m", 0.0)) / 1.5,
            float(e.get("memory_dispersion_deg", 0.0)) / 15.0,
            float(e.get("temporal_risk", 0.0)),
        )
    if src == "geometry":
        pair_ratio = float(e.get("roma_pair_support_ratio", 0.0))
        # New traces expose disjoint components.  Keep the fallback for old
        # AVL-012/013 artifacts so historical evaluation remains readable, but
        # do not reintroduce the coupled geometry_risk into new re-trust runs.
        intrinsic = float(e.get("geometry_intrinsic_risk",
                               e.get("geometry_risk", 0.0)))
        cross_source = float(e.get("geometry_cross_source_risk", max(
            float(e.get("geometry_vs_consensus_m", 0.0)) / 2.0,
            float(e.get("geometry_vs_consensus_deg", 0.0)) / 30.0,
            float(e.get("geometry_vs_memory_m", 0.0)) / 2.0,
            float(e.get("geometry_vs_memory_deg", 0.0)) / 30.0)))
        previous = float(e.get("geometry_previous_state_risk", max(
            float(e.get("geometry_vs_previous_m", 0.0)) / 2.0,
            float(e.get("geometry_vs_previous_deg", 0.0)) / 30.0)))
        return max(
            intrinsic, cross_source, previous,
            max(0.0, 1.0 - pair_ratio),
        )
    return None


def metrics(rows, threshold):
    accepted = [r for r in rows if r["score"] <= threshold]
    correct = sum(bool(r["success"]) for r in accepted)
    return {
        "frames": len(rows),
        "accepted": len(accepted),
        "correct_accepted": correct,
        "wrong_accepted": len(accepted) - correct,
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "selective_accuracy": correct / len(accepted) if accepted else 0.0,
        "false_accept_rate": (len(accepted) - correct) / len(accepted) if accepted else 0.0,
        "full_frame_success": correct / len(rows) if rows else 0.0,
    }


def choose_threshold(validation_rows):
    candidates = []
    for t in THRESHOLDS:
        m = metrics(validation_rows, t)
        # Per-source calibration cannot select a trivial near-zero coverage
        # point.  The 20% floor is pre-registered for this exploratory Q2
        # report and is not a claim about the final system's coverage target.
        if m["coverage"] >= 0.20:
            candidates.append((m["selective_accuracy"], m["coverage"], -t, t, m))
    if not candidates:
        return None, metrics(validation_rows, max(THRESHOLDS))
    candidates.sort(reverse=True)
    return candidates[0][3], candidates[0][4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("outputs/lilocbench/AVL-014/AVL-012_selective"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/lilocbench/AVL-015/source_verification/summary.json"))
    args = p.parse_args()
    report = {
        "experiment": "AVL-015",
        "component": "Q2_source_specific_verification",
        "protocol": {
            "evidence_only_score": True,
            "thresholds": list(THRESHOLDS),
            "validation_selection": "per-source maximum selective accuracy subject to coverage>=0.20; tie higher coverage then lower threshold",
            "source_scores": {
                "consensus": "max(consensus_dispersion_m/1.5, consensus_dispersion_deg/15, top10_dispersion_m/1.5, top10_dispersion_deg/15, temporal_risk)",
                "memory": "max(memory_dispersion_m/1.5, memory_dispersion_deg/15, temporal_risk)",
                "geometry": "max(geometry_risk, cross-source pose disagreement, 1-pair_support_ratio)",
            },
            "query_gt_read_by_route": False,
            "uses_future_by_route": False,
            "selection_uses_gt_by_route": False,
            "dynamic_retrust": "evaluated separately by full-frame evidence replay; not an online route regeneration",
        },
        "sources": {},
    }
    rows_by_source = {s: {"validation": [], "test": [], "all": []} for s in SOURCES}
    for seq in SEQUENCES:
        evidence = {int(json.loads(line)["query_index"]): json.loads(line)
                    for line in (args.root / seq / "evidence.jsonl").read_text().splitlines() if line.strip()}
        evaluated = {int(json.loads(line)["query_index"]): json.loads(line)
                     for line in (args.root / seq / "evaluator_rows.jsonl").read_text().splitlines() if line.strip()}
        indices = sorted(evidence)
        cut = int(math.ceil(len(indices) * 0.4))
        for pos, qi in enumerate(indices):
            e, m = evidence[qi], evaluated[qi]
            src = source_name(e)
            if src not in SOURCES or not m.get("route_accepted", False):
                continue
            row = {"sequence": seq, "query_index": qi, "source": src,
                   "score": direct_score(e), "success": bool(m.get("success", False)),
                   "route_risk": float(e.get("reliability_risk", 0.0)),
                   "reject_streak": None}
            split = "validation" if pos < cut else "test"
            rows_by_source[src][split].append(row)
            rows_by_source[src]["all"].append(row)

    for src in SOURCES:
        val, test, all_rows = (rows_by_source[src][x] for x in ("validation", "test", "all"))
        threshold, chosen_val = choose_threshold(val)
        report["sources"][src] = {
            "frames": {"all": len(all_rows), "validation": len(val), "test": len(test)},
            "chosen_threshold": threshold,
            "validation_chosen": chosen_val,
            "curves": {
                "validation": {str(t): metrics(val, t) for t in THRESHOLDS},
                "test": {str(t): metrics(test, t) for t in THRESHOLDS},
            },
            "frozen_test": metrics(test, threshold) if threshold is not None else None,
            "baseline_unified_risk_test": metrics(
                [{**r, "score": r["route_risk"]} for r in test], 1.5),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({s: {"threshold": report["sources"][s]["chosen_threshold"],
                          "test": report["sources"][s]["frozen_test"]}
                      for s in SOURCES}, indent=2))


if __name__ == "__main__":
    main()
