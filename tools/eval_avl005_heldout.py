#!/usr/bin/env python3
"""Validation/test split for a frozen risk-gated geometry policy.

The threshold is selected from validation success labels only.  Test labels
are read after the threshold is frozen and never affect selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def rows(path):
    return {int(x["query_index"]): x for x in
            (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def consensus_success(path, qi):
    return bool(rows(path)[qi]["methods"]["consensus3"]["success_0.25m_5deg"])


def evaluate_sequence(args, seq, threshold):
    risk = rows(args.risk_root / seq / "risk_rows.jsonl")
    base = rows(args.avl003_root / seq / "frame_results.jsonl")
    roma = {int(x["query_index"]): x for x in
            json.loads((args.always_route / f"roma_{seq}.json").read_text())["frames"]}
    outcomes = []
    calls = 0
    for qi in sorted(risk):
        high = float(risk[qi]["risk_score"]) > threshold
        if high:
            calls += 1
            ok = bool(roma[qi].get("metric") and roma[qi]["metric"]["success_0.25m_5deg"])
            source = "roma"
        else:
            ok = bool(base[qi]["methods"]["consensus3"]["success_0.25m_5deg"])
            source = "consensus3"
        outcomes.append({"query_index": qi, "risk_score": float(risk[qi]["risk_score"]),
                         "action": source, "success": ok})
    return {"frames": len(outcomes), "geometry_calls": calls,
            "geometry_call_rate": calls / len(outcomes),
            "success": sum(x["success"] for x in outcomes) / len(outcomes),
            "rows": outcomes}


def aggregate(seq_reports):
    n = sum(x["frames"] for x in seq_reports.values())
    calls = sum(x["geometry_calls"] for x in seq_reports.values())
    good = sum(sum(row["success"] for row in x["rows"]) for x in seq_reports.values())
    return {"frames": n, "geometry_calls": calls,
            "geometry_call_rate": calls / n, "success": good / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--risk-root", type=Path, required=True)
    p.add_argument("--avl003-root", type=Path, required=True)
    p.add_argument("--always-route", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--max-geometry-rate", type=float, default=0.60)
    args = p.parse_args()
    validation = {}
    for threshold in THRESHOLDS:
        per = {seq: evaluate_sequence(args, seq, threshold) for seq in args.validation}
        validation[str(threshold)] = {"sequences": per, "aggregate": aggregate(per)}
    # Frozen selection rule: enforce a pre-registered 60% geometry budget,
    # then maximize validation success and minimize calls as a tie-break.
    budget = args.max_geometry_rate
    eligible = [t for t in THRESHOLDS
                if validation[str(t)]["aggregate"]["geometry_call_rate"] <= budget]
    if not eligible:
        raise RuntimeError(f"no threshold satisfies max geometry rate {budget}")
    selected = max(eligible, key=lambda t: (validation[str(t)]["aggregate"]["success"],
                                            -validation[str(t)]["aggregate"]["geometry_calls"]))
    test_per = {seq: evaluate_sequence(args, seq, selected) for seq in args.test}
    report = {
        "protocol": "validation-selected threshold; test labels read only after freeze",
        "validation_sequences": args.validation,
        "test_sequences": args.test,
        "threshold_grid": list(THRESHOLDS),
        "selected_threshold": selected,
        "max_geometry_rate": budget,
        "validation": validation,
        "test": {"sequences": test_per, "aggregate": aggregate(test_per)},
        "selection_uses_test_gt": False,
        "online_policy_uses_query_gt": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_threshold": selected,
                      "validation": validation[str(selected)]["aggregate"],
                      "test": report["test"]["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
