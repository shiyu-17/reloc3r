#!/usr/bin/env python3
"""Aggregate per-sequence outputs of the online active-evidence evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


POLICIES = (
    "fixed", "fixed_geometry", "geometry_greedy", "uncertainty_trigger",
    "one_shot_router", "random", "feedback_rule",
)


def weighted(rows: List[Dict[str, Any]], oracle_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = sum(int(row["frames"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    correct = sum(int(row["accepted_correct"]) for row in rows)
    geometry = sum(int(row["geometry_calls"]) for row in rows)
    oracle_correct = sum(int(row["accepted_correct"]) for row in oracle_rows)
    def mean(name: str) -> float:
        return sum(float(row[name]) * int(row["frames"]) for row in rows) / frames
    return {
        "frames": frames,
        "accepted": accepted,
        "accepted_correct": correct,
        "coverage": accepted / frames,
        "accepted_accuracy": correct / accepted if accepted else 0.0,
        "false_accepts": accepted - correct,
        "false_accept_rate": (accepted - correct) / accepted if accepted else 0.0,
        "full_frame_success": correct / frames,
        "geometry_calls": geometry,
        "geometry_call_rate": geometry / frames,
        "mean_measured_tool_cost_s": mean("mean_measured_tool_cost_s"),
        "oracle_full_frame_success": (oracle_correct / frames if oracle_rows else None),
        "oracle_gap_full_frame_success": ((oracle_correct - correct) / frames if oracle_rows else None),
    }


def aggregate_oracle(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "online_policy_collection_used_skip_oracle"}
    frames = sum(int(row["frames"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    correct = sum(int(row["accepted_correct"]) for row in rows)
    geometry = sum(int(row["geometry_calls"]) for row in rows)
    def mean(name: str) -> float:
        return sum(float(row[name]) * int(row["frames"]) for row in rows) / frames
    return {
        "frames": frames,
        "accepted": accepted,
        "accepted_correct": correct,
        "coverage": accepted / frames,
        "full_frame_success": correct / frames,
        "geometry_calls": geometry,
        "geometry_call_rate": geometry / frames,
        "mean_selected_program_cost_s": mean("mean_measured_tool_cost_s"),
        "mean_search_wall_time_s": mean("mean_wall_time_s"),
        "oracle_uses_gt": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = sorted(args.root.glob("*/summary.json"))
    direct_summary = args.root / "summary.json"
    if direct_summary.exists():
        summaries = [direct_summary]
    if not summaries:
        raise FileNotFoundError("no per-sequence summary.json beneath " + str(args.root))
    per_sequence: Dict[str, Dict[str, Any]] = {}
    policy_rows = {policy: [] for policy in POLICIES}
    oracle_rows: List[Dict[str, Any]] = []
    config = None
    for path in summaries:
        data = json.loads(path.read_text())
        config = config or data["config"]
        sequences = (data["config"]["sequences"] if path == direct_summary
                     else [data["config"]["sequences"][0]])
        for sequence in sequences:
            oracle_entry = data.get("oracle", {}).get(sequence)
            per_sequence[sequence] = {"policies": {}, "oracle": oracle_entry}
            if oracle_entry is not None:
                oracle_rows.append(oracle_entry["summary"])
            for policy in data["policies"]:
                summary = data["policies"][policy][sequence]["summary"]
                per_sequence[sequence]["policies"][policy] = summary
                policy_rows.setdefault(policy, []).append(summary)
    output = {
        "config": config,
        "sequences": sorted(per_sequence),
        "per_sequence": per_sequence,
        "aggregate": {
            **{policy: weighted(rows, oracle_rows)
               for policy, rows in policy_rows.items() if rows},
            "oracle": aggregate_oracle(oracle_rows),
        },
        "interpretation": {
            "online_policies": "all policy actions use live Reloc3r/RoMa receipts; no query GT in selection",
            "oracle": "budget-matched finite-depth executable, verifier-gated evaluator upper bound; it uses GT only to select a terminal branch",
            "cost": "mean sum of backend receipt elapsed_s; model initialization/search overhead excluded",
        },
    }
    destination = args.output or args.root / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
