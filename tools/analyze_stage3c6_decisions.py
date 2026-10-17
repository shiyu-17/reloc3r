#!/usr/bin/env python3
"""Decompose closed-loop errors into terminal admission and acquisition rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


ACQUISITION = {"localize", "geometry", "memory"}
TERMINAL = {"accept", "defer", "reject"}


def key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind")), tuple(int(x) for x in action.get("references", ()))


def divide(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


def audit(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    admission_opportunities = admitted = acquisition_calls = worthwhile_acquisition_calls = 0
    geometry_opportunities = geometry_hits = geometry_calls = optimal_geometry_calls = 0
    conditional_geometry_opportunities = conditional_geometry_hits = 0
    regrets = []; terminal_regrets = []; acquisition_rank_regrets = []
    for decision in decisions:
        nodes = (decision.get("oracle_label") or {}).get("oracle_action_values") or []
        if not nodes:
            continue
        q = {key(node["action"]): float(node["action_value"]["q_star"]) for node in nodes}
        selected = key(decision["action"])
        if selected not in q:
            raise ValueError(f"selected action missing from Oracle labels: {selected}")
        acquisition = {k: value for k, value in q.items() if k[0] in ACQUISITION}
        terminal = {k: value for k, value in q.items() if k[0] in TERMINAL}
        best_q = max(q.values()); selected_q = q[selected]
        regrets.append(best_q - selected_q)
        best_acq = max(acquisition.values()) if acquisition else float("-inf")
        best_terminal = max(terminal.values()) if terminal else float("-inf")
        should_acquire = best_acq > best_terminal + 1e-12
        selected_acquisition = selected[0] in ACQUISITION
        if should_acquire:
            admission_opportunities += 1
            admitted += int(selected_acquisition)
            if not selected_acquisition:
                terminal_regrets.append(best_acq - selected_q)
        if selected_acquisition:
            acquisition_calls += 1
            worthwhile_acquisition_calls += int(should_acquire)
            acquisition_rank_regrets.append(best_acq - selected_q)
        geometry = {k: value for k, value in acquisition.items() if k[0] == "geometry"}
        if geometry:
            best_geometry = max(geometry.values())
            geometry_is_global = best_geometry >= best_q - 1e-12
            geometry_is_best_acquisition = best_geometry >= best_acq - 1e-12
            if geometry_is_global:
                geometry_opportunities += 1
                geometry_hits += int(selected[0] == "geometry")
            if should_acquire and geometry_is_best_acquisition:
                conditional_geometry_opportunities += 1
                conditional_geometry_hits += int(selected[0] == "geometry")
        if selected[0] == "geometry":
            geometry_calls += 1
            optimal_geometry_calls += int(q[selected] >= best_q - 1e-12)
    def stats(values):
        return {"count": len(values), "mean": None if not values else float(np.mean(values)),
                "p90": None if not values else float(np.quantile(values, 0.9))}
    return {
        "decisions": len(decisions),
        "acquisition_admission": {
            "oracle_should_acquire": admission_opportunities, "policy_acquired": admitted,
            "recall": divide(admitted, admission_opportunities),
            "policy_acquisition_calls": acquisition_calls,
            "precision": divide(worthwhile_acquisition_calls, acquisition_calls),
            "terminal_preemption_regret": stats(terminal_regrets),
        },
        "conditional_geometry_ranking": {
            "oracle_geometry_best_given_acquisition": conditional_geometry_opportunities,
            "policy_geometry_hits": conditional_geometry_hits,
            "recall": divide(conditional_geometry_hits, conditional_geometry_opportunities),
            "acquisition_rank_regret": stats(acquisition_rank_regrets),
        },
        "end_to_end_geometry": {
            "oracle_geometry_global_optimal": geometry_opportunities,
            "policy_geometry_hits": geometry_hits, "recall": divide(geometry_hits, geometry_opportunities),
            "policy_geometry_calls": geometry_calls, "precision": divide(optimal_geometry_calls, geometry_calls),
        },
        "overall_oracle_regret": stats(regrets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = []
    paths = sorted(set(args.trace_root.glob("traces/*_learned.jsonl")) |
                   set(args.trace_root.glob("*/traces/*_learned.jsonl")))
    if not paths:
        raise FileNotFoundError("no learned traces found")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for decision in row.get("decisions", []):
                if decision.get("oracle_label") is not None:
                    decisions.append(decision)
    report = {"schema_version": 1, "trace_root": str(args.trace_root), **audit(decisions),
              "online_boundary": "Oracle labels are joined after policy selection only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
