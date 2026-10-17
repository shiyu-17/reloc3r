#!/usr/bin/env python3
"""Attribute finite-depth action value on policy-induced v2 traces.

This keeps acquisition decisions (``localize``/``geometry``) separate from
terminal ``accept``/``defer`` decisions.  It is deliberately an audit report:
the finite-depth values are evaluator-only and are not online policy inputs.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ACQUISITION_KINDS = {"localize", "geometry"}
TERMINAL_KINDS = {"accept", "defer"}


def action_key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind")), tuple(int(x) for x in action.get("references", ()))


def roots(label: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [node for node in label.get("oracle_action_values", ())
            if not node.get("parent_path_node_ids")]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def empty_summary() -> dict[str, Any]:
    return {"labelled_decisions": 0, "missing_selected_root": 0,
            "selected_oracle_optimal": 0, "mean_selected_regret": 0.0,
            "action_dependent_states": 0, "action_dependent_fraction": 0.0,
            "acquisition": {"states": 0, "success_dependent": 0,
                            "success_dependent_fraction": 0.0,
                            "selected_kind": {}},
            "terminal": {"states": 0, "accept_defer_success_dependent": 0,
                         "accept_defer_success_dependent_fraction": 0.0,
                         "selected_kind": {}},
            "oracle_best_kind": {}, "selected_to_oracle_kind": {},
            "selected_regret_sum": 0.0}


def consume(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = empty_summary()
    selected_kind = {"acquisition": Counter(), "terminal": Counter()}
    oracle_best_kind, selected_to_best = Counter(), Counter()
    regret_sum = 0.0
    for row in rows:
        decisions = {int(item["decision_index"]): item
                     for item in row.get("online", {}).get("decisions", ())}
        labels = row.get("evaluator", {}).get("oracle_decisions", ())
        for item in labels:
            decision = decisions.get(int(item.get("decision_index", -1)))
            label = item.get("oracle_label", {})
            root_nodes = roots(label)
            if decision is None or not root_nodes:
                continue
            summary["labelled_decisions"] += 1
            chosen = action_key(decision["action"])
            root_by_action = {action_key(node.get("action", {})): node for node in root_nodes}
            selected = root_by_action.get(chosen)
            if selected is None:
                summary["missing_selected_root"] += 1
                continue
            values = [float(node["action_value"]["q_star"]) for node in root_nodes]
            best_q = max(values)
            selected_q = float(selected["action_value"]["q_star"])
            regret = best_q - selected_q
            regret_sum += regret
            is_optimal = abs(regret) <= 1e-8
            summary["selected_oracle_optimal"] += int(is_optimal)
            best_nodes = [node for node in root_nodes
                          if abs(float(node["action_value"]["q_star"]) - best_q) <= 1e-8]
            best_kind = str(best_nodes[0]["action"]["kind"])
            kind = chosen[0]
            oracle_best_kind[best_kind] += 1
            selected_to_best[f"{kind}->{best_kind}"] += 1
            outcomes = {bool(node["action_value"].get("best_terminal_success", 0))
                        for node in root_nodes}
            summary["action_dependent_states"] += int(len(outcomes) > 1)
            if kind in ACQUISITION_KINDS:
                summary["acquisition"]["states"] += 1
                selected_kind["acquisition"][kind] += 1
                acquisition_outcomes = {bool(node["action_value"].get("best_terminal_success", 0))
                                        for node in root_nodes
                                        if str(node.get("action", {}).get("kind")) in ACQUISITION_KINDS}
                summary["acquisition"]["success_dependent"] += int(len(acquisition_outcomes) > 1)
            elif kind in TERMINAL_KINDS:
                summary["terminal"]["states"] += 1
                selected_kind["terminal"][kind] += 1
                terminal_outcomes = {bool(node["action_value"].get("best_terminal_success", 0))
                                     for node in root_nodes
                                     if str(node.get("action", {}).get("kind")) in TERMINAL_KINDS}
                summary["terminal"]["accept_defer_success_dependent"] += int(len(terminal_outcomes) > 1)
    count = summary["labelled_decisions"] - summary["missing_selected_root"]
    summary["mean_selected_regret"] = regret_sum / count if count else 0.0
    summary["selected_oracle_optimal_fraction"] = ratio(summary["selected_oracle_optimal"], count)
    summary["action_dependent_fraction"] = ratio(summary["action_dependent_states"], count)
    summary["acquisition"]["success_dependent_fraction"] = ratio(
        summary["acquisition"]["success_dependent"], summary["acquisition"]["states"])
    summary["terminal"]["accept_defer_success_dependent_fraction"] = ratio(
        summary["terminal"]["accept_defer_success_dependent"], summary["terminal"]["states"])
    summary["acquisition"]["selected_kind"] = dict(selected_kind["acquisition"])
    summary["terminal"]["selected_kind"] = dict(selected_kind["terminal"])
    summary["oracle_best_kind"] = dict(oracle_best_kind)
    summary["selected_to_oracle_kind"] = dict(selected_to_best)
    summary.pop("selected_regret_sum")
    return summary


def read_rows(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                    if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True,
                        help="NAME=glob for policy-induced v2 traces")
    parser.add_argument("--episode-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_policy = {}
    for spec in args.trace:
        if "=" not in spec:
            parser.error("--trace requires NAME=glob")
        name, pattern = spec.split("=", 1)
        by_policy[name] = consume(read_rows(pattern))
    episode = json.loads(args.episode_report.read_text(encoding="utf-8"))
    output = {
        "schema_version": 1,
        "scope": "LILocBench single-site audit; evaluator-only depth-2 values on policy-induced states",
        "episode_value": episode,
        "policy_induced_action_attribution": by_policy,
        "interpretation": {
            "acquisition": "Only compares localize/geometry successor programs; it does not count accept/defer effects.",
            "terminal": "Only compares accept/defer alternatives when both are legal at the recorded state.",
            "training_boundary": "Values are labels, never online observations; scene_disjoint_ready remains false.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_policy, indent=2))


if __name__ == "__main__":
    main()
