#!/usr/bin/env python3
"""Turn executable Oracle traces into masked state/action/value examples.

This is deliberately an evaluator-side conversion.  The produced labels use
terminal GT utility, while ``observation`` contains only the exact
policy-visible state saved before each real GPU action.  It never calls a
localization backend and cannot be used as an online decision path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from tools.eval_active_evidence_sequential import load_utility


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def action_key(action: Mapping[str, Any]) -> str:
    return canonical({"kind": action["kind"], "references": action.get("references", [])})


def defer_value(state: Mapping[str, Any], prior_cost: float,
                utility: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = state.get("state", {})
    streak_after = int(snapshot.get("reject_streak", 0)) + 1
    q_value = (-float(utility["cost_penalty_per_s"]) * float(prior_cost)
               -float(utility["recovery_penalty_per_frame"]) * streak_after)
    return {
        "kind": "defer", "references": [], "q_star": q_value,
        "best_terminal_type": "defer", "best_leaf_id": None,
        "best_terminal_metrics": {
            "correct_accept": False, "false_accept": False,
            "measured_tool_cost_s": float(prior_cost),
            "recovery_time_frames": streak_after, "memory_pollution": False,
        },
        "is_terminal_action": True,
    }


def coverage_tags(state: Mapping[str, Any], actions: Iterable[Mapping[str, Any]]) -> List[str]:
    """Coarse, auditable strata for coverage reports; labels are not inputs."""
    tags: List[str] = []
    candidate = state.get("candidate") or {}
    snapshot = state.get("state") or {}
    if not candidate:
        tags.append("candidate:none")
    else:
        tags.append("candidate:pose" if candidate.get("pose_available") else "candidate:no_pose")
        if candidate.get("verification"):
            tags.append("verifier:passed" if candidate.get("verified") else "verifier:rejected")
        tags.append("source:" + str(candidate.get("source", "unknown")))
        verification = candidate.get("verification") or {}
        risk = verification.get("total_risk", verification.get("independent_risk"))
        if isinstance(risk, (int, float)):
            tags.append("risk:high" if float(risk) >= 0.8 else "risk:low")
    tags.append("temporal:stale" if snapshot.get("accepted_pose_available") and
                int(snapshot.get("frame_index", 0)) - int(snapshot.get("accepted_frame_index", 0) or 0) > 5
                else "temporal:fresh_or_lost")
    streak = int(snapshot.get("reject_streak", 0))
    tags.append("reject:multiple" if streak >= 2 else "reject:one" if streak else "reject:none")
    memory_available = bool(snapshot.get("memory_ids")) or int(snapshot.get("memory_size", 0)) > 0
    tags.append("memory:available" if memory_available else "memory:empty")
    if any(action.get("kind") == "geometry" for action in actions):
        tags.append("geometry:available")
    for action in actions:
        terminal = action.get("best_terminal_type")
        if terminal:
            tags.append("terminal:" + str(terminal))
    return sorted(set(tags))


def trace_paths(inputs: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*_oracle.jsonl")))
        else:
            paths.append(input_path)
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError("no *_oracle.jsonl trace found")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Oracle trace file(s) or directories containing them")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path,
                        default=Path("configs/stage3a_terminal_utility_v1.json"))
    args = parser.parse_args()
    utility = load_utility(args.utility_config)
    states: Dict[str, Dict[str, Any]] = {}
    source_nodes = 0
    for path in trace_paths(args.input):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for node in row.get("oracle_action_values", []):
                value = node.get("action_value") or {}
                if "q_star" not in value:
                    raise ValueError(f"{path}:{line_number} has old success-only Oracle labels; rerun evaluator")
                # Older traces were generated before frame_tool_cost_s became
                # part of the environment observation.  Reconstruct that
                # policy-visible budget feature from the saved live receipt
                # ledger so they remain valid Stage 3A examples.
                observation = json.loads(canonical(node["state"]))
                snapshot = observation.setdefault("state", {})
                # Preserve the exact live DAgger observation when present.
                # Only legacy traces require reconstruction from prior_cost_s.
                spent = float(snapshot.get("frame_tool_cost_s", node.get("prior_cost_s", 0.0)))
                budget = float(snapshot.get("budget_s", row.get("budget_s", 0.0)))
                snapshot["frame_tool_cost_s"] = spent
                snapshot["budget_s"] = budget
                snapshot["remaining_budget_s"] = float(
                    snapshot.get("remaining_budget_s", max(0.0, budget - spent)))
                key_payload = {"sequence": row["sequence"], "frame_index": row["frame_index"],
                               "observation": observation}
                state_id = hashlib.sha256(canonical(key_payload).encode("utf-8")).hexdigest()[:24]
                entry = states.setdefault(state_id, {
                    "state_id": state_id, "sequence": row["sequence"],
                    "frame_index": int(row["frame_index"]), "observation": observation,
                    "actions": {}, "source_nodes": 0, "dagger_diagnostics": [],
                })
                diagnostic = row.get("dagger_diagnostic")
                if isinstance(diagnostic, Mapping) and diagnostic not in entry["dagger_diagnostics"]:
                    entry["dagger_diagnostics"].append(dict(diagnostic))
                kind = "defer" if node["action"]["kind"] == "reject" else node["action"]["kind"]
                action = {"kind": kind,
                          "references": node["action"].get("references", []),
                          "q_star": float(value["q_star"]),
                          "best_terminal_type": value["best_terminal_type"],
                          "best_leaf_id": value["best_leaf_id"],
                          "best_terminal_metrics": value["best_terminal_metrics"],
                          "is_terminal_action": bool(node["action"].get("is_terminal_action", False))}
                key = action_key(action)
                previous = entry["actions"].get(key)
                if previous is not None and abs(float(previous["q_star"]) - action["q_star"]) > 1e-9:
                    raise ValueError(f"inconsistent Q labels for state/action at {path}:{line_number}")
                entry["actions"][key] = action
                entry["source_nodes"] += 1
                source_nodes += 1
    records: List[Dict[str, Any]] = []
    tag_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for entry in states.values():
        # Only v1 acquisition-only traces need an analytic Defer fallback.
        # Insert it after all executed nodes have been merged so a real v2
        # Defer (including the legacy Reject alias) always takes precedence.
        snapshot = entry["observation"].get("state", {})
        defer = defer_value(entry["observation"],
                            float(snapshot.get("frame_tool_cost_s", 0.0)), utility)
        entry["actions"].setdefault(action_key(defer), defer)
        actions = sorted(entry.pop("actions").values(), key=action_key)
        best_q = max(float(action["q_star"]) for action in actions)
        for action in actions:
            action["is_optimal_action"] = abs(float(action["q_star"]) - best_q) <= 1e-12
            action_counts[action["kind"]] += 1
        entry["optimal_actions"] = [action["kind"] for action in actions if action["is_optimal_action"]]
        target_tags: List[str] = []
        policy_induced = bool(entry["dagger_diagnostics"])
        if policy_induced and "geometry" in entry["optimal_actions"]:
            target_tags.append("geometry_optimal")
        if policy_induced and "memory" in entry["optimal_actions"]:
            target_tags.append("memory_optimal")
        if policy_induced and "accept" in entry["optimal_actions"]:
            target_tags.append("accept_optimal")
        if any(float(item.get("q_gap", 0.0)) >= 0.10 for item in entry["dagger_diagnostics"]):
            target_tags.append("high_oracle_regret")
        if any(bool(item.get("premature_accept")) for item in entry["dagger_diagnostics"]):
            target_tags.append("premature_accept")
        entry["target_tags"] = target_tags
        entry["q_star"] = best_q
        entry["action_values"] = actions
        entry["coverage_tags"] = coverage_tags(entry["observation"], actions)
        tag_counts.update(entry["coverage_tags"])
        records.append(entry)
    records.sort(key=lambda row: (row["sequence"], row["frame_index"], row["state_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "state_action_values.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "purpose": "evaluator-only Oracle state/action/value training labels",
        "utility": utility,
        "input_traces": [str(path) for path in trace_paths(args.input)],
        "unique_policy_states": len(records),
        "executed_state_actions": source_nodes,
        "actions_per_state": (sum(len(row["action_values"]) for row in records) / len(records) if records else 0.0),
        "action_label_counts": dict(sorted(action_counts.items())),
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "target_tag_counts": dict(sorted(Counter(
            tag for row in records for tag in row.get("target_tags", [])).items())),
        "online_information_boundary": "observation is saved policy-visible state; Q labels and terminal outcomes are evaluator-only",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
