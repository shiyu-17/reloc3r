#!/usr/bin/env python3
"""Build grouped recovery/cost/ranking labels from Stage 3B Oracle states.

The source state/action file contains evaluator-only terminal outcomes.  This
converter retains policy-visible observations but replaces the scalar utility
target with two same-horizon labels for each *acquisition* action:

``recovery_target``
    Whether the Oracle-selected terminal continuation after this action reaches
    a correct accepted pose.
``incremental_cost_s``
    Additional measured tool time from the live state to that same terminal.

The resulting advantage label is only an offline ranking target.  At runtime
the policy predicts recovery and cost from pre-execution features and an
explicit controller compares against Defer's zero-new-evidence baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tools.stage3b_policy import is_deployment_legal


ACQUISITION = {"localize", "geometry", "memory"}


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def recovery_state_tags(observation: Mapping[str, Any], actions: List[Dict[str, Any]]) -> List[str]:
    state = observation.get("state") or {}
    candidate = observation.get("candidate") or {}
    tags: List[str] = []
    if candidate:
        tags.append("recovery:has_candidate")
        tags.append("candidate:" + str(candidate.get("source", "unknown")))
        if candidate.get("verified"):
            tags.append("verifier:passed")
        else:
            tags.append("verifier:rejected")
    else:
        tags.append("recovery:root")
    if int(state.get("memory_size", 0) or 0) > 0:
        tags.append("memory:available")
    if bool(state.get("accepted_pose_available")) and int(state.get("anchor_age", 0) or 0) > 5:
        tags.append("anchor:stale")
    if int(state.get("reject_streak", 0) or 0) >= 2:
        tags.append("reject:multiple")
    if any(item["kind"] == "geometry" for item in actions):
        tags.append("geometry:available")
    return sorted(set(tags))


def state_weight(tags: List[str], boundary_margin: float, *, boundary_threshold: float) -> float:
    # Target the difficult recovery strata without making every post-action
    # state dominant. The unweighted natural distribution remains in the file.
    weight = 1.0
    if "candidate:memory" in tags or "candidate:reloc3r_online" in tags:
        weight = max(weight, 1.5)
    if "anchor:stale" in tags or "reject:multiple" in tags:
        weight = max(weight, 1.5)
    if boundary_margin <= boundary_threshold:
        weight = max(weight, 2.0)
    return weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Stage 3B state_action_values.jsonl with live-cost Oracle labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, default=0.05,
                        help="frozen offline utility coefficient in seconds^-1")
    parser.add_argument("--boundary-margin", type=float, default=0.02,
                        help="target states whose best two acquisition advantages are close")
    args = parser.parse_args()
    if args.lambda_cost < 0.0 or args.boundary_margin < 0.0:
        parser.error("cost and margin parameters must be non-negative")

    source = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: List[Dict[str, Any]] = []
    tag_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    recovery_counts: Counter[str] = Counter()
    for row in source:
        observation = row["observation"]
        spent = float_or((observation.get("state") or {}).get("frame_tool_cost_s"))
        actions: List[Dict[str, Any]] = []
        for value in row.get("action_values", []):
            kind = "defer" if str(value.get("kind")) == "reject" else str(value.get("kind"))
            if kind not in ACQUISITION or not is_deployment_legal(observation, value):
                continue
            metrics = value.get("best_terminal_metrics") or {}
            terminal_cost = float_or(metrics.get("measured_tool_cost_s"), spent)
            incremental_cost = max(0.0, terminal_cost - spent)
            recovery = float(bool(metrics.get("correct_accept", False)))
            action = {
                "kind": kind, "references": list(value.get("references", [])),
                "is_terminal_action": False,
                "recovery_target": recovery,
                "incremental_cost_s": incremental_cost,
                "oracle_advantage": recovery - args.lambda_cost * incremental_cost,
                # Kept only for evaluator-side diagnosis; never passed to the model online.
                "oracle_q_star": float_or(value.get("q_star")),
                "best_terminal_type": value.get("best_terminal_type"),
            }
            actions.append(action)
            action_counts[kind] += 1
            recovery_counts[kind + (":recover" if recovery else ":no_recover")] += 1
        if not actions:
            continue
        ordered = sorted((item["oracle_advantage"] for item in actions), reverse=True)
        boundary = float("inf") if len(ordered) < 2 else float(ordered[0] - ordered[1])
        tags = recovery_state_tags(observation, actions)
        weight = state_weight(tags, boundary, boundary_threshold=args.boundary_margin)
        tag_counts.update(tags)
        records.append({
            "state_id": row["state_id"], "sequence": row["sequence"], "frame_index": int(row["frame_index"]),
            "observation": observation, "actions": actions,
            "dagger_diagnostics": row.get("dagger_diagnostics", []),
            "recovery_tags": tags, "boundary_margin": boundary,
            "training_weight": weight,
        })
    records.sort(key=lambda item: (item["sequence"], item["frame_index"], item["state_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "recovery_cost_ranking.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "purpose": "Stage 3C grouped recovery/cost/pairwise ranking supervision",
        "source": str(args.input), "states": len(records),
        "actions": int(sum(len(item["actions"]) for item in records)),
        "lambda_cost": args.lambda_cost, "boundary_margin": args.boundary_margin,
        "action_counts": dict(sorted(action_counts.items())),
        "recovery_counts": dict(sorted(recovery_counts.items())),
        "recovery_tag_counts": dict(sorted(tag_counts.items())),
        "targeted_state_count": sum(item["training_weight"] > 1.0 for item in records),
        "online_boundary": "observations and action descriptors are policy-visible; recovery/cost/advantage labels are evaluator-only",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
