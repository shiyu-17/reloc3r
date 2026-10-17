#!/usr/bin/env python3
"""Build utility-aligned Stage 3C.6 supervision from terminal-Q labels.

Every legal action, including Accept and Defer, is represented in the same
incremental utility space.  The component targets reconstruct the exact
evaluator utility after removing the action-independent cost already spent in
the live state.  This prevents the online controller from comparing a recovery
probability with an uncalibrated verifier confidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from tools.stage3b_policy import is_deployment_legal


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_utility(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = ("success_reward", "cost_penalty_per_s", "false_accept_penalty",
                "recovery_penalty_per_frame", "memory_pollution_penalty")
    if not isinstance(value, Mapping) or any(key not in value for key in required):
        raise ValueError("utility config is missing required non-negative weights")
    if any(not isinstance(value[key], (int, float)) or float(value[key]) < 0 for key in required):
        raise ValueError("utility weights must be non-negative numbers")
    return dict(value)


def component_targets(action: Mapping[str, Any], observation: Mapping[str, Any],
                      utility: Mapping[str, Any]) -> Dict[str, float]:
    metrics = action.get("best_terminal_metrics") or {}
    state = observation.get("state") or {}
    spent = _float(state.get("frame_tool_cost_s"))
    total_cost = _float(metrics.get("measured_tool_cost_s"), spent)
    incremental_cost = max(0.0, total_cost - spent)
    result = {
        "correct_accept": float(bool(metrics.get("correct_accept"))),
        "false_accept": float(bool(metrics.get("false_accept"))),
        "incremental_cost_s": incremental_cost,
        "recovery_time_frames": max(0.0, _float(metrics.get("recovery_time_frames"))),
        "memory_pollution": float(bool(metrics.get("memory_pollution"))),
    }
    result["component_utility"] = (
        float(utility["success_reward"]) * result["correct_accept"]
        - float(utility["false_accept_penalty"]) * result["false_accept"]
        - float(utility["cost_penalty_per_s"]) * result["incremental_cost_s"]
        - float(utility["recovery_penalty_per_frame"]) * result["recovery_time_frames"]
        - float(utility["memory_pollution_penalty"]) * result["memory_pollution"]
    )
    # Q* includes cost already spent before this decision.  Adding it back
    # produces the same incremental utility used by the component controller.
    result["oracle_incremental_utility"] = (
        _float(action.get("q_star")) + float(utility["cost_penalty_per_s"]) * spent
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    args = parser.parse_args()
    utility = load_utility(args.utility_config)
    records = []
    kinds: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    max_reconstruction_error = 0.0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        observation = row["observation"]
        actions = []
        for source in row.get("action_values", []):
            if not is_deployment_legal(observation, source):
                continue
            action = dict(source)
            if str(action.get("kind")) == "reject":
                action["kind"] = "defer"
            targets = component_targets(action, observation, utility)
            error = abs(targets["component_utility"] - targets["oracle_incremental_utility"])
            max_reconstruction_error = max(max_reconstruction_error, error)
            if error > 1e-6:
                raise ValueError(
                    f"component utility does not reconstruct Q* for {row.get('state_id')}: {error}")
            action.update(targets)
            actions.append(action)
            kinds[str(action["kind"])] += 1
            for name in ("correct_accept", "false_accept", "memory_pollution"):
                outcomes[f"{name}:{int(targets[name])}"] += 1
        if not actions:
            continue
        records.append({
            "state_id": row["state_id"], "sequence": row["sequence"],
            "frame_index": int(row["frame_index"]), "observation": observation,
            "actions": actions, "dagger_diagnostics": row.get("dagger_diagnostics", []),
            "target_tags": row.get("target_tags", []),
        })
    records.sort(key=lambda x: (x["sequence"], x["frame_index"], x["state_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "utility_components.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema_version": 1, "name": "stage3c6_utility_components",
        "source": str(args.input), "states": len(records),
        "actions": sum(len(row["actions"]) for row in records),
        "action_counts": dict(sorted(kinds.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "utility": utility, "maximum_q_reconstruction_error": max_reconstruction_error,
        "online_boundary": "component labels and Oracle utility are evaluator-only; online policy receives only observation/action features",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
