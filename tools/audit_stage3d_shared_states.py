#!/usr/bin/env python3
"""Audit frozen F0/F2 controllers on the union of policy-induced states.

The audit never changes a rollout.  It deduplicates semantic online states
from F0, F2, and feedback-rule collector traces, joins their post-hoc Oracle
action values, and replays the two frozen controllers on every shared state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from tools.stage3b_policy import action_features, observation_dict
from tools.stage3c6_utility_policy import (TERMINAL_EVIDENCE_SOURCES,
                                            UtilityComponentPolicy,
                                            compose_utility, policy_state_features)


ACQUISITION = {"localize", "geometry", "memory"}
CHEAP = {"localize", "memory"}
RUNTIME_KEYS = {"elapsed_s", "wall_time_s", "causal", "candidate_pose"}


def action_key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind")), tuple(int(value) for value in action.get("references", []))


def stable_state(value: Any) -> Any:
    """Remove run-time-only receipt fields before cross-policy deduplication."""
    if isinstance(value, Mapping):
        return {str(key): stable_state(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [stable_state(item) for item in value]
    return value


def signature(observation: Mapping[str, Any], actions: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "observation": stable_state(observation),
        "actions": sorted((kind, list(refs)) for kind, refs in map(action_key, actions)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect(root: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    paths = sorted(root.glob("*/*_rollout.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rollout files under {root}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            rollout = json.loads(line)
            for decision in rollout.get("decisions", []):
                label = decision.get("oracle_label") or {}
                nodes = label.get("oracle_action_values") or []
                if not nodes:
                    continue
                values = {}
                actions = []
                for node in nodes:
                    action = dict(node["action"])
                    key = action_key(action)
                    value = float(node["action_value"]["q_star"])
                    values[key] = max(values.get(key, float("-inf")), value)
                    actions.append(action)
                state = decision["state"]
                rows.append({
                    "source": source,
                    "sequence": str(rollout["sequence"]),
                    "frame_index": int(rollout["frame_index"]),
                    "decision_index": int(decision["decision_index"]),
                    "observation": state,
                    "actions": actions,
                    "oracle_values": values,
                    "selected": action_key(decision["action"]),
                })
    return rows


def score_rows(policy: UtilityComponentPolicy, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score every state/action in one XGBoost batch per component.

    Per-state ``predict`` calls are unexpectedly expensive for small matrices;
    batching is mathematically identical and keeps this audit practical.
    """
    eligible = []
    features, spans = [], []
    for row in rows:
        actions = list(row["actions"])
        kinds = [str(action["kind"]) for action in actions]
        if "geometry" not in kinds or not any(kind in CHEAP for kind in kinds):
            continue
        state = policy_state_features(row["observation"], policy.feature_level)
        begin = len(features)
        features.extend(np.concatenate((state, action_features(row["observation"], action))) for action in actions)
        spans.append((begin, len(features)))
        eligible.append((row, actions, kinds))
    if not eligible:
        return []
    x = np.asarray(features, dtype=np.float32)
    components = {
        "correct_accept": policy.models["correct_accept"].predict_proba(x)[:, 1],
        "false_accept": policy.models["false_accept"].predict_proba(x)[:, 1],
        "memory_pollution": policy.models["memory_pollution"].predict_proba(x)[:, 1],
        "incremental_cost_s": np.expm1(policy.models["incremental_cost_s"].predict(x)).clip(0.0, None),
        "recovery_time_frames": np.expm1(policy.models["recovery_time_frames"].predict(x)).clip(0.0, None),
    }
    expected = compose_utility(components, policy.utility)
    output = []
    for (row, actions, kinds), (begin, end) in zip(eligible, spans):
        scores = expected[begin:end].astype(float, copy=True)
        geometry = [index for index, kind in enumerate(kinds) if kind == "geometry"]
        cheap = [index for index, kind in enumerate(kinds) if kind in CHEAP]
        geometry_index = max(geometry, key=lambda index: scores[index])
        cheap_index = max(cheap, key=lambda index: scores[index])
        obs = observation_dict(row["observation"])
        candidate = obs.get("candidate") or {}
        terminal_eligible = (bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
                             and str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES)
        for index, kind in enumerate(kinds):
            if kind == "accept" and not terminal_eligible:
                scores[index] = -np.inf
        choice = int(np.argmax(scores))
        oracle = row["oracle_values"]
        oracle_geometry = max(value for key, value in oracle.items() if key[0] == "geometry")
        oracle_cheap = max(value for key, value in oracle.items() if key[0] in CHEAP)
        chosen_key = action_key(actions[choice])
        output.append({
            "predicted_margin": float(expected[begin + geometry_index] - expected[begin + cheap_index]),
            "true_margin": float(oracle_geometry - oracle_cheap),
            "predict_geometry_better": bool(expected[begin + geometry_index] >= expected[begin + cheap_index]),
            "oracle_geometry_better": bool(oracle_geometry >= oracle_cheap),
            "chosen_kind": chosen_key[0], "chosen_geometry": chosen_key[0] == "geometry",
            "chosen_oracle_regret": float(max(oracle.values()) - oracle[chosen_key]),
            "model_geometry_key": action_key(actions[geometry_index]),
            "model_cheap_key": action_key(actions[cheap_index]),
        })
    return output


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    truth = np.asarray([row["oracle_geometry_better"] for row in rows], dtype=bool)
    prediction = np.asarray([row["predict_geometry_better"] for row in rows], dtype=bool)
    selected = np.asarray([row["chosen_geometry"] for row in rows], dtype=bool)
    true_margin = np.asarray([row["true_margin"] for row in rows], dtype=float)
    predicted_margin = np.asarray([row["predicted_margin"] for row in rows], dtype=float)
    positives = int(truth.sum())
    calls = int(selected.sum())
    return {
        "states": len(rows),
        "oracle_geometry_advantage_states": positives,
        "margin_sign_accuracy": float(np.mean(prediction == truth)),
        "advantage_mae": float(np.mean(np.abs(predicted_margin - true_margin))),
        "advantage_correlation": (None if len(rows) < 2 or np.std(predicted_margin) == 0 or np.std(true_margin) == 0
                                  else float(np.corrcoef(predicted_margin, true_margin)[0, 1])),
        "geometry_selection_recall": None if positives == 0 else float(np.sum(selected & truth) / positives),
        "geometry_selection_precision": None if calls == 0 else float(np.sum(selected & truth) / calls),
        "geometry_calls": calls,
        "mean_selected_oracle_regret": float(np.mean([row["chosen_oracle_regret"] for row in rows])),
        "confusion": {"tp": int(np.sum(prediction & truth)), "fp": int(np.sum(prediction & ~truth)),
                      "tn": int(np.sum(~prediction & ~truth)), "fn": int(np.sum(~prediction & truth))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-root", type=Path, required=True)
    parser.add_argument("--f2-root", type=Path, required=True)
    parser.add_argument("--feedback-root", type=Path, required=True)
    parser.add_argument("--f0-checkpoint", type=Path, required=True)
    parser.add_argument("--f2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = []
    for root, name in ((args.f0_root, "f0"), (args.f2_root, "f2"),
                       (args.feedback_root, "feedback_rule")):
        sources.extend(collect(root, name))
    merged: dict[str, dict[str, Any]] = {}
    for row in sources:
        key = signature(row["observation"], row["actions"])
        if key not in merged:
            merged[key] = {**row, "sources": [row["source"]], "source_actions": {row["source"]: row["selected"]}}
        else:
            merged[key]["sources"].append(row["source"])
            merged[key]["source_actions"][row["source"]] = row["selected"]
    shared = list(merged.values())
    f0, f2 = UtilityComponentPolicy.load(args.f0_checkpoint), UtilityComponentPolicy.load(args.f2_checkpoint)
    eligible_count = sum(1 for row in shared if any(action_key(action)[0] == "geometry" for action in row["actions"])
                         and any(action_key(action)[0] in CHEAP for action in row["actions"]))
    replay = {"f0": score_rows(f0, shared), "f2": score_rows(f2, shared)}
    skipped = len(shared) - eligible_count
    divergences = []
    for f0_result, f2_result in zip(replay["f0"], replay["f2"]):
        if f0_result["chosen_kind"] != f2_result["chosen_kind"]:
            divergences.append({"f0": f0_result["chosen_kind"], "f2": f2_result["chosen_kind"],
                                "f0_regret": f0_result["chosen_oracle_regret"],
                                "f2_regret": f2_result["chosen_oracle_regret"]})
    source_counts = Counter(source for row in sources for source in [row["source"]])
    state_source_counts = Counter(source for row in shared for source in set(row["sources"]))
    membership = Counter("+".join(sorted(set(row["sources"]))) for row in shared)
    report = {
        "schema_version": 1,
        "purpose": "frozen F0/F2 shared-state Geometry advantage audit",
        "input_decisions": dict(source_counts),
        "unique_semantic_states": len(shared),
        "states_by_source_membership": dict(state_source_counts),
        "exact_source_overlap": dict(membership),
        "eligible_geometry_vs_cheap_states": len(replay["f0"]),
        "skipped_without_geometry_or_cheap_action": skipped,
        "f0": metrics(replay["f0"]), "f2": metrics(replay["f2"]),
        "f0_f2_action_divergence": {
            "count": len(divergences),
            "fraction": len(divergences) / max(1, len(replay["f0"])),
            "mean_f0_regret": None if not divergences else float(np.mean([row["f0_regret"] for row in divergences])),
            "mean_f2_regret": None if not divergences else float(np.mean([row["f2_regret"] for row in divergences])),
            "pairs": dict(Counter(f"{row['f0']}->{row['f2']}" for row in divergences)),
        },
        "online_boundary": "all Oracle values are joined after recorded policy actions; replay uses frozen online observations only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
