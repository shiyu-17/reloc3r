#!/usr/bin/env python3
"""Read-only Q-calibration audit for the Stage 3B closed-loop policy.

The audit deliberately uses only states reached by the deployed policy and
their evaluator-only Oracle labels.  It reports both absolute Q calibration
and *decision margins*: a small absolute error can still reverse Accept versus
Defer or Geometry versus cheap evidence.  No labels are fed back into the
checkpoint and this tool never invokes a visual-localization backend.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from tools.stage3b_policy import (action_features, is_deployment_legal,
                                  state_features, StructuredPolicy)


EPS = 1e-12


def key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    kind = "defer" if str(action.get("kind")) == "reject" else str(action.get("kind"))
    return kind, tuple(int(item) for item in action.get("references", ()))


def finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if np.isfinite(value)]


def numeric(values: Iterable[float]) -> Dict[str, Any]:
    values = np.asarray(finite(values), dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
    }


def calibration(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    predicted = np.asarray([item["predicted"] for item in rows], dtype=np.float64)
    oracle = np.asarray([item["oracle"] for item in rows], dtype=np.float64)
    error = predicted - oracle
    result = {
        "count": int(len(rows)), "predicted_q": numeric(predicted), "oracle_q": numeric(oracle),
        "bias_predicted_minus_oracle": float(error.mean()),
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
    }
    if len(rows) >= 2 and predicted.std() > EPS and oracle.std() > EPS:
        result["pearson_r"] = float(np.corrcoef(predicted, oracle)[0, 1])
        # Least-squares calibration slope/intercept: oracle ~= slope * prediction + intercept.
        result["oracle_on_prediction_slope"] = float(np.polyfit(predicted, oracle, 1)[0])
        result["oracle_on_prediction_intercept"] = float(np.polyfit(predicted, oracle, 1)[1])
    return result


def margin_summary(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    """Calibration for scalar action-value margins (preferred - alternative)."""
    base = calibration(rows)
    if not rows:
        return base
    signs = [
        (float(item["predicted"]) >= 0.0) == (float(item["oracle"]) >= 0.0)
        for item in rows if abs(float(item["oracle"])) > EPS
    ]
    base["oracle_positive_fraction"] = float(np.mean([item["oracle"] > 0.0 for item in rows]))
    base["predicted_positive_fraction"] = float(np.mean([item["predicted"] > 0.0 for item in rows]))
    base["margin_sign_accuracy_non_ties"] = float(np.mean(signs)) if signs else None
    return base


def cost_of(action: Mapping[str, Any]) -> float | None:
    metrics = action.get("best_terminal_metrics") or {}
    value = metrics.get("measured_tool_cost_s")
    try:
        return float(value) if value is not None and np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def strata(observation: Mapping[str, Any]) -> List[str]:
    snapshot = observation.get("state") or {}
    candidate = observation.get("candidate") or {}
    sequence = str(observation.get("sequence", "unknown"))
    if sequence.startswith("dynamics"):
        regime = "regime:dynamic"
    elif sequence.startswith("lt_changes"):
        regime = "regime:long_term_change"
    elif sequence.startswith("static"):
        regime = "regime:static"
    else:
        regime = "regime:other"
    anchor_age = int(snapshot.get("anchor_age", 0) or 0)
    return [
        regime,
        "anchor:stale" if bool(snapshot.get("accepted_pose_available")) and anchor_age > 5 else "anchor:fresh_or_lost",
        "memory:available" if int(snapshot.get("memory_size", 0) or 0) > 0 else "memory:empty",
        "verifier:passed" if bool(candidate.get("pose_available")) and bool(candidate.get("verified")) else "verifier:not_passed",
        "candidate:" + (str(candidate.get("source")) if candidate else "none"),
    ]


def load_dagger(path: Path) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [row for row in records if row.get("dagger_diagnostics")]
    if not records:
        raise ValueError("dataset has no policy-induced DAgger states")
    return records


def predict(policy: StructuredPolicy, observation: Mapping[str, Any],
            actions: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    state = (state_features(observation) - policy.state_mean) / policy.state_std
    action_array = np.stack([(action_features(observation, action) - policy.action_mean) / policy.action_std
                             for action in actions])
    with torch.no_grad():
        logits, values = policy.model(
            torch.from_numpy(state)[None].to(policy.device),
            torch.from_numpy(action_array)[None].to(policy.device),
            torch.ones((1, len(actions)), dtype=torch.bool, device=policy.device))
    return (logits[0].detach().cpu().numpy().astype(np.float64),
            values[0].detach().cpu().numpy().astype(np.float64) * policy.value_std + policy.value_mean)


def compact(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown(summary: Mapping[str, Any]) -> str:
    terminal = summary["terminal_accept"]
    acquisition = summary["acquisition"]
    disagreement = summary["policy_q_disagreement"]
    lines = [
        "# Stage 3B Q-calibration audit",
        "",
        "This is a read-only audit on policy-induced DAgger states. Q labels are evaluator-only Oracle utilities.",
        "",
        "## Terminal Accept",
        "",
        f"- Verifier-passed states: {terminal['states']}; Oracle-Accept-optimal: {terminal['oracle_accept_optimal_states']}; policy under-accepts: {terminal['policy_underaccept_states']}.",
        f"- `Qhat(Accept) - Q*(Accept)` bias: {compact(terminal['accept_q_calibration'].get('bias_predicted_minus_oracle'))}; MAE: {compact(terminal['accept_q_calibration'].get('mae'))}.",
        f"- Accept-vs-best-nonaccept margin sign accuracy: {compact(terminal['accept_vs_best_other_margin'].get('margin_sign_accuracy_non_ties'))}; predicted-positive: {compact(terminal['accept_vs_best_other_margin'].get('predicted_positive_fraction'))}; Oracle-positive: {compact(terminal['accept_vs_best_other_margin'].get('oracle_positive_fraction'))}.",
        "",
        "## Acquisition value",
        "",
        f"- Geometry candidates: {acquisition['geometry_q_calibration']['count']}; Geometry bias: {compact(acquisition['geometry_q_calibration'].get('bias_predicted_minus_oracle'))}.",
        f"- Geometry-vs-cheapest-evidence margin sign accuracy: {compact(acquisition['geometry_vs_cheap_margin'].get('margin_sign_accuracy_non_ties'))}; predicted-positive: {compact(acquisition['geometry_vs_cheap_margin'].get('predicted_positive_fraction'))}; Oracle-positive: {compact(acquisition['geometry_vs_cheap_margin'].get('oracle_positive_fraction'))}.",
        f"- Policy/Q-head disagreement: {disagreement['disagreement_states']}/{disagreement['states']} ({compact(disagreement.get('disagreement_fraction'))}); mean Oracle regret policy/Q: {compact(disagreement['policy_oracle_regret'].get('mean'))}/{compact(disagreement['q_head_oracle_regret'].get('mean'))}.",
        "",
        "## Interpretation guardrail",
        "",
        "Absolute Q bias alone does not establish a control error. The decision-relevant evidence is the corresponding margin bias and whether it reverses the Oracle ordering.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="state_action_values.jsonl that includes policy-induced DAgger states")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy = StructuredPolicy.load(args.checkpoint, device=args.device)
    records = load_dagger(args.dataset)
    by_action: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_stratum_action: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    terminal_accept: List[Dict[str, Any]] = []
    geometry_margins: List[Dict[str, float]] = []
    geometry_margins_by_stratum: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    disagreements: List[Dict[str, Any]] = []
    policy_regrets, q_regrets, policy_cost_deltas, q_cost_deltas = [], [], [], []
    pair_counter: Counter[str] = Counter()
    pair_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    disagreement_by_stratum: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for row in records:
        observation = row["observation"]
        actions = [action for action in row["action_values"] if is_deployment_legal(observation, action)]
        if not actions:
            continue
        logits, predicted = predict(policy, observation, actions)
        oracle = np.asarray([float(action["q_star"]) for action in actions], dtype=np.float64)
        policy_index, q_index, oracle_index = int(np.argmax(logits)), int(np.argmax(predicted)), int(np.argmax(oracle))
        state_strata = strata(observation)
        for action, q_hat, q_star in zip(actions, predicted, oracle):
            kind = key(action)[0]
            measurement = {"predicted": float(q_hat), "oracle": float(q_star)}
            by_action[kind].append(measurement)
            for name in state_strata:
                by_stratum_action[name][kind].append(measurement)

        # Terminal control: compare Accept to the best legal alternative rather than to an
        # arbitrary Defer candidate. This directly measures the tendency to stop too late.
        accept_indices = [index for index, action in enumerate(actions) if key(action)[0] == "accept"]
        if accept_indices and bool((observation.get("candidate") or {}).get("verified")):
            accept_index = accept_indices[0]
            other = [index for index in range(len(actions)) if index != accept_index]
            if other:
                other_index = max(other, key=lambda index: oracle[index])
                terminal_accept.append({
                    "predicted": float(predicted[accept_index]), "oracle": float(oracle[accept_index]),
                    "margin_predicted": float(predicted[accept_index] - max(predicted[index] for index in other)),
                    "margin_oracle": float(oracle[accept_index] - oracle[other_index]),
                    "oracle_accept_optimal": bool(oracle[accept_index] >= oracle.max() - EPS),
                    "policy_accept": policy_index == accept_index,
                    "q_accept": q_index == accept_index,
                })

        geometry_indices = [index for index, action in enumerate(actions) if key(action)[0] == "geometry"]
        for geometry_index in geometry_indices:
            cheap = [index for index, action in enumerate(actions)
                     if index != geometry_index and key(action)[0] in {"localize", "memory", "defer"}]
            if cheap:
                best_cheap = max(cheap, key=lambda index: oracle[index])
                measurement = {
                    "predicted": float(predicted[geometry_index] - max(predicted[index] for index in cheap)),
                    "oracle": float(oracle[geometry_index] - oracle[best_cheap]),
                }
                geometry_margins.append(measurement)
                for name in state_strata:
                    geometry_margins_by_stratum[name].append(measurement)

        policy_action, q_action, oracle_action = actions[policy_index], actions[q_index], actions[oracle_index]
        policy_regret = float(oracle.max() - oracle[policy_index])
        q_regret = float(oracle.max() - oracle[q_index])
        policy_cost, q_cost, oracle_cost = cost_of(policy_action), cost_of(q_action), cost_of(oracle_action)
        policy_regrets.append(policy_regret); q_regrets.append(q_regret)
        if policy_cost is not None and oracle_cost is not None:
            policy_cost_deltas.append(policy_cost - oracle_cost)
        if q_cost is not None and oracle_cost is not None:
            q_cost_deltas.append(q_cost - oracle_cost)
        if policy_index != q_index:
            pair = f"{key(policy_action)[0]}→{key(q_action)[0]}"
            pair_counter[pair] += 1
            pair_rows[pair].append({
                "policy_regret": policy_regret, "q_head_regret": q_regret,
                "policy_cost_minus_oracle_s": (float("nan") if policy_cost is None or oracle_cost is None
                                                else policy_cost - oracle_cost),
                "q_head_cost_minus_oracle_s": (float("nan") if q_cost is None or oracle_cost is None
                                                else q_cost - oracle_cost),
            })
            for name in state_strata:
                disagreement_by_stratum[name].append(pair_rows[pair][-1])
            disagreements.append({
                "sequence": row["sequence"], "frame_index": int(row["frame_index"]),
                "decision_index": int((row.get("dagger_diagnostics") or [{}])[0].get("decision_index", -1)),
                "strata": state_strata, "policy_kind": key(policy_action)[0], "q_head_kind": key(q_action)[0],
                "oracle_kind": key(oracle_action)[0], "policy_regret": policy_regret, "q_head_regret": q_regret,
                "policy_cost_minus_oracle_s": None if policy_cost is None or oracle_cost is None else policy_cost - oracle_cost,
                "q_head_cost_minus_oracle_s": None if q_cost is None or oracle_cost is None else q_cost - oracle_cost,
            })

    terminal_optimal = [item for item in terminal_accept if item["oracle_accept_optimal"]]
    result = {
        "schema_version": 1,
        "purpose": "read-only Q calibration and policy/Q disagreement audit",
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset), "dagger_states": len(records),
        "action_q_calibration": {kind: calibration(values) for kind, values in sorted(by_action.items())},
        "terminal_accept": {
            "states": len(terminal_accept), "oracle_accept_optimal_states": len(terminal_optimal),
            "policy_underaccept_states": sum(not item["policy_accept"] for item in terminal_optimal),
            "q_head_underaccept_states": sum(not item["q_accept"] for item in terminal_optimal),
            "policy_overaccept_states": sum(item["policy_accept"] and not item["oracle_accept_optimal"]
                                             for item in terminal_accept),
            "q_head_overaccept_states": sum(item["q_accept"] and not item["oracle_accept_optimal"]
                                             for item in terminal_accept),
            "accept_q_calibration": calibration(terminal_accept),
            "accept_vs_best_other_margin": margin_summary([
                {"predicted": item["margin_predicted"], "oracle": item["margin_oracle"]} for item in terminal_accept]),
            "by_stratum": {},
        },
        "acquisition": {
            "geometry_q_calibration": calibration(by_action.get("geometry", [])),
            "localize_q_calibration": calibration(by_action.get("localize", [])),
            "memory_q_calibration": calibration(by_action.get("memory", [])),
            "geometry_vs_cheap_margin": margin_summary(geometry_margins),
            "geometry_vs_cheap_margin_by_stratum": {name: margin_summary(values)
                                                       for name, values in sorted(geometry_margins_by_stratum.items())},
        },
        "stratified_action_q_calibration": {name: {kind: calibration(values) for kind, values in sorted(kinds.items())}
                                              for name, kinds in sorted(by_stratum_action.items())},
        "policy_q_disagreement": {
            "states": len(records), "disagreement_states": len(disagreements),
            "disagreement_fraction": len(disagreements) / len(records),
            "by_policy_to_q_action": dict(sorted(pair_counter.items())),
            "by_policy_to_q_action_outcome": {
                name: {metric: numeric([item[metric] for item in values])
                       for metric in ("policy_regret", "q_head_regret", "policy_cost_minus_oracle_s",
                                      "q_head_cost_minus_oracle_s")}
                for name, values in sorted(pair_rows.items())},
            "by_stratum": {
                name: {"count": len(values),
                       "policy_oracle_regret": numeric([item["policy_regret"] for item in values]),
                       "q_head_oracle_regret": numeric([item["q_head_regret"] for item in values])}
                for name, values in sorted(disagreement_by_stratum.items())},
            "policy_oracle_regret": numeric(policy_regrets), "q_head_oracle_regret": numeric(q_regrets),
            "policy_cost_minus_oracle_s": numeric(policy_cost_deltas), "q_head_cost_minus_oracle_s": numeric(q_cost_deltas),
            "examples": sorted(disagreements, key=lambda item: item["policy_regret"], reverse=True)[:40],
        },
    }
    # Terminal strata need the exact terminal-state subset, so derive it after the general pass.
    terminal_by_stratum: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in records:
        observation = row["observation"]
        candidate = observation.get("candidate") or {}
        if not (candidate.get("pose_available") and candidate.get("verified")):
            continue
        actions = [a for a in row["action_values"] if is_deployment_legal(observation, a)]
        accept = next((a for a in actions if key(a)[0] == "accept"), None)
        if accept is None:
            continue
        _, predicted = predict(policy, observation, actions)
        accept_index = actions.index(accept)
        for name in strata(observation):
            terminal_by_stratum[name].append({"predicted": float(predicted[accept_index]), "oracle": float(accept["q_star"])})
    result["terminal_accept"]["by_stratum"] = {name: calibration(values)
                                                   for name, values in sorted(terminal_by_stratum.items())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "q_calibration_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "q_calibration_audit.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "dagger_states": result["dagger_states"], "terminal_accept": result["terminal_accept"],
        "acquisition": result["acquisition"], "policy_q_disagreement": result["policy_q_disagreement"],
    }, indent=2))


if __name__ == "__main__":
    main()
