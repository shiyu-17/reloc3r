#!/usr/bin/env python3
"""Audit Geometry opportunities on live states visited by a frozen policy.

The rollout trace must have been created with ``--audit-policy-states``.  The
policy never sees these Oracle labels: they are added only after its decision
by forking the exact live environment state.  This separates missed Geometry
opportunities from a low Geometry rate that is correct under the utility.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from tools.eval_active_evidence_policies import load_query_gt, success_pose


def action_key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind")), tuple(int(x) for x in action.get("references", ()))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def numeric(values: Iterable[float]) -> Dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"count": int(len(data)), "mean": float(np.mean(data)), "median": float(np.median(data)),
            "p10": float(np.quantile(data, 0.1)), "p90": float(np.quantile(data, 0.9))}


def safe_divide(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def audit_decisions(decisions: Sequence[Mapping[str, Any]], *, lambda_cost: float) -> Dict[str, Any]:
    opportunities = hits = calls = optimal_calls = 0
    missed_regret, all_geometry_regret, delta_p, lambda_delta_cost, advantage_margin = [], [], [], [], []
    positive_p_cost_dominated = ranker_prefers_geometry = controller_prefers_geometry = 0
    recovery_opportunities = recovery_hits = recovery_calls = recovery_decisions = 0
    by_sequence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in decisions:
        label = item.get("oracle_label") or {}
        values = label.get("oracle_action_values") or []
        if not values:
            continue
        q_by_key = {action_key(node["action"]): float(node["action_value"]["q_star"]) for node in values}
        geometry = [(key, value) for key, value in q_by_key.items() if key[0] == "geometry"]
        if not geometry:
            continue
        geometry_key, geometry_q = max(geometry, key=lambda pair: pair[1])
        best_q = max(q_by_key.values())
        selected_key = action_key(item["action"])
        selected_q = q_by_key.get(selected_key)
        if selected_q is None:
            raise RuntimeError(f"selected action absent from Oracle labels: {selected_key}")
        is_opportunity = geometry_q >= best_q - 1e-12
        is_geometry = selected_key[0] == "geometry"
        state = item.get("state") or {}
        recovery = bool(state.get("candidate"))
        sequence = str(item.get("sequence", "unknown"))
        if recovery:
            recovery_decisions += 1
        if is_opportunity:
            opportunities += 1; by_sequence[sequence]["opportunities"] += 1
            all_geometry_regret.append(best_q - geometry_q)
            if recovery:
                recovery_opportunities += 1
            if is_geometry:
                hits += 1; by_sequence[sequence]["hits"] += 1
                if recovery:
                    recovery_hits += 1
            else:
                missed_regret.append(best_q - selected_q)
        if is_geometry:
            calls += 1; by_sequence[sequence]["calls"] += 1
            if geometry_q >= best_q - 1e-12:
                optimal_calls += 1
            if recovery:
                recovery_calls += 1

        # Controller decomposition requires the XGB model receipt.  It is
        # online-predictable and is merely logged before the Oracle is joined.
        receipt = item.get("model_decision") or {}
        acquisition_nodes = [node for node in values if str(node["action"].get("kind")) in {"localize", "geometry", "memory"}]
        probabilities = receipt.get("recovery_probability") or []
        costs = receipt.get("predicted_incremental_cost_s") or []
        ranks = receipt.get("rank_score") or []
        explicit = receipt.get("explicit_advantage") or []
        if len(acquisition_nodes) == len(probabilities) == len(costs):
            kinds = [str(node["action"].get("kind")) for node in acquisition_nodes]
            if "geometry" in kinds:
                g = kinds.index("geometry")
                cheap = [i for i, kind in enumerate(kinds) if kind in {"localize", "memory"}]
                if cheap:
                    cheap_best = max(cheap, key=lambda i: float(probabilities[i]) - lambda_cost * float(costs[i]))
                    p_gap = float(probabilities[g]) - float(probabilities[cheap_best])
                    cost_term = lambda_cost * (float(costs[g]) - float(costs[cheap_best]))
                    margin = p_gap - cost_term
                    delta_p.append(p_gap); lambda_delta_cost.append(cost_term); advantage_margin.append(margin)
                    positive_p_cost_dominated += int(p_gap > 0.0 and margin < 0.0)
                    if len(ranks) == len(kinds):
                        ranker_prefers_geometry += int(float(ranks[g]) >= max(float(ranks[i]) for i in cheap))
                    if len(explicit) == len(kinds):
                        controller_prefers_geometry += int(float(explicit[g]) >= max(float(explicit[i]) for i in cheap))
    return {
        "geometry_opportunity": {
            "oracle_geometry_optimal": opportunities,
            "policy_geometry_calls": calls,
            "hits": hits,
            "recall": safe_divide(hits, opportunities),
            "precision": safe_divide(optimal_calls, calls),
            "missed_geometry_regret": numeric(missed_regret),
            "geometry_oracle_regret": numeric(all_geometry_regret),
            "by_sequence": {key: dict(value) for key, value in sorted(by_sequence.items())},
        },
        "recovery_state": {
            "decisions_with_candidate": recovery_decisions,
            "oracle_geometry_optimal": recovery_opportunities,
            "policy_geometry_calls": recovery_calls,
            "geometry_opportunity_recall": safe_divide(recovery_hits, recovery_opportunities),
        },
        "controller_decomposition": {
            "states_with_geometry_and_cheap_alternative": len(delta_p),
            "delta_p": numeric(delta_p),
            "lambda_delta_cost": numeric(lambda_delta_cost),
            "explicit_advantage_margin": numeric(advantage_margin),
            "positive_probability_gain_but_cost_dominated": positive_p_cost_dominated,
            "fraction_positive_probability_gain_but_cost_dominated": safe_divide(positive_p_cost_dominated, len(delta_p)),
            "ranker_prefers_geometry": ranker_prefers_geometry,
            "explicit_controller_prefers_geometry": controller_prefers_geometry,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions, frame_rows = [], []
    # Accept both evaluator layouts: one root containing sequence directories
    # and the compact layout where all sequence traces are directly under
    # ``trace_root/traces``.
    paths = sorted(set(args.trace_root.glob("traces/*_learned.jsonl")) |
                   set(args.trace_root.glob("*/traces/*_learned.jsonl")))
    for path in paths:
        sequence = path.name.removesuffix("_learned.jsonl")
        gt = load_query_gt(args.data_root, args.gt_root, sequence)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            correct = bool(row.get("accepted")) and success_pose(row.get("pose"), gt[int(row["frame_index"])])
            recovery = any(bool((d.get("state") or {}).get("candidate")) for d in row.get("decisions", []))
            frame_rows.append({"sequence": sequence, "frame_index": int(row["frame_index"]),
                               "recovery_state": recovery, "success": correct})
            for decision in row.get("decisions", []):
                if decision.get("oracle_label") is None:
                    raise ValueError(f"{path} lacks --audit-policy-states labels")
                decisions.append({**decision, "sequence": sequence, "frame_index": int(row["frame_index"])})
    report = audit_decisions(decisions, lambda_cost=args.lambda_cost)
    recovery_frames = [row for row in frame_rows if row["recovery_state"]]
    report.update({"schema_version": 1, "purpose": "Stage 3C.5 live-state Geometry opportunity audit",
                   "trace_root": str(args.trace_root), "policy_decisions": len(decisions),
                   "recovery_frame_success": {"frames": len(recovery_frames),
                      "correct_accepted": sum(row["success"] for row in recovery_frames),
                      "success": safe_divide(sum(row["success"] for row in recovery_frames), len(recovery_frames))},
                   "online_boundary": "Oracle labels/GT are joined only after the frozen policy has selected an action from its live observation."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
