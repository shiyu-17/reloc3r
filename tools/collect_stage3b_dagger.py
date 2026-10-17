#!/usr/bin/env python3
"""Collect policy-induced states and relabel them with the terminal Oracle.

This is one DAgger round, not an online policy component.  The learned MLP
selects actions from policy-visible evidence; after each decision the evaluator
forks that exact state and computes finite-horizon Q* labels with GT only at
terminal scoring time.  The output JSONL can be appended to Stage 3B-v2 SFT
data without leaking labels into the rollout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tools.eval_active_evidence_policies import SEQUENCES, load_query_gt
from tools.eval_active_evidence_sequential import (label_oracle_state, load_utility,
                                                    make_backend, run_policy, summarize)
from tools.stage3b_policy import StructuredPolicy
from tools.stage3c6_utility_policy import UtilityComponentPolicy
from tools.stage3d_geometry_advantage import GeometryAdvantagePolicy
from tools.stage3e_recovery_ranking_policy import RecoveryCostRankingPolicy


def key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind")), tuple(int(x) for x in action.get("references", ()))


def diagnostic(decision: Mapping[str, Any]) -> Dict[str, Any]:
    labels = decision["oracle_label"]["oracle_action_values"]
    values = {key(node["action"]): float(node["action_value"]["q_star"]) for node in labels}
    chosen_key = key(decision["action"])
    if chosen_key not in values:
        raise RuntimeError(f"selected action is absent from its Oracle label set: {chosen_key}")
    selected = values[chosen_key]
    best = max(values.values())
    tolerance = 1e-12
    kind = chosen_key[0]
    return {
        "decision_index": int(decision["decision_index"]), "selected_kind": kind,
        "selected_q": selected, "best_q": best, "q_gap": best - selected,
        "optimal": abs(best - selected) <= tolerance,
        "premature_accept": kind == "accept" and selected < best - tolerance,
        "should_geometry_but_accept": kind == "accept" and any(
            action_kind == "geometry" and value > selected + tolerance
            for (action_kind, _), value in values.items()),
        "should_defer_but_accept": kind == "accept" and any(
            action_kind == "defer" and value > selected + tolerance
            for (action_kind, _), value in values.items()),
        "candidate_verified": bool((decision["state"].get("candidate") or {}).get("verified")),
    }


def aggregate(diagnostics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(diagnostics)
    accepts = [row for row in rows if row["selected_kind"] == "accept"]
    def fraction(items): return len(items) / len(rows) if rows else 0.0
    def conditional(items): return len(items) / len(accepts) if accepts else 0.0
    by_kind: Dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["selected_kind"])].append(float(row["q_gap"]))
    return {
        "policy_decisions": len(rows), "accept_decisions": len(accepts),
        "first_step_action_accuracy": (sum(row["optimal"] for row in rows if row["decision_index"] == 0) /
                                        max(1, sum(row["decision_index"] == 0 for row in rows))),
        "second_step_off_expert_action_accuracy": (sum(row["optimal"] for row in rows if row["decision_index"] > 0) /
                                                     max(1, sum(row["decision_index"] > 0 for row in rows))),
        "premature_accept_fraction_all_decisions": fraction([row for row in rows if row["premature_accept"]]),
        "premature_accept_fraction_of_accepts": conditional([row for row in accepts if row["premature_accept"]]),
        "should_geometry_but_accept_fraction_of_accepts": conditional(
            [row for row in accepts if row["should_geometry_but_accept"]]),
        "should_defer_but_accept_fraction_of_accepts": conditional(
            [row for row in accepts if row["should_defer_but_accept"]]),
        "mean_q_gap": float(np.mean([row["q_gap"] for row in rows])) if rows else 0.0,
        "q_gap_by_selected_action": {kind: {"count": len(gaps), "mean": float(np.mean(gaps)),
                                              "max": float(np.max(gaps))}
                                     for kind, gaps in sorted(by_kind.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learned-policy-checkpoint", type=Path,
                        help="required only when --rollout-policy=learned")
    parser.add_argument("--terminal-policy-checkpoint", type=Path,
                        help="optional frozen Stage 3C.6 terminal controller for non-learned rollout")
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--verifier-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--backend", choices=("cpu_orb", "reloc3r"), default="reloc3r")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learned-policy-device", default="cpu")
    parser.add_argument("--learned-decision", choices=("policy", "value", "terminal_value"), default="policy")
    parser.add_argument("--learned-policy-family", choices=("stage3b", "stage3c6_utility", "stage3d_geometry_advantage", "stage3e_recovery_cost_ranking"), default="stage3b",
                        help="Controller used to induce the relabelled state distribution.")
    parser.add_argument("--rollout-policy", choices=("learned", "feedback_rule"), default="learned",
                        help="Use feedback_rule only for data coverage; all states are still relabelled by Oracle.")
    parser.add_argument("--stage3d-preaction-features", action="store_true",
                        help="Attach legal cheap/temporal/appearance evidence before each Geometry decision.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--budget-ms", type=float, default=30000.0)
    # These are only used by the explicitly selected coverage controller.
    # Keeping them versioned here makes its induced state distribution fully
    # reproducible rather than inheriting evaluator defaults implicitly.
    parser.add_argument("--geometry-schedule-rate", type=float, default=0.35)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.8)
    parser.add_argument("--router-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.oracle_depth != args.max_acquisitions:
        parser.error("--oracle-depth must equal --max-acquisitions")
    args.utility = load_utility(args.utility_config)
    if args.rollout_policy == "learned" and args.learned_policy_checkpoint is None:
        parser.error("--learned-policy-checkpoint is required when --rollout-policy=learned")
    if args.rollout_policy != "learned":
        policy = None
    elif args.learned_policy_family == "stage3b":
        policy = StructuredPolicy.load(args.learned_policy_checkpoint, device=args.learned_policy_device,
                                       decision=args.learned_decision)
    elif args.learned_policy_family == "stage3c6_utility":
        policy = UtilityComponentPolicy.load(args.learned_policy_checkpoint)
    elif args.learned_policy_family == "stage3d_geometry_advantage":
        policy = GeometryAdvantagePolicy.load(args.learned_policy_checkpoint)
    else:
        policy = RecoveryCostRankingPolicy.load(args.learned_policy_checkpoint, device=args.learned_policy_device)
    if args.terminal_policy_checkpoint is None:
        terminal_policy = None
    else:
        if args.rollout_policy == "learned":
            parser.error("--terminal-policy-checkpoint is only valid with a non-learned --rollout-policy")
        if args.learned_policy_family != "stage3c6_utility":
            parser.error("terminal-aligned rollout requires --learned-policy-family stage3c6_utility")
        terminal_policy = UtilityComponentPolicy.load(args.terminal_policy_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_rows, diagnostics, summaries = [], [], {}
    for sequence in args.sequences:
        gt = load_query_gt(args.data_root, args.gt_root, sequence)
        end = min(args.start + args.max_frames, len(gt))
        backend = make_backend(args, sequence)
        def labeler(env, groups, remaining, geometry_used, memory_used, acquisitions_used, *, _gt=gt):
            return label_oracle_state(args, env, groups, remaining, geometry_used, memory_used,
                                      acquisitions_used, _gt[env.state.frame_index])
        run = run_policy(args, sequence, args.rollout_policy, args.start, end, backend=backend,
                         learned_policy=policy, terminal_policy=terminal_policy, oracle_labeler=labeler)
        summaries[sequence] = summarize(run["records"], gt)
        (args.output_dir / f"{sequence}_rollout.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in run["records"]), encoding="utf-8")
        for rollout in run["records"]:
            for decision in rollout["decisions"]:
                if decision["oracle_label"] is None:
                    continue
                decision_diagnostic = diagnostic(decision)
                row = {"sequence": sequence, "frame_index": rollout["frame_index"],
                       "budget_s": rollout["budget_s"], "oracle_action_values":
                       decision["oracle_label"]["oracle_action_values"],
                       "oracle_terminal_programs": decision["oracle_label"]["oracle_terminal_programs"],
                       "dagger_round": 2, "decision_index": decision["decision_index"],
                       "policy_action": decision["action"],
                       "dagger_diagnostic": decision_diagnostic}
                label_rows.append(row)
                diagnostics.append(decision_diagnostic)
    (args.output_dir / "dagger_oracle_labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8")
    result = {"schema_version": 2, "purpose": "policy-induced terminal-Q labels with optional Stage 3D pre-action evidence",
              "config": vars(args), "rollout_summary": summaries,
              "diagnostics": aggregate(diagnostics), "labelled_policy_states": len(label_rows),
              "online_information_boundary": "MLP rollout sees only observations; Oracle Q/GT are added after each action choice"}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
