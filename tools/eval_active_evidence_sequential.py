#!/usr/bin/env python3
"""Closed-loop evaluation for the visual-localization evidence agent.

Unlike the earlier pilot, this evaluator does not pre-compute an ordering and
blindly execute it.  Every next acquisition is selected after reading the
previous online execution/verification receipt.  Costs are charged from the
receipt's measured elapsed time; action count is retained as a safety limit.

The CPU backend is an interface smoke path.  With ``--backend reloc3r`` a
selected geometry action executes RoMa + map triangulation + PnP online.  No
cached AVL-004 prediction is read by either policy or oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.active_evidence_episodes import load_episode_spans
from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend
from tools.eval_active_evidence_policies import (
    SEQUENCES, audit_flags, groups_from_observation, load_query_gt, success_pose,
)
from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend
from tools.active_evidence_protocol import (ACTION_PROTOCOL_VERSION,
                                            EVALUATOR_RECORD_VERSION,
                                            ONLINE_RECORD_VERSION)
from tools.stage3b_policy import StructuredPolicy, legal_actions
from tools.stage3c_recovery_policy import RecoveryCostPolicy
from tools.stage3c_xgboost_policy import XGBRecoveryPolicy
from tools.stage3c6_utility_policy import UtilityComponentPolicy


POLICIES = (
    "fixed", "fixed_geometry", "geometry_greedy", "uncertainty_trigger",
    "one_shot_router", "random", "feedback_rule", "learned",
)


DEFAULT_UTILITY = {
    "schema_version": 1,
    "success_reward": 1.0,
    "cost_penalty_per_s": 0.05,
    "false_accept_penalty": 2.0,
    "recovery_penalty_per_frame": 0.02,
    "memory_pollution_penalty": 1.0,
    "definition": (
        "U(tau)=success_reward*correct_accept - cost_penalty_per_s*measured_tool_cost_s "
        "- false_accept_penalty*false_accept - recovery_penalty_per_frame*reject_streak_after "
        "- memory_pollution_penalty*incorrect_memory_write"
    ),
}


def load_utility(path: Optional[Path]) -> Dict[str, Any]:
    """Load the evaluator-only terminal-utility contract.

    The policy and online backend never see these weights.  Keeping them in a
    versioned file makes action-value labels reproducible and prevents an
    implicit success-only Oracle from being mistaken for a cost-aware target.
    """
    value = dict(DEFAULT_UTILITY)
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, Mapping):
            raise ValueError("utility config must be a JSON object")
        value.update(supplied)
    required = ("success_reward", "cost_penalty_per_s", "false_accept_penalty",
                "recovery_penalty_per_frame", "memory_pollution_penalty")
    for key in required:
        if not isinstance(value.get(key), (int, float)) or float(value[key]) < 0.0:
            raise ValueError(f"utility field {key} must be a non-negative number")
    return value


def terminal_utility(leaf: Mapping[str, Any], utility: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute a single evaluator-only terminal utility for an executed leaf."""
    accepted = bool(leaf["accepted"])
    correct = bool(leaf["success"])
    false_accept = bool(leaf["false_accept"])
    state = leaf["env"].state
    assert state is not None
    memory_pollution = bool(false_accept and state.memory and
                            state.memory[-1].frame_index == state.frame_index - 1)
    recovery_time = int(state.reject_streak) if not correct else 0
    cost = float(leaf["cost"])
    result = (
        float(utility["success_reward"]) * float(correct)
        - float(utility["cost_penalty_per_s"]) * cost
        - float(utility["false_accept_penalty"]) * float(false_accept)
        - float(utility["recovery_penalty_per_frame"]) * recovery_time
        - float(utility["memory_pollution_penalty"]) * float(memory_pollution)
    )
    return {
        "utility": result,
        "terminal_type": ("correct_accept" if correct else
                          "false_accept" if accepted else "defer"),
        "correct_accept": correct,
        "false_accept": false_accept,
        "measured_tool_cost_s": cost,
        "recovery_time_frames": recovery_time,
        "memory_pollution": memory_pollution,
    }


def make_backend(args, sequence: str):
    retrieval = args.retrieval_root / sequence / "retrieval.npz"
    if args.backend == "cpu_orb":
        return CpuOrbLocalizationBackend(args.data_root, args.mapping_gt, sequence,
                                         retrieval, nfeatures=args.nfeatures)
    return LILocBenchReloc3rBackend(args.data_root, args.mapping_gt, sequence,
                                    retrieval, device=args.device,
                                    verifier_config=args.verifier_config,
                                    preaction_features=bool(getattr(args, "stage3d_preaction_features", False)))


def receipt_cost(event: Mapping[str, Any]) -> float:
    """Return only tool-reported measured time, never evaluator wall time."""
    execution = event.get("execution", {})
    evidence = event.get("evidence", {})
    value = execution.get("elapsed_s", evidence.get("elapsed_s", 0.0))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def do_step(env: AgenticLocalizationEnv, action: LocalizationAction):
    observation, _, _, info = env.step(action)
    return observation, receipt_cost(info["event"]), info["event"]


def rank_groups(groups: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted(range(len(groups)), key=lambda i: (
        float(groups[i].get("mean_retrieval_score", 0.0)),
        float(groups[i].get("map_spread_m", 0.0))), reverse=True)


def verification_risk(observation) -> float:
    candidate = observation.candidate or {}
    verification = candidate.get("verification", {})
    # These fields come from online tools only.  Missing evidence is itself
    # treated as uncertainty, which is what makes the rule acquire more.
    values = [verification.get(key) for key in
              ("total_risk", "independent_risk", "quality_risk", "pnp_risk")]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if numeric:
        return max(numeric)
    return 1.0


def stable_frame_fraction(sequence: str, frame_index: int) -> float:
    """A reproducible, GT-free schedule independent of retrieval evidence."""
    digest = hashlib.sha256(f"{sequence}:{frame_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2 ** 64)


def retrieval_router_score(groups: Sequence[Mapping[str, Any]]) -> float:
    """One-shot, policy-visible score: lower retrieval confidence means geometry."""
    if not groups:
        return float("-inf")
    first = groups[rank_groups(groups)[0]]
    score = float(first.get("mean_retrieval_score", 0.0))
    # A large map span is a helpful triangulation opportunity, not evidence of
    # query correctness.  It is intentionally the only second feature used by
    # this transparent lightweight router.
    spread = float(first.get("map_spread_m", 0.0))
    return score + 0.02 * min(spread, 20.0)


def choose_acquisition(policy: str, observation, groups: Sequence[Mapping[str, Any]],
                       remaining: Sequence[int], geometry_used: bool,
                       rng: random.Random, *,
                       geometry_schedule_rate: float,
                       uncertainty_threshold: float,
                       router_threshold: float) -> Optional[Dict[str, Any]]:
    """Feedback-dependent action selection; never reads evaluation GT."""
    candidate = observation.candidate or {}
    failed_verification = bool(candidate.get("pose_available")) and not bool(candidate.get("verified")) \
        and bool(candidate.get("verification"))
    last_refs = tuple(int(x) for x in candidate.get("reference_indices", ()))
    ranked = [i for i in rank_groups(groups) if i in remaining]
    first_step = not candidate and not geometry_used and len(remaining) == len(groups)
    candidates = legal_actions(observation, remaining, geometry_used=geometry_used,
                               allow_acquisition=True)
    acquisitions = [item for item in candidates if not item.get("is_terminal_action", False)]
    localize = [item for item in acquisitions if item["kind"] == "localize"]
    geometry = [item for item in acquisitions if item["kind"] == "geometry"]
    def geometry_for(refs: Tuple[int, ...]):
        return next((item for item in geometry if tuple(item["references"]) == refs), None)
    preferred_refs = last_refs or (tuple(groups[rank_groups(groups)[0]]["reference_indices"]) if groups else ())
    preferred_geometry = geometry_for(preferred_refs) or (geometry[0] if geometry else None)
    ranked_localize = [next((item for item in localize
                            if int(item.get("reference_group", -1)) == index), None)
                       for index in rank_groups(groups) if index in remaining]
    ranked_localize = [item for item in ranked_localize if item is not None]
    if policy == "geometry_greedy":
        if preferred_geometry is not None:
            return preferred_geometry
        if ranked_localize:
            return ranked_localize[0]
    elif policy == "fixed_geometry":
        if not geometry_used and stable_frame_fraction(observation.sequence, observation.frame_index) < geometry_schedule_rate:
            return preferred_geometry
        if ranked_localize:
            return ranked_localize[0]
        if preferred_geometry is not None:
            return preferred_geometry
    elif policy == "uncertainty_trigger":
        # The only geometry trigger is a pre-registered scalar threshold over
        # the first candidate's verifier receipt.  This is a fair simple-rule
        # comparator, not an adaptive controller.
        if failed_verification and not geometry_used and verification_risk(observation) >= uncertainty_threshold:
            return preferred_geometry
        if ranked_localize:
            return ranked_localize[0]
    elif policy == "one_shot_router":
        # Decide localize versus geometry once from retrieval-only features;
        # after that, it may try another cheap group but never changes to
        # geometry based on verification feedback.
        if first_step and not geometry_used and retrieval_router_score(groups) < router_threshold:
            return preferred_geometry
        if ranked_localize:
            return ranked_localize[0]
    elif policy == "random":
        if acquisitions:
            return rng.choice(acquisitions)
    elif policy == "feedback_rule":
        # A failed independent/quality check can trigger stronger geometry
        # immediately.  A low-risk failure instead tries a different map view
        # first, preserving the cheaper path where it remains plausible.
        if failed_verification and not geometry_used and verification_risk(observation) >= 0.8:
            return preferred_geometry
        if ranked_localize:
            return ranked_localize[0]
        if preferred_geometry is not None:
            return preferred_geometry
    else:  # fixed: control program, but still stops/continues on verification feedback.
        if localize:
            return localize[0]
        if preferred_geometry is not None:
            return preferred_geometry
    return None


def run_policy(args, sequence: str, policy: str, start: int, end: int, *,
               backend=None, learned_policy: Optional[Any] = None,
               terminal_policy: Optional[Any] = None,
               oracle_labeler=None) -> Dict[str, Any]:
    # The backend is model/read-only state.  Reusing it across policies avoids
    # repeatedly loading 1.7GB Reloc3r weights; environments themselves stay
    # isolated and retain their own causal Memory/state.
    backend = make_backend(args, sequence) if backend is None else backend
    # verify/accept/defer can accompany at most ``max_acquisitions`` tools.
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=3 * args.max_acquisitions + 2,
                                 budget_s=args.budget_ms / 1000.0,
                                 allow_memory_query=False)
    observation = env.reset(sequence, start_frame=start)
    rng = random.Random(args.seed + sum(map(ord, sequence + policy)))
    records: List[Dict[str, Any]] = []
    if policy == "learned" and learned_policy is None:
        raise ValueError("learned policy requested without --learned-policy-checkpoint")
    if terminal_policy is not None and policy == "learned":
        raise ValueError("terminal_policy is only valid for a non-learned acquisition controller")
    while not observation.done and observation.frame_index < end:
        frame_index = observation.frame_index
        groups = groups_from_observation(observation)
        remaining = list(range(len(groups)))
        geometry_used = False
        total_cost = 0.0
        acquisitions: List[Dict[str, Any]] = []
        accepted_pose = None
        memory_write = False
        terminal_action = None
        decisions: List[Dict[str, Any]] = []
        budget_overshoot = False
        frame_start = time.perf_counter()
        while terminal_action is None:
            model_decision = None
            if policy == "learned":
                acquisition_allowed = (len(acquisitions) < args.max_acquisitions and
                                       total_cost < args.budget_ms / 1000.0)
                candidates = legal_actions(observation, remaining, geometry_used=geometry_used,
                                           allow_acquisition=acquisition_allowed)
                selected, model_decision = learned_policy.choose(observation, candidates)
                candidate = candidates[selected]
                oracle_label = (oracle_labeler(env, groups, remaining, geometry_used, False,
                                                len(acquisitions))
                                if oracle_labeler is not None else None)
                kind = candidate["kind"]
                decisions.append({"decision_index": len(decisions),
                                  "state": observation.as_dict(), "action": candidate,
                                  "model_decision": model_decision,
                                  "oracle_label": oracle_label,
                                  "acquisitions_before": len(acquisitions)})
                if kind in {"accept", "reject", "defer"}:
                    if kind == "accept":
                        accepted_pose = env.state.current_candidate.pose.copy()
                        observation, _, terminal_event = do_step(env, LocalizationAction.accept())
                        memory_write = "memory_id" in terminal_event
                    elif kind == "reject":
                        observation, _, _ = do_step(env, LocalizationAction.reject())
                    else:
                        observation, _, _ = do_step(env, LocalizationAction.defer())
                    terminal_action = kind
                    break
                choice = candidate
            else:
                if len(acquisitions) >= args.max_acquisitions or total_cost >= args.budget_ms / 1000.0:
                    break
                choice = choose_acquisition(policy, observation, groups, remaining,
                                            geometry_used, rng,
                                            geometry_schedule_rate=args.geometry_schedule_rate,
                                            uncertainty_threshold=args.uncertainty_threshold,
                                            router_threshold=args.router_threshold)
            if choice is None:
                break
            kind, refs = str(choice["kind"]), tuple(choice.get("references", ()))
            # Coverage rollouts (for example feedback_rule during Stage 3D
            # collection) must be relabelled at the exact policy-visible
            # pre-action state too.  They do not expose these labels online;
            # this merely makes their state distribution usable by DAgger.
            if policy != "learned" and oracle_labeler is not None:
                decisions.append({
                    "decision_index": len(decisions), "state": observation.as_dict(),
                    "action": choice,
                    "model_decision": {"controller": policy, "reason": "coverage_rollout"},
                    "oracle_label": oracle_labeler(env, groups, remaining, geometry_used, False,
                                                     len(acquisitions)),
                    "acquisitions_before": len(acquisitions),
                })
            if kind == "localize":
                group_index = next((i for i in remaining if tuple(groups[i]["reference_indices"]) == refs), None)
                if group_index is not None:
                    remaining.remove(group_index)
                action = LocalizationAction.localize(*refs)
            elif kind == "geometry":
                geometry_used = True
                action = LocalizationAction.geometry(*refs)
            else:
                raise RuntimeError("v2 action contract exposed unsupported acquisition: " + kind)
            observation, cost, event = do_step(env, action)
            total_cost += cost
            acquisitions.append({"kind": kind, "references": list(refs),
                                 "cost_s": cost, "execution": event.get("execution", {}),
                                 "candidate": observation.candidate,
                                 "learned_decision": model_decision,
                                 # Evaluator-only pose for post-hoc labels.  It is
                                 # never supplied to policy selection or backend.
                                 "candidate_pose": (None if env.state.current_candidate is None or
                                                    env.state.current_candidate.pose is None else
                                                    env.state.current_candidate.pose.tolist())})
            if total_cost >= args.budget_ms / 1000.0:
                budget_overshoot = total_cost > args.budget_ms / 1000.0
                break
            if observation.candidate and observation.candidate.get("pose_available"):
                observation, cost, event = do_step(env, LocalizationAction.verify())
                total_cost += cost
                acquisitions[-1]["verification"] = event.get("evidence", {})
                acquisitions[-1]["verification_cost_s"] = cost
                acquisitions[-1]["verified_candidate"] = observation.candidate
                # This diagnostic keeps a non-learned acquisition program but
                # applies the frozen controller only to Accept/Defer.  It
                # separates acquisition quality from the legacy auto-Accept
                # interface without changing the frozen baseline itself.
                if (policy != "learned" and terminal_policy is not None and
                        observation.candidate and observation.candidate.get("verified")):
                    terminal_candidates = legal_actions(
                        observation, [], geometry_used=geometry_used,
                        allow_acquisition=False)
                    selected, terminal_metadata = terminal_policy.choose(observation, terminal_candidates)
                    terminal_candidate = terminal_candidates[selected]
                    terminal_kind = str(terminal_candidate["kind"])
                    if terminal_kind not in {"accept", "defer", "reject"}:
                        raise RuntimeError("terminal controller selected a non-terminal action")
                    terminal_oracle = (oracle_labeler(env, groups, remaining, geometry_used, False,
                                                      len(acquisitions))
                                       if oracle_labeler is not None else None)
                    decisions.append({
                        "decision_index": len(decisions), "state": observation.as_dict(),
                        "action": terminal_candidate, "model_decision": terminal_metadata,
                        "oracle_label": terminal_oracle,
                        "acquisitions_before": len(acquisitions),
                    })
                    if terminal_kind == "accept":
                        accepted_pose = env.state.current_candidate.pose.copy()
                        observation, _, terminal_event = do_step(env, LocalizationAction.accept())
                        memory_write = "memory_id" in terminal_event
                    elif terminal_kind == "reject":
                        observation, _, _ = do_step(env, LocalizationAction.reject())
                    else:
                        observation, _, _ = do_step(env, LocalizationAction.defer())
                    terminal_action = terminal_kind
                    break
                # v2 keeps the deterministic rule's terminal choice explicit:
                # it selects the same shared ``accept`` action exposed to all
                # policies instead of relying on a hidden verifier side effect.
                if policy != "learned" and observation.candidate and observation.candidate.get("verified"):
                    if oracle_labeler is not None:
                        terminal_candidates = legal_actions(
                            observation, [], geometry_used=geometry_used,
                            allow_acquisition=False)
                        accept_candidate = next(item for item in terminal_candidates
                                                if item["kind"] == "accept")
                        decisions.append({
                            "decision_index": len(decisions), "state": observation.as_dict(),
                            "action": accept_candidate,
                            "model_decision": {"controller": policy,
                                               "reason": "deterministic_verified_accept"},
                            "oracle_label": oracle_labeler(
                                env, groups, remaining, geometry_used, False,
                                len(acquisitions)),
                            "acquisitions_before": len(acquisitions),
                        })
                    accepted_pose = env.state.current_candidate.pose.copy()
                    observation, _, terminal_event = do_step(env, LocalizationAction.accept())
                    memory_write = "memory_id" in terminal_event
                    terminal_action = "accept_rule"
                    break
        if terminal_action is None:
            if oracle_labeler is not None:
                terminal_candidates = legal_actions(
                    observation, remaining, geometry_used=geometry_used,
                    allow_acquisition=False)
                defer_candidate = next(item for item in terminal_candidates
                                       if item["kind"] == "defer")
                decisions.append({
                    "decision_index": len(decisions), "state": observation.as_dict(),
                    "action": defer_candidate,
                    "model_decision": {"controller": policy, "reason": "budget_or_no_action_defer"},
                    "oracle_label": oracle_labeler(env, groups, remaining, geometry_used, False,
                                                     len(acquisitions)),
                    "acquisitions_before": len(acquisitions),
                })
            observation, _, _ = do_step(env, LocalizationAction.defer())
            terminal_action = "defer_fallback"
        online_decisions = []
        evaluator_decisions = []
        for decision in decisions:
            online = {key: value for key, value in decision.items() if key != "oracle_label"}
            online_decisions.append(online)
            if decision.get("oracle_label") is not None:
                evaluator_decisions.append({"decision_index": decision["decision_index"],
                                            "oracle_label": decision["oracle_label"]})
        online_acquisitions = [{key: value for key, value in item.items() if key != "candidate_pose"}
                               for item in acquisitions]
        records.append({"schema_version": ACTION_PROTOCOL_VERSION,
                        "sequence": sequence, "frame_index": frame_index, "policy": policy,
                        "online": {"schema_version": ONLINE_RECORD_VERSION,
                                   "acquisitions": online_acquisitions,
                                   "tool_actions": len(acquisitions),
                                   "terminal_action": terminal_action,
                                   "decisions": online_decisions,
                                   "measured_tool_cost_s": total_cost,
                                   "budget_s": args.budget_ms / 1000.0,
                                   "budget_overshoot": budget_overshoot,
                                   "causal": audit_flags()},
                        "evaluator": {"schema_version": EVALUATOR_RECORD_VERSION,
                                      "accepted": accepted_pose is not None,
                                      "pose": None if accepted_pose is None else accepted_pose.tolist(),
                                      "memory_write": memory_write,
                                      "oracle_decisions": evaluator_decisions,
                                      "wall_time_s": time.perf_counter() - frame_start}})
    audit = env.audit_trace()
    if not audit["passed"]:
        raise RuntimeError(f"{sequence}/{policy} causal audit failed: {audit}")
    return {"records": records, "audit": audit}


def summarize(records: Iterable[Mapping[str, Any]], gt: np.ndarray) -> Dict[str, Any]:
    rows = list(records)
    accepted = [x for x in rows if x["evaluator"]["accepted"]]
    correct = sum(success_pose(x["evaluator"]["pose"], gt[int(x["frame_index"])]) for x in accepted)
    geometry_calls = sum(
        1 for row in rows for action in row["online"].get("acquisitions", [])
        if action.get("kind") == "geometry")
    geometry_frames = sum(any(
        action.get("kind") == "geometry"
        for action in row["online"].get("acquisitions", [])) for row in rows)
    false_accepts = len(accepted) - correct
    return {"frames": len(rows), "accepted": len(accepted),
            "coverage": len(accepted) / len(rows) if rows else 0.0,
            "accepted_correct": int(correct),
            "accepted_accuracy": correct / len(accepted) if accepted else 0.0,
            "false_accepts": int(false_accepts),
            "false_accept_rate": false_accepts / len(accepted) if accepted else 0.0,
            "full_frame_success": correct / len(rows) if rows else 0.0,
            "mean_tool_actions": float(np.mean([x["online"]["tool_actions"] for x in rows])) if rows else 0.0,
            "geometry_calls": int(geometry_calls),
            "geometry_call_rate": geometry_calls / len(rows) if rows else 0.0,
            "geometry_frame_rate": geometry_frames / len(rows) if rows else 0.0,
            "mean_measured_tool_cost_s": float(np.mean([x["online"]["measured_tool_cost_s"] for x in rows])) if rows else 0.0,
            "budget_overshoot_frames": int(sum(x["online"]["budget_overshoot"] for x in rows)),
            "mean_wall_time_s": float(np.mean([x["evaluator"]["wall_time_s"] for x in rows])) if rows else 0.0}


def _oracle_expand(env: AgenticLocalizationEnv, groups: Sequence[Mapping[str, Any]],
                   remaining: Tuple[int, ...], geometry_used: bool, memory_used: bool,
                   depth: int, max_depth: int, cost: float, budget_s: float,
                   nodes: List[Dict[str, Any]], path_node_ids: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Enumerate feedback-gated *policy* decisions, including terminals.

    Verification remains a mandatory system operation immediately after a
    pose-producing tool.  In contrast, Accept/Reject/Defer are explicit
    policy choices at the resulting state.  This lets the Oracle label the
    central counterexample: a verifier-passed pose whose best continuation is
    stronger geometry rather than immediate Accept.
    """
    state_before = env._observation().as_dict()
    candidate = state_before.get("candidate") or {}
    checked = bool(candidate.get("pose_available")) and bool(candidate.get("verification"))
    verified = checked and bool(candidate.get("verified"))
    first_decision = int(state_before.get("state", {}).get("frame_action_count", 0)) == 0

    # Shared v2 action contract: all reference groups are legal Geometry
    # parameters, and Memory cannot propose a current-frame pose.
    allow_acquisition = depth < max_depth and cost < budget_s
    options = legal_actions(state_before, remaining, geometry_used=geometry_used,
                            allow_acquisition=allow_acquisition)

    # A deliberately constrained first state could exhaust acquisitions.  It
    # has no learnable terminal action under the deployment mask, so end it
    # safely without manufacturing a label that deployment cannot execute.
    if not options:
        leaf = env.fork(); leaf.step(LocalizationAction.defer())
        return [{"env": leaf, "pose": None, "cost": cost, "program": [],
                 "path_node_ids": list(path_node_ids)}]

    leaves: List[Dict[str, Any]] = []
    for option in options:
        kind, refs = str(option["kind"]), tuple(int(x) for x in option.get("references", ()))
        terminal = bool(option.get("is_terminal_action", False))
        next_remaining = tuple(x for x in remaining if tuple(groups[x]["reference_indices"]) != refs) \
            if kind == "localize" else remaining
        next_geometry = geometry_used or kind == "geometry"
        next_memory = memory_used
        branch = env.fork()
        node_id = f"{state_before['sequence']}:{state_before['frame_index']}:{len(nodes)}"
        action_cost = verification_cost = 0.0
        action_event: Mapping[str, Any] | None = None
        verification_event: Mapping[str, Any] | None = None
        program: List[Dict[str, Any]] = []
        pose = None
        if terminal:
            action = (LocalizationAction.accept() if kind == "accept" else
                      LocalizationAction.reject() if kind == "reject" else LocalizationAction.defer())
            if kind == "accept":
                pose = branch.state.current_candidate.pose.copy()
            _, _, _, info = branch.step(action)
            action_event = info["event"]
            program.append({"kind": kind, "references": [], "terminal": True})
        else:
            action = (LocalizationAction.localize(*refs) if kind == "localize" else
                      LocalizationAction.geometry(*refs))
            observation, action_cost, action_event = do_step(branch, action)
            branch_cost = cost + action_cost
            program.append({"kind": kind, "references": list(refs), "cost_s": action_cost})
            if observation.candidate and observation.candidate.get("pose_available") and branch_cost < budget_s:
                observation, verification_cost, verification_event = do_step(branch, LocalizationAction.verify())
                branch_cost += verification_cost
                program[-1]["verification_cost_s"] = verification_cost
        branch_cost = cost + action_cost + verification_cost
        candidate_pose = (None if branch.state.current_candidate is None or
                          branch.state.current_candidate.pose is None else
                          branch.state.current_candidate.pose.tolist())
        nodes.append({
            "node_id": node_id, "parent_path_node_ids": list(path_node_ids),
            "depth": depth + (0 if terminal else 1), "state": state_before,
            "action": {"kind": kind, "references": list(refs), "is_terminal_action": terminal},
            "action_event": action_event, "verification_event": verification_event,
            "candidate": branch._observation().candidate, "candidate_pose": candidate_pose,
            "action_cost_s": action_cost, "verification_cost_s": verification_cost,
            "prior_cost_s": cost, "cumulative_cost_s": branch_cost, "action_value": None,
        })
        next_path = path_node_ids + (node_id,)
        if terminal or branch_cost >= budget_s:
            if not terminal:
                branch.step(LocalizationAction.defer())
                program.append({"kind": "defer", "references": [], "terminal": True, "forced": True})
            leaves.append({"env": branch, "pose": pose, "cost": branch_cost,
                           "program": program, "path_node_ids": list(next_path)})
            continue
        for leaf in _oracle_expand(branch, groups, next_remaining, next_geometry, next_memory,
                                   depth + 1, max_depth, branch_cost, budget_s, nodes, next_path):
            leaf["program"] = program + leaf["program"]
            leaves.append(leaf)
    return leaves


def label_oracle_state(args, env: AgenticLocalizationEnv, groups: Sequence[Mapping[str, Any]],
                       remaining: Sequence[int], geometry_used: bool, memory_used: bool,
                       acquisitions_used: int, gt_pose: np.ndarray) -> Dict[str, Any]:
    """Evaluator-only two-step Q labels for one live policy-induced state.

    ``env`` is forked before expansion.  Thus this function can be called
    during DAgger collection without changing the agent trajectory or exposing
    GT to its action selector.  Returned observations are exactly the online
    state seen by the policy; only ``q_star`` is evaluator-only.
    """
    nodes: List[Dict[str, Any]] = []
    start_state = env._observation().as_dict()
    current_cost = float(env.state.frame_tool_cost_s) if env.state is not None else 0.0
    leaves = _oracle_expand(env.fork(), groups, tuple(remaining), geometry_used, memory_used,
                            acquisitions_used, args.oracle_depth, current_cost,
                            args.budget_ms / 1000.0, nodes, ())
    for leaf_id, leaf in enumerate(leaves):
        success = leaf["pose"] is not None and success_pose(leaf["pose"], gt_pose)
        leaf["leaf_id"] = leaf_id
        leaf["accepted"] = leaf["pose"] is not None
        leaf["success"] = bool(success)
        leaf["false_accept"] = bool(leaf["accepted"] and not success)
        leaf["terminal_utility"] = terminal_utility(leaf, args.utility)
    for node in nodes:
        descendants = [leaf for leaf in leaves if node["node_id"] in leaf["path_node_ids"]]
        if not descendants:
            continue
        successful = [leaf for leaf in descendants if leaf["success"]]
        best = max(descendants, key=lambda leaf: (leaf["terminal_utility"]["utility"], -float(leaf["cost"])))
        node["action_value"] = {
            "terminal_count": len(descendants), "any_success": bool(successful),
            "best_terminal_success": int(bool(successful)),
            "min_success_cost_s": None if not successful else min(float(leaf["cost"]) for leaf in successful),
            "min_terminal_cost_s": min(float(leaf["cost"]) for leaf in descendants),
            "any_false_accept": any(bool(leaf["false_accept"]) for leaf in descendants),
            "q_star": float(best["terminal_utility"]["utility"]),
            "best_leaf_id": int(best["leaf_id"]),
            "best_terminal_type": best["terminal_utility"]["terminal_type"],
            "best_terminal_metrics": {key: value for key, value in best["terminal_utility"].items()
                                      if key != "utility"},
        }
    root_nodes = [node for node in nodes if not node["parent_path_node_ids"]]
    if root_nodes:
        best_q = max(float(node["action_value"]["q_star"]) for node in root_nodes)
        for node in root_nodes:
            node["action_value"]["is_optimal_action"] = abs(
                float(node["action_value"]["q_star"]) - best_q) <= 1e-12
    return {"observation": start_state, "oracle_action_values": root_nodes,
            "oracle_terminal_programs": [{
                "leaf_id": leaf["leaf_id"], "accepted": leaf["accepted"],
                "success": leaf["success"], "false_accept": leaf["false_accept"],
                "cost_s": leaf["cost"], "terminal_utility": leaf["terminal_utility"],
                "program": leaf["program"], "path_node_ids": leaf["path_node_ids"]}
                for leaf in leaves]}


def run_oracle(args, sequence: str, start: int, end: int, gt: np.ndarray, *,
               backend=None) -> Dict[str, Any]:
    """GT-only utility selector over executable, feedback-gated programs.

    GT joins the trace only after every branch has executed.  The selected
    branch and per-action labels are therefore offline training/evaluation
    targets, never online policy inputs.
    """
    env = AgenticLocalizationEnv(make_backend(args, sequence) if backend is None else backend,
                                 max_actions_per_frame=3 * args.oracle_depth + 2,
                                 budget_s=args.budget_ms / 1000.0,
                                 allow_memory_query=False)
    observation = env.reset(sequence, start_frame=start)
    rows = []
    while not observation.done and observation.frame_index < end:
        frame_index = observation.frame_index
        groups = groups_from_observation(observation)
        wall_start = time.perf_counter()
        nodes: List[Dict[str, Any]] = []
        leaves = _oracle_expand(env, groups, tuple(range(len(groups))), False, False,
                                0, args.oracle_depth, 0.0, args.budget_ms / 1000.0,
                                nodes, ())
        # GT is deliberately read only at this terminal evaluator selection.
        for leaf_id, leaf in enumerate(leaves):
            success = leaf["pose"] is not None and success_pose(leaf["pose"], gt[frame_index])
            leaf["leaf_id"] = leaf_id
            leaf["accepted"] = leaf["pose"] is not None
            leaf["success"] = bool(success)
            leaf["false_accept"] = bool(leaf["accepted"] and not success)
            leaf["terminal_utility"] = terminal_utility(leaf, args.utility)
        for node in nodes:
            descendants = [leaf for leaf in leaves if node["node_id"] in leaf["path_node_ids"]]
            successful = [leaf for leaf in descendants if leaf["success"]]
            best = max(descendants, key=lambda leaf: (leaf["terminal_utility"]["utility"],
                                                       -float(leaf["cost"])))
            node["action_value"] = {
                "terminal_count": len(descendants),
                "any_success": bool(successful),
                "best_terminal_success": int(bool(successful)),
                "min_success_cost_s": (None if not successful else
                                       min(float(leaf["cost"]) for leaf in successful)),
                "min_terminal_cost_s": min(float(leaf["cost"]) for leaf in descendants),
                "any_false_accept": any(bool(leaf["false_accept"]) for leaf in descendants),
                "q_star": float(best["terminal_utility"]["utility"]),
                "best_leaf_id": int(best["leaf_id"]),
                "best_terminal_type": best["terminal_utility"]["terminal_type"],
                "best_terminal_metrics": {
                    key: value for key, value in best["terminal_utility"].items()
                    if key != "utility"
                },
            }
        best_q = max(float(node["action_value"]["q_star"]) for node in nodes) if nodes else float("-inf")
        for node in nodes:
            node["action_value"]["is_optimal_action"] = bool(
                abs(float(node["action_value"]["q_star"]) - best_q) <= 1e-12)
        chosen = max(leaves, key=lambda leaf: (leaf["terminal_utility"]["utility"],
                                               -float(leaf["cost"])))
        env, pose, cost, program = chosen["env"], chosen["pose"], chosen["cost"], chosen["program"]
        observation = env._observation()
        rows.append({"schema_version": ACTION_PROTOCOL_VERSION,
                     "sequence": sequence, "frame_index": frame_index, "policy": "finite_depth_oracle",
                     "online": {"schema_version": ONLINE_RECORD_VERSION,
                                "acquisitions": program,
                                "tool_actions": sum(item.get("kind") in {"localize", "geometry"}
                                                    for item in program),
                                "terminal_action": next((item["kind"] for item in reversed(program)
                                                         if item.get("terminal")), "defer"),
                                "measured_tool_cost_s": cost,
                                "budget_s": args.budget_ms / 1000.0,
                                "budget_overshoot": cost > args.budget_ms / 1000.0,
                                "causal": audit_flags()},
                     "evaluator": {"schema_version": EVALUATOR_RECORD_VERSION,
                                   "accepted": pose is not None,
                                   "pose": None if pose is None else pose.tolist(),
                                   "memory_write": bool(pose is not None and env.state.memory and
                                                        env.state.memory[-1].frame_index == frame_index),
                                   "oracle_depth": args.oracle_depth,
                                   "oracle_action_values": nodes,
                                   "oracle_terminal_programs": [{
                         "leaf_id": leaf["leaf_id"], "accepted": leaf["accepted"],
                         "success": leaf["success"], "false_accept": leaf["false_accept"],
                         "cost_s": leaf["cost"], "terminal_utility": leaf["terminal_utility"],
                         "program": leaf["program"],
                         "path_node_ids": leaf["path_node_ids"]} for leaf in leaves],
                                   "chosen_leaf_id": chosen["leaf_id"],
                                   "search_compute_s": time.perf_counter() - wall_start,
                                   "wall_time_s": time.perf_counter() - wall_start}})
    return {"records": rows, "audit": env.audit_trace(), "oracle_uses_gt": True,
            "utility": args.utility,
            "definition": "terminal-GT utility selector over finite-depth executable, verifier-gated action programs"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--backend", choices=("cpu_orb", "reloc3r"), default="cpu_orb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--episode-manifest", type=Path,
                        help="run the contiguous spans covering registered recovery episodes")
    parser.add_argument("--episode-ids", nargs="*", default=(),
                        help="optional episode subset for sharded execution")
    parser.add_argument("--warmup-frames", type=int, default=0,
                        help="causal prefix before the first episode, used to build policy-specific Memory")
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--oracle-depth", type=int,
                        help="must equal --max-acquisitions; omitted means the same two-step budget")
    parser.add_argument("--budget-ms", type=float, default=30000.0)
    parser.add_argument("--nfeatures", type=int, default=500)
    parser.add_argument("--verifier-config", type=Path,
                        help="frozen source-specific verifier calibration JSON")
    parser.add_argument("--utility-config", type=Path,
                        help="evaluator-only terminal utility JSON; defaults to the versioned built-in contract")
    parser.add_argument("--geometry-schedule-rate", type=float, default=0.35,
                        help="GT-free geometry rate for fixed_geometry, calibrated externally")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.8,
                        help="frozen verifier-risk threshold for uncertainty_trigger")
    parser.add_argument("--router-threshold", type=float, default=0.0,
                        help="frozen retrieval-only threshold for one_shot_router")
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        help="optional subset, useful for calibration collection")
    parser.add_argument("--learned-policy-checkpoint", type=Path,
                        help="Stage 3B or Stage 3C checkpoint; required when policy=learned")
    parser.add_argument("--learned-policy-family", choices=("stage3b", "stage3c_recovery", "stage3c_xgb_explicit", "stage3c_xgb_ranker", "stage3c6_utility"), default="stage3b",
                        help="Stage 3B scalar-Q, Stage 3C recovery/ranker, or utility-aligned Stage 3C.6 controller")
    parser.add_argument("--audit-policy-states", action="store_true",
                        help="attach evaluator-only live-cost Oracle labels to every learned-policy decision")
    parser.add_argument("--stage3d-preaction-features", action="store_true",
                        help="compute cheap ORB correspondence/appearance cues after localize and before Geometry")
    parser.add_argument("--learned-policy-device", default="cpu",
                        help="device for small learned policy, independent of the localization backend")
    parser.add_argument("--learned-decision", choices=("policy", "value", "terminal_value"), default="policy",
                        help="choose by supervised policy logits or auxiliary Q-value head")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="collect online candidate receipts without evaluator-only branching")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if not 0.0 <= args.geometry_schedule_rate <= 1.0:
        parser.error("--geometry-schedule-rate must be in [0, 1]")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.oracle_depth is None:
        args.oracle_depth = args.max_acquisitions
    elif args.oracle_depth != args.max_acquisitions:
        parser.error("--oracle-depth must equal --max-acquisitions for a budget-matched Oracle")
    args.utility = load_utility(args.utility_config)
    selected_policies = tuple(args.policies) if args.policies else POLICIES
    if "learned" in selected_policies and args.learned_policy_checkpoint is None:
        parser.error("--learned-policy-checkpoint is required when --policies includes learned")
    if "learned" in selected_policies:
        if args.learned_policy_family == "stage3c_recovery":
            learned_policy = RecoveryCostPolicy.load(args.learned_policy_checkpoint,
                                                      device=args.learned_policy_device)
        elif args.learned_policy_family == "stage3c6_utility":
            learned_policy = UtilityComponentPolicy.load(args.learned_policy_checkpoint)
        elif args.learned_policy_family in {"stage3c_xgb_explicit", "stage3c_xgb_ranker"}:
            learned_policy = XGBRecoveryPolicy.load(
                args.learned_policy_checkpoint,
                mode="ranker" if args.learned_policy_family.endswith("ranker") else "explicit",
                lambda_cost=float(args.utility["cost_penalty_per_s"]),
            )
        else:
            learned_policy = StructuredPolicy.load(args.learned_policy_checkpoint,
                                                   device=args.learned_policy_device,
                                                   decision=args.learned_decision)
    else:
        learned_policy = None
    episode_spans = (load_episode_spans(args.episode_manifest, args.warmup_frames, args.episode_ids)
                     if args.episode_manifest is not None else {})
    if episode_spans:
        episode_spans = {sequence: span for sequence, span in episode_spans.items()
                         if sequence in set(args.sequences)}
        if not episode_spans:
            parser.error("selected episodes do not belong to --sequences")
    output: Dict[str, Any] = {"schema_version": ACTION_PROTOCOL_VERSION,
                              "config": vars(args), "policies": {}, "oracle": {},
                              "episode_spans": episode_spans,
                              "online_audit": audit_flags(),
                              "budget_definition": "sum of backend-reported elapsed_s; a single action may overshoot before subsequent actions are blocked",
                              "oracle_scope": "per-frame finite-depth executable action tree; not episode-optimal planning"}
    for sequence in args.sequences:
        gt = load_query_gt(args.data_root, args.gt_root, sequence)
        if episode_spans:
            if sequence not in episode_spans:
                continue
            start, requested_end = episode_spans[sequence]
            end = min(requested_end, len(gt))
        else:
            start = args.start
            end = min(args.start + args.max_frames, len(gt))
        backend = make_backend(args, sequence)
        for policy in selected_policies:
            labeler = None
            # Labels are evaluator-only, but must cover the states *each*
            # deployed controller actually visits.  Restricting this to the
            # learned controller would train/evaluate against Oracle-only
            # state distributions and violate the counterfactual-audit
            # contract.
            if args.audit_policy_states:
                def labeler(env, groups, remaining, geometry_used, memory_used, acquisitions_used, *, _gt=gt):
                    return label_oracle_state(args, env, groups, remaining, geometry_used, memory_used,
                                              acquisitions_used, _gt[env.state.frame_index])
            run = run_policy(args, sequence, policy, start, end, backend=backend,
                             learned_policy=learned_policy, oracle_labeler=labeler)
            output["policies"].setdefault(policy, {})[sequence] = {"summary": summarize(run["records"], gt), "audit": run["audit"]}
            path = args.output_dir / "traces" / f"{sequence}_{policy}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(x) + "\n" for x in run["records"]), encoding="utf-8")
        if not args.skip_oracle:
            oracle = run_oracle(args, sequence, start, end, gt, backend=backend)
            output["oracle"][sequence] = {"summary": summarize(oracle["records"], gt), **{k: v for k, v in oracle.items() if k != "records"}}
            path = args.output_dir / "traces" / f"{sequence}_oracle.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in oracle["records"]), encoding="utf-8")
            oracle_success = output["oracle"][sequence]["summary"]["full_frame_success"]
            for policy in selected_policies:
                summary = output["policies"][policy][sequence]["summary"]
                summary["oracle_gap_full_frame_success"] = oracle_success - summary["full_frame_success"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
