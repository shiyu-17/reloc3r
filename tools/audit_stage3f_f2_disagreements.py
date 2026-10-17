#!/usr/bin/env python3
"""Explain paired F2-vs-feedback-rule outcome disagreements.

This is an offline evaluator only: it joins GT success and two-step Oracle
labels after completed causal rollouts have been written.  It never feeds an
Oracle label back into a policy.  The audit separates outcome disagreement,
the first action-program divergence, pre-decision state disagreement, and the
post-hoc Geometry opportunity label so that a policy failure is not casually
misattributed to the Geometry head.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses,
)


FrameKey = Tuple[str, int]
TERMINAL = {"accept", "reject", "defer"}

# Kept local to make this evaluator independent of policy-training imports.
DEFAULT_BLOCKS = {
    "100_119": (
        "outputs/lilocbench/stage3d_v2_fresh_collect_r1/f2",
        "outputs/lilocbench/stage3d_v2_feedback_rule_fresh_baseline_r1",
    ),
    "120_139": (
        "outputs/lilocbench/stage3d_geometry_advantage_fresh_base_r1/f2",
        "outputs/lilocbench/stage3d_geometry_advantage_fresh_eval_r1/feedback_rule",
    ),
    "140_154": (
        "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/f2_base_15/f2",
        "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/feedback_rule_15",
    ),
}


def _load_query_gt(data_root: Path, gt_root: Path, sequence: str):
    _, _, query_times, _, query_root = read_manifest(data_root, sequence)
    poses, _ = tum_camera_poses(
        gt_root / f"{sequence}_gt_poses.txt", query_times,
        base_to_front_rgb_optical(query_root), 0.1)
    return poses


def _load_rollouts(root: Path) -> Dict[FrameKey, Dict[str, Any]]:
    rows: Dict[FrameKey, Dict[str, Any]] = {}
    paths = sorted(root.glob("*/*_rollout.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no collector rollout files below {root}")
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sequence"]), int(row["frame_index"]))
            if key in rows:
                raise ValueError(f"duplicate frame {key} at {path}:{line_number}")
            causal = row.get("causal", {})
            if any(bool(causal.get(name, False)) for name in
                   ("query_gt_read", "uses_future", "selection_uses_gt")):
                raise ValueError(f"online causal boundary violated at {key}")
            rows[key] = row
    return rows


def _action_key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind", "unknown")), tuple(int(x) for x in action.get("references", ()))


def _terminal_name(row: Mapping[str, Any]) -> str:
    name = str(row.get("terminal_action", "unknown"))
    return "accept" if name in {"accept", "accept_legacy"} else name


def _program(row: Mapping[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    """Return non-terminal policy actions, normalising rule's legacy accept."""
    return [_action_key(item.get("action", {})) for item in row.get("decisions", [])
            if str(item.get("action", {}).get("kind")) not in TERMINAL]


def _first_decision(row: Mapping[str, Any], index: int) -> Mapping[str, Any] | None:
    decisions = [item for item in row.get("decisions", [])
                 if str(item.get("action", {}).get("kind")) not in TERMINAL]
    return decisions[index] if index < len(decisions) else None


def _state_signature(decision: Mapping[str, Any] | None) -> Dict[str, Any]:
    observation = {} if decision is None else dict(decision.get("state") or {})
    state = dict(observation.get("state") or observation)
    return {key: state.get(key) for key in
            ("anchor_age", "reject_streak", "memory_size", "accepted_pose_available",
             "frame_action_count", "remaining_budget_s")}


def _first_divergence(f2: Mapping[str, Any], rule: Mapping[str, Any]) -> Dict[str, Any]:
    left, right = _program(f2), _program(rule)
    shared = min(len(left), len(right))
    for index in range(shared):
        if left[index] != right[index]:
            return {"kind": "acquisition_action", "decision_index": index,
                    "f2_action": left[index], "rule_action": right[index],
                    "f2_decision": _first_decision(f2, index),
                    "rule_decision": _first_decision(rule, index)}
    if len(left) != len(right):
        return {"kind": "acquisition_length", "decision_index": shared,
                "f2_action": left[shared] if len(left) > shared else None,
                "rule_action": right[shared] if len(right) > shared else None,
                "f2_decision": _first_decision(f2, shared),
                "rule_decision": _first_decision(rule, shared)}
    if _terminal_name(f2) != _terminal_name(rule):
        return {"kind": "terminal", "decision_index": shared,
                "f2_action": (_terminal_name(f2), ()), "rule_action": (_terminal_name(rule), ()),
                "f2_decision": None, "rule_decision": None}
    return {"kind": "none", "decision_index": None, "f2_action": None, "rule_action": None,
            "f2_decision": None, "rule_decision": None}


def _oracle_values(decision: Mapping[str, Any] | None) -> Dict[tuple[str, tuple[int, ...]], float]:
    if decision is None:
        return {}
    nodes = (decision.get("oracle_label") or {}).get("oracle_action_values") or []
    return {_action_key(node.get("action", {})): float(node["action_value"]["q_star"])
            for node in nodes if "action_value" in node and "q_star" in node["action_value"]}


def _geometry_context(decision: Mapping[str, Any] | None,
                      selected: tuple[str, tuple[int, ...]] | None) -> Dict[str, Any]:
    values = _oracle_values(decision)
    geometry = {key: value for key, value in values.items() if key[0] == "geometry"}
    acquisition = {key: value for key, value in values.items() if key[0] in {"localize", "geometry", "memory"}}
    if not geometry or not acquisition:
        return {"available": False}
    best_geometry = max(geometry, key=geometry.get)
    best_acquisition = max(acquisition, key=acquisition.get)
    return {
        "available": True,
        "selected_value": None if selected not in values else values[selected],
        "best_geometry_action": best_geometry,
        "best_geometry_value": geometry[best_geometry],
        "best_acquisition_action": best_acquisition,
        "best_acquisition_value": acquisition[best_acquisition],
        "geometry_best_acquisition": best_geometry == best_acquisition,
    }


def _oracle_decision_audit(decision: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare a chosen action to post-hoc Oracle values at the same state."""
    action = _action_key(decision.get("action", {}))
    values = _oracle_values(decision)
    if action not in values or not values:
        return {"available": False, "action": action}
    best_action = max(values, key=values.get)
    accept = {key: value for key, value in values.items() if key[0] == "accept"}
    best_accept = None if not accept else max(accept, key=accept.get)
    best_nonterminal = {key: value for key, value in values.items() if key[0] != "accept"}
    model = dict(decision.get("model_decision") or {})
    oracle_accept_beats_selected = (best_accept is not None and action[0] != "accept" and
                                    accept[best_accept] > values[action] + 1e-12)
    return {
        "available": True, "action": action, "selected_q_star": values[action],
        "best_action": best_action, "best_q_star": values[best_action],
        "oracle_regret": values[best_action] - values[action],
        "best_accept_action": best_accept,
        "best_accept_q_star": None if best_accept is None else accept[best_accept],
        "oracle_accept_beats_selected": oracle_accept_beats_selected,
        "oracle_accept_margin_over_selected": (None if best_accept is None else accept[best_accept] - values[action]),
        "policy_terminal_eligible": model.get("terminal_eligible"),
        "policy_best_acquisition_utility": model.get("best_acquisition_utility"),
        "policy_best_terminal_utility": model.get("best_terminal_utility"),
    }


def _decision_audits(row: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [_oracle_decision_audit(decision) for decision in row.get("decisions", [])]


def _missed_terminal_accept(row: Mapping[str, Any]) -> bool:
    return any(item.get("oracle_accept_beats_selected", False) for item in _decision_audits(row))


def _root_cause(divergence: Mapping[str, Any], outcome: str) -> str:
    """A conservative, post-hoc taxonomy; not a claim of causal mechanism."""
    kind = str(divergence["kind"])
    if kind == "terminal":
        return "terminal_program_difference"
    if kind == "acquisition_length":
        f2_action, rule_action = divergence["f2_action"], divergence["rule_action"]
        f2_context = _geometry_context(divergence["f2_decision"], f2_action)
        if outcome == "rule_only_success" and f2_action and f2_action[0] == "geometry":
            return "f2_extra_geometry_without_oracle_advantage" if not f2_context.get("geometry_best_acquisition") else "f2_extra_geometry_with_oracle_advantage"
        if outcome == "f2_only_success" and f2_action and f2_action[0] == "geometry":
            return "f2_extra_geometry_observed_recovery"
        return "acquisition_length_difference"
    if kind != "acquisition_action":
        return "no_observed_action_program_difference" if kind == "none" else "acquisition_length_difference"
    f2_action, rule_action = divergence["f2_action"], divergence["rule_action"]
    kinds = {f2_action[0], rule_action[0]}
    f2_context = _geometry_context(divergence["f2_decision"], f2_action)
    rule_context = _geometry_context(divergence["rule_decision"], rule_action)
    geometry_context = f2_context if f2_context.get("available") else rule_context
    # The first switch may be memory-vs-localize while the F2 Oracle label
    # already says Geometry is best.  Preserve that signal instead of hiding
    # it behind the superficial cheap-action difference.
    if outcome == "rule_only_success" and f2_action[0] != "geometry" and f2_context.get("geometry_best_acquisition"):
        return "f2_missed_oracle_geometry_opportunity"
    if outcome == "f2_only_success" and rule_action[0] != "geometry" and rule_context.get("geometry_best_acquisition"):
        return "rule_missed_oracle_geometry_opportunity"
    if "geometry" in kinds and len(kinds) == 2:
        if geometry_context.get("geometry_best_acquisition"):
            if outcome == "rule_only_success" and rule_action[0] == "geometry":
                return "f2_missed_oracle_geometry_opportunity"
            if outcome == "f2_only_success" and f2_action[0] == "geometry":
                return "rule_missed_oracle_geometry_opportunity"
            return "geometry_switch_with_oracle_opportunity"
        return "geometry_switch_without_oracle_opportunity"
    if f2_action[0] == rule_action[0] and f2_action[1] != rule_action[1]:
        return "reference_group_switch"
    return "non_geometry_acquisition_switch"


def _success(row: Mapping[str, Any], gt: Sequence[Any], frame_index: int) -> bool:
    pose = row.get("pose")
    if not bool(row.get("accepted")) or pose is None:
        return False
    translation, rotation = pose_error(np.asarray(pose, dtype=np.float64), gt[frame_index])
    return bool(translation <= 0.25 and rotation <= 5.0)


def _outcome(f2_success: bool, rule_success: bool) -> str:
    if f2_success and rule_success:
        return "both_success"
    if f2_success:
        return "f2_only_success"
    if rule_success:
        return "rule_only_success"
    return "both_failure"


def audit_pairs(f2_rows: Mapping[FrameKey, Mapping[str, Any]],
                rule_rows: Mapping[FrameKey, Mapping[str, Any]], *,
                data_root: Path, gt_root: Path, block: str) -> list[Dict[str, Any]]:
    if set(f2_rows) != set(rule_rows):
        raise ValueError(f"unaligned block {block}: F2-only={set(f2_rows)-set(rule_rows)}, "
                         f"rule-only={set(rule_rows)-set(f2_rows)}")
    ground_truth = {sequence: _load_query_gt(data_root, gt_root, sequence)
                    for sequence in sorted({key[0] for key in f2_rows})}
    rows = []
    for key in sorted(f2_rows):
        sequence, frame_index = key
        f2, rule = f2_rows[key], rule_rows[key]
        f2_success, rule_success = (_success(f2, ground_truth[sequence], frame_index),
                                    _success(rule, ground_truth[sequence], frame_index))
        outcome = _outcome(f2_success, rule_success)
        divergence = _first_divergence(f2, rule)
        f2_state = _state_signature(divergence["f2_decision"])
        rule_state = _state_signature(divergence["rule_decision"])
        changed_state = [key for key in f2_state if f2_state[key] != rule_state[key]]
        f2_action = divergence["f2_action"]
        rule_action = divergence["rule_action"]
        taxonomy = _root_cause(divergence, outcome)
        # Give a terminal mistake precedence for outcome disagreement.  It is
        # more diagnostic than the preceding acquisition switch when the
        # Oracle says an already-verified candidate should have been accepted.
        if outcome == "rule_only_success" and _missed_terminal_accept(f2):
            taxonomy = "f2_missed_oracle_terminal_accept"
        rows.append({
            "block": block, "sequence": sequence, "frame_index": frame_index,
            "outcome": outcome, "f2_accepted": bool(f2.get("accepted")),
            "rule_accepted": bool(rule.get("accepted")),
            "f2_terminal_action": _terminal_name(f2), "rule_terminal_action": _terminal_name(rule),
            "f2_geometry_called": any(item.get("kind") == "geometry" for item in f2.get("acquisitions", [])),
            "rule_geometry_called": any(item.get("kind") == "geometry" for item in rule.get("acquisitions", [])),
            "first_divergence": {key: divergence[key] for key in ("kind", "decision_index", "f2_action", "rule_action")},
            "predecision_state": {"f2": f2_state, "feedback_rule": rule_state,
                                  "different_fields": changed_state},
            "f2_geometry_context": _geometry_context(divergence["f2_decision"], f2_action),
            "rule_geometry_context": _geometry_context(divergence["rule_decision"], rule_action),
            "taxonomy": taxonomy,
            "f2_oracle_decision_audit": _decision_audits(f2),
            "rule_oracle_decision_audit": _decision_audits(rule),
            "offline_oracle_join": True,
        })
    return rows


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def report_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    disagreements = [row for row in rows if row["outcome"] in {"f2_only_success", "rule_only_success"}]
    f2_decisions = [item for row in rows for item in row.get("f2_oracle_decision_audit", [])
                    if item.get("available")]
    return {
        "schema_version": 1,
        "purpose": "frozen F2-vs-feedback-rule paired disagreement audit; no policy tuning",
        "online_boundary": "GT success and Oracle labels are joined only after completed online rollouts",
        "frames": len(rows), "outcome_counts": _count(rows, "outcome"),
        "disagreement_frames": len(disagreements),
        "disagreement_outcome_counts": _count(disagreements, "outcome"),
        "disagreement_taxonomy": _count(disagreements, "taxonomy"),
        "all_frame_divergence": _count(rows, "taxonomy"),
        "f2_terminal_oracle_audit": {
            "decisions_with_oracle": len(f2_decisions),
            "oracle_accept_beats_selected": sum(
                bool(item.get("oracle_accept_beats_selected")) for item in f2_decisions),
            "eligible_but_missed_accept": sum(
                bool(item.get("oracle_accept_beats_selected")) and
                item.get("policy_terminal_eligible") is True for item in f2_decisions),
        },
        "rows": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows: list[Dict[str, Any]] = []
    for block, (f2_root, rule_root) in DEFAULT_BLOCKS.items():
        all_rows.extend(audit_pairs(_load_rollouts(Path(f2_root)), _load_rollouts(Path(rule_root)),
                                    data_root=args.data_root, gt_root=args.gt_root, block=block))
    report = report_from_rows(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
