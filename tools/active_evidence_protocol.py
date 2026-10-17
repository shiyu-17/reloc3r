"""Versioned deployable action contract for active-evidence localization.

Rules, learned policies, and the finite-depth Oracle must call
``legal_actions`` with the same context.  Evaluator labels never enter this
module or its returned actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


ACTION_PROTOCOL_VERSION = "active_evidence_action_v2"
ONLINE_RECORD_VERSION = "active_evidence_online_v2"
EVALUATOR_RECORD_VERSION = "active_evidence_evaluator_v2"


@dataclass(frozen=True)
class ActionSpace:
    """First paper protocol: Memory remains state, not a pose-proposal tool."""

    allow_memory_query: bool = False
    auto_heldout_verify: bool = True


DEFAULT_ACTION_SPACE = ActionSpace()


def observation_dict(observation: Any) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    as_dict = getattr(observation, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError("observation must be a mapping or expose as_dict()")


def reference_groups(observation: Any) -> List[Mapping[str, Any]]:
    groups = observation_dict(observation).get("frame", {}).get("reference_groups", [])
    return [group for group in groups if len(group.get("reference_indices", ())) >= 2]


def legal_actions(observation: Any, remaining_group_indices: Sequence[int], *,
                  geometry_used: bool, allow_acquisition: bool = True,
                  action_space: ActionSpace = DEFAULT_ACTION_SPACE) -> List[Dict[str, Any]]:
    """Return every policy-visible legal action in a stable order.

    Strong geometry is available for every reference group.  That removes the
    prior mismatch where a feedback rule could reuse a candidate group while
    the learned policy and Oracle only had group zero.
    """
    obs = observation_dict(observation)
    groups = reference_groups(obs)
    actions: List[Dict[str, Any]] = []
    if allow_acquisition:
        for index in remaining_group_indices:
            if 0 <= int(index) < len(groups):
                actions.append({"kind": "localize", "references": list(groups[int(index)]["reference_indices"]),
                                "reference_group": int(index), "is_terminal_action": False})
        if not geometry_used:
            for index, group in enumerate(groups):
                actions.append({"kind": "geometry", "references": list(group["reference_indices"]),
                                "reference_group": int(index), "is_terminal_action": False})
        if action_space.allow_memory_query and obs.get("memory"):
            actions.append({"kind": "memory", "references": [], "is_terminal_action": False})
    candidate = obs.get("candidate") or {}
    verified = bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
    if verified:
        actions.append({"kind": "accept", "references": [], "is_terminal_action": True})
    first_decision = int(obs.get("state", {}).get("frame_action_count", 0)) == 0
    if not first_decision or not actions:
        actions.append({"kind": "defer", "references": [], "is_terminal_action": True})
    return actions


def action_key(action: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(action.get("kind", "")), tuple(int(x) for x in action.get("references", ()))


def is_deployment_legal(observation: Any, action: Mapping[str, Any]) -> bool:
    """Apply the v2 first-decision safety mask to stored labels."""
    kind = str(action.get("kind", ""))
    if kind == "memory":
        return False
    if kind != "defer":
        return True
    return int(observation_dict(observation).get("state", {}).get("frame_action_count", 0)) > 0
