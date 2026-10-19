"""Legal actions for the retained LILocBench v1 localization loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UNIFIED_LIVE_ACTION_PROTOCOL_VERSION = "lilocbench_v1_precomputed_references"


@dataclass(frozen=True)
class LiveCapabilities:
    reference_retrieval: str = "precomputed"
    strong_geometry: bool = True

    def __post_init__(self) -> None:
        if self.reference_retrieval != "precomputed":
            raise ValueError("retained v1 baseline requires precomputed references")


def observation_dict(observation: Any) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    as_dict = getattr(observation, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError("observation must be a mapping or expose as_dict()")


def unified_live_legal_actions(observation: Any, capabilities: LiveCapabilities,
                               remaining_group_indices: Sequence[int] = (), *,
                               geometry_used: bool = False) -> list[dict[str, Any]]:
    """Return v1 choices from policy-visible state; never use query GT."""
    obs = observation_dict(observation)
    if obs.get("done"):
        return []
    state = obs.get("state", {})
    groups = obs.get("frame", {}).get("reference_groups", [])
    actions: list[dict[str, Any]] = []
    count = int(state.get("frame_action_count", 0))
    room = count < int(state.get("max_actions_per_frame", 1000000)) - 1
    budget = state.get("remaining_budget_s")
    if room and (budget is None or float(budget) > 0.):
        for index in remaining_group_indices:
            if 0 <= int(index) < len(groups):
                refs = list(groups[int(index)]["reference_indices"])
                if refs:
                    actions.append({"kind": "localize_with_references",
                                    "reference_indices": refs, "reference_group": int(index)})
        if capabilities.strong_geometry and not geometry_used:
            for index, group in enumerate(groups):
                refs = list(group["reference_indices"])
                if len(refs) >= 2:
                    actions.append({"kind": "run_strong_geometry",
                                    "reference_indices": refs, "reference_group": index})
        candidate = obs.get("candidate") or {}
        if candidate.get("pose_available") and not candidate.get("verification"):
            actions.append({"kind": "verify_candidate"})
    active = obs.get("candidate") or {}
    for candidate in obs.get("candidates", []):
        if (candidate.get("pose_available") and candidate.get("verified") and
                candidate.get("candidate_index") != active.get("candidate_index")):
            actions.append({"kind": "select_candidate",
                            "candidate_index": int(candidate["candidate_index"])})
    if active.get("pose_available") and active.get("verified"):
        actions.append({"kind": "accept_candidate"})
    if count > 0 or not actions:
        actions.append({"kind": "defer_frame"})
    return actions
