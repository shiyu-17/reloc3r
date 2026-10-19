"""Fixed Reloc3r-first loop for the retained LILocBench v1 baseline.

Policies see only observations and legal actions. Backends own image loading,
pose estimation, verification, and tool receipts. Ground truth is not read here.
"""
from __future__ import annotations

import time
from typing import Any

from tools.active_evidence_protocol import (
    LiveCapabilities, UNIFIED_LIVE_ACTION_PROTOCOL_VERSION, unified_live_legal_actions,
)
from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction


def run_fixed_order(env: AgenticLocalizationEnv, sequence: str, start: int, count: int,
                    capabilities: LiveCapabilities, *,
                    geometry_on_failure: bool = False) -> dict[str, Any]:
    """Run A/B/C Reloc3r, then optional strong geometry, on consecutive frames."""
    if count < 1 or start < 0:
        raise ValueError("start must be non-negative and count positive")
    if geometry_on_failure and not capabilities.strong_geometry:
        raise ValueError("strong geometry was not validated for this backend")
    observation = env.reset(sequence, start_frame=start)
    end = min(observation.frame_count, start + count)
    decisions: list[dict[str, Any]] = []
    while not observation.done and observation.frame_index < end:
        frame_index = observation.frame_index
        frame_started = time.perf_counter()
        geometry_used = False

        def take(action: LocalizationAction, remaining=()) -> None:
            nonlocal observation, geometry_used
            options = unified_live_legal_actions(observation, capabilities, remaining,
                                                  geometry_used=geometry_used)
            key = {"kind": action.kind}
            if action.kind in ("localize_with_references", "run_strong_geometry"):
                key["reference_indices"] = list(action.reference_indices)
            elif action.kind == "select_candidate":
                key["candidate_index"] = action.candidate_index
            if not any(all(option.get(field) == value for field, value in key.items())
                       for option in options):
                raise RuntimeError(f"action {key} violates {UNIFIED_LIVE_ACTION_PROTOCOL_VERSION}")
            observation = env.step(action)[0]
            geometry_used |= action.kind == "run_strong_geometry"

        groups = observation.frame.get("reference_groups", [])
        query_frame_id = int(observation.frame.get("query_frame_id", frame_index))
        attempted: list[dict[str, Any]] = []
        accepted_index = None

        def try_pose(action: LocalizationAction, remaining: range, name: str) -> bool:
            nonlocal accepted_index
            take(action, remaining)
            candidate_index = len(observation.candidates) - 1
            verification = {"accepted": False, "evidence": {"status": "unavailable"}}
            if (observation.candidate and observation.candidate["pose_available"] and
                    observation.state.get("remaining_budget_s", 1.) > 0. and
                    observation.state["frame_action_count"] < env.max_actions_per_frame - 1):
                take(LocalizationAction.verify(), remaining)
                verification = observation.last_event
            accepted = bool(verification["accepted"])
            attempted.append({"group": name, "candidate_index": candidate_index,
                              "accepted_by_verifier": accepted,
                              "verification_status": verification["evidence"].get(
                                  "status", "supported" if accepted else "rejected")})
            if accepted:
                accepted_index = candidate_index
            return accepted

        for index, group in enumerate(groups):
            if (observation.state.get("remaining_budget_s", 1.) <= 0. or
                    observation.state["frame_action_count"] >= env.max_actions_per_frame - 2):
                break
            refs = tuple(int(x) for x in group["reference_indices"])
            if try_pose(LocalizationAction.localize(*refs), range(index, len(groups)),
                        str(group.get("name", f"group_{index}"))):
                break

        if (accepted_index is None and geometry_on_failure and groups and
                observation.state.get("remaining_budget_s", 1.) > 0. and
                observation.state["frame_action_count"] < env.max_actions_per_frame - 2):
            refs = tuple(int(x) for x in groups[0]["reference_indices"])
            try_pose(LocalizationAction.geometry(*refs), range(0), "strong_geometry")

        if accepted_index is None:
            take(LocalizationAction.defer())
            terminal = "defer"
        else:
            if observation.candidate["candidate_index"] != accepted_index:
                take(LocalizationAction.select_candidate(accepted_index))
            take(LocalizationAction.accept())
            terminal = "accept"
        decisions.append({"query_index": frame_index, "query_frame_id": query_frame_id,
                          "groups_attempted": attempted,
                          "accepted_candidate_index": accepted_index,
                          "terminal": terminal,
                          "frame_wall_s_excluding_model_startup": time.perf_counter() - frame_started})
    return {"schema_version": "lilocbench_v1_reset_step",
            "action_protocol_version": UNIFIED_LIVE_ACTION_PROTOCOL_VERSION,
            "reference_retrieval_provenance": capabilities.reference_retrieval,
            "strong_geometry_available": capabilities.strong_geometry,
            "strong_geometry_enabled": bool(geometry_on_failure),
            "max_actions_per_frame": env.max_actions_per_frame,
            "query_index_start": start, "query_count_requested": count,
            "query_count_executed": len(decisions),
            "decisions": decisions, "trace": env.trace,
            "environment_audit": env.audit_trace(),
            "evaluator_gt_loaded": False}
