#!/usr/bin/env python3
"""Causal reset/step environment for evidence-seeking visual localization.

This module is intentionally independent of a policy implementation.  A
policy can be a rule, a small network, or an LLM, but all policies use the
same state machine:

    reset -> acquire evidence -> verify -> accept/reject/defer -> next frame

The environment does not expose query ground truth.  A localization backend
must provide an execution receipt for every reference-selection action.  This
prevents a frozen candidate cache from being silently presented as an online
reference-conditioned action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


CAUSAL_FLAGS = ("query_gt_read", "uses_future", "selection_uses_gt")
FORBIDDEN_EVALUATOR_FIELDS = {
    "query_gt", "query_gt_pose", "ground_truth", "ground_truth_pose",
    "gt_pose", "translation_error", "rotation_error", "success",
    "oracle", "outcome", "future_frame", "future_frame_read",
}
ACTION_KINDS = {
    "localize_with_references",
    "run_strong_geometry",
    "retrieve_memory",
    "verify_candidate",
    "accept_candidate",
    "reject_candidate",
    "defer_frame",
}


class CausalViolation(RuntimeError):
    """Raised when a backend result violates the online information boundary."""


class InvalidAction(ValueError):
    """Raised when an action is invalid for the current environment state."""


def _as_pose(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 matrix")
    return pose


def _causal_audit(value: Mapping[str, Any], *, require_keys: bool = True) -> Dict[str, bool]:
    """Validate and normalize audit flags supplied by a backend."""
    audit = value.get("causal", value.get("audit", {}))
    if not isinstance(audit, Mapping):
        raise CausalViolation("backend result has no causal audit mapping")
    if require_keys and any(key not in audit for key in CAUSAL_FLAGS):
        missing = [key for key in CAUSAL_FLAGS if key not in audit]
        raise CausalViolation("backend result is missing causal audit fields: " + ", ".join(missing))
    flags = {key: bool(audit.get(key, False)) for key in CAUSAL_FLAGS}
    if any(flags.values()):
        raise CausalViolation("online backend used forbidden information: " + repr(flags))
    return flags


def _assert_no_evaluator_fields(value: Any, path: str = "result") -> None:
    """Reject accidental evaluator/outcome data crossing the online boundary."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_EVALUATOR_FIELDS:
                raise CausalViolation(f"forbidden evaluator field at {path}.{key}")
            _assert_no_evaluator_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_evaluator_fields(child, f"{path}[{index}]")


def _copy_jsonable(value: Any) -> Any:
    """Copy arrays into ordinary lists for policy observations and traces."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _copy_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


@dataclass(frozen=True)
class LocalizationAction:
    """One policy action.  Acquisition actions do not advance the frame."""

    kind: str
    reference_indices: Tuple[int, ...] = ()
    memory_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise InvalidAction("unknown action kind: " + self.kind)
        refs = tuple(int(x) for x in self.reference_indices)
        memories = tuple(int(x) for x in self.memory_ids)
        if len(set(refs)) != len(refs) or any(x < 0 for x in refs):
            raise InvalidAction("reference_indices must be unique non-negative integers")
        if len(set(memories)) != len(memories) or any(x < 0 for x in memories):
            raise InvalidAction("memory_ids must be unique non-negative integers")
        object.__setattr__(self, "reference_indices", refs)
        object.__setattr__(self, "memory_ids", memories)

    @classmethod
    def localize(cls, *reference_indices: int) -> "LocalizationAction":
        return cls("localize_with_references", tuple(reference_indices))

    @classmethod
    def geometry(cls, *reference_indices: int) -> "LocalizationAction":
        """Run an explicitly selected, online strong-geometry tool."""
        return cls("run_strong_geometry", tuple(reference_indices))

    @classmethod
    def retrieve(cls, *memory_ids: int) -> "LocalizationAction":
        return cls("retrieve_memory", memory_ids=tuple(memory_ids))

    @classmethod
    def verify(cls) -> "LocalizationAction":
        return cls("verify_candidate")

    @classmethod
    def accept(cls) -> "LocalizationAction":
        return cls("accept_candidate")

    @classmethod
    def reject(cls) -> "LocalizationAction":
        return cls("reject_candidate")

    @classmethod
    def defer(cls) -> "LocalizationAction":
        return cls("defer_frame")


@dataclass
class Candidate:
    pose: Optional[np.ndarray]
    source: str
    frame_index: int
    reference_indices: Tuple[int, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    memory_id: int
    frame_index: int
    pose: np.ndarray
    source: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationState:
    sequence: str
    frame_index: int
    frame_count: int
    reject_streak: int = 0
    accepted_pose: Optional[np.ndarray] = None
    accepted_frame_index: Optional[int] = None
    current_candidate: Optional[Candidate] = None
    memory: List[MemoryRecord] = field(default_factory=list)
    action_count: int = 0
    frame_action_count: int = 0
    frame_tool_cost_s: float = 0.0
    done: bool = False


@dataclass
class Observation:
    """Policy-visible observation.  It contains no evaluator outcome."""

    sequence: str
    frame_index: int
    frame_count: int
    frame: Dict[str, Any]
    state: Dict[str, Any]
    candidate: Optional[Dict[str, Any]]
    memory: List[Dict[str, Any]]
    last_event: Optional[Dict[str, Any]]
    done: bool

    def as_dict(self) -> Dict[str, Any]:
        return _copy_jsonable({
            "sequence": self.sequence,
            "frame_index": self.frame_index,
            "frame_count": self.frame_count,
            "frame": self.frame,
            "state": self.state,
            "candidate": self.candidate,
            "memory": self.memory,
            "last_event": self.last_event,
            "done": self.done,
            "causal": dict((key, False) for key in CAUSAL_FLAGS),
        })


class LocalizationBackend(Protocol):
    """Online backend contract used by :class:`AgenticLocalizationEnv`."""

    def num_frames(self, sequence: str) -> int:
        ...

    def frame_observation(self, sequence: str, frame_index: int,
                          state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def localize(self, sequence: str, frame_index: int,
                 reference_indices: Sequence[int],
                 state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run reference-conditioned matching and pose estimation now.

        The return mapping must include ``execution.executed=True``,
        ``execution.cache_hit=False``, and all three false causal flags.
        """
        ...

    def strong_geometry(self, sequence: str, frame_index: int,
                        reference_indices: Sequence[int],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run a higher-cost correspondence/geometry/PnP action online."""
        ...

    def retrieve_memory(self, sequence: str, frame_index: int,
                        memory: Sequence[Mapping[str, Any]],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def verify(self, sequence: str, frame_index: int, candidate: Candidate,
               state: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class CallableLocalizationBackend:
    """Adapter for existing model/tool code.

    ``localize_fn`` is deliberately a callable rather than a cache reader. It
    should load the current query image, run the selected reference pairs,
    aggregate the resulting poses, and return a pose plus an execution receipt.
    This keeps model-specific code out of the causal environment while making
    the online boundary explicit.
    """

    def __init__(self, frame_counts: Mapping[str, int],
                 frame_fn: Callable[[str, int, Mapping[str, Any]], Mapping[str, Any]],
                 localize_fn: Callable[[str, int, Sequence[int], Mapping[str, Any]], Mapping[str, Any]],
                 verify_fn: Callable[[str, int, Candidate, Mapping[str, Any]], Mapping[str, Any]],
                 retrieve_memory_fn: Optional[Callable[[str, int, Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]]] = None,
                 strong_geometry_fn: Optional[Callable[[str, int, Sequence[int], Mapping[str, Any]], Mapping[str, Any]]] = None):
        self._frame_counts = dict(frame_counts)
        self._frame_fn = frame_fn
        self._localize_fn = localize_fn
        self._verify_fn = verify_fn
        self._retrieve_memory_fn = retrieve_memory_fn
        self._strong_geometry_fn = strong_geometry_fn

    def num_frames(self, sequence: str) -> int:
        return int(self._frame_counts[sequence])

    def frame_observation(self, sequence: str, frame_index: int,
                          state: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._frame_fn(sequence, frame_index, state)

    def localize(self, sequence: str, frame_index: int,
                 reference_indices: Sequence[int],
                 state: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._localize_fn(sequence, frame_index, tuple(reference_indices), state)

    def retrieve_memory(self, sequence: str, frame_index: int,
                        memory: Sequence[Mapping[str, Any]],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._retrieve_memory_fn is not None:
            return self._retrieve_memory_fn(sequence, frame_index, memory, state)
        if not memory:
            return {"pose": None, "source": "memory", "evidence": {"available": False},
                    "causal": dict((key, False) for key in CAUSAL_FLAGS)}
        item = memory[-1]
        return {"pose": item["pose"], "source": "memory", "reference_indices": (),
                "evidence": {"available": True, "memory_id": item["memory_id"]},
                "causal": dict((key, False) for key in CAUSAL_FLAGS)}

    def strong_geometry(self, sequence: str, frame_index: int,
                        reference_indices: Sequence[int], state: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._strong_geometry_fn is None:
            raise RuntimeError("backend does not provide an online strong-geometry action")
        return self._strong_geometry_fn(sequence, frame_index, tuple(reference_indices), state)

    def verify(self, sequence: str, frame_index: int, candidate: Candidate,
               state: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._verify_fn(sequence, frame_index, candidate, state)


class AgenticLocalizationEnv:
    """Causal sequential environment with guarded Memory writes."""

    def __init__(self, backend: LocalizationBackend, *, max_actions_per_frame: int = 8,
                 require_online_execution: bool = True,
                 budget_s: Optional[float] = None,
                 allow_memory_query: bool = True):
        self.backend = backend
        self.max_actions_per_frame = int(max_actions_per_frame)
        self.require_online_execution = bool(require_online_execution)
        self.budget_s = None if budget_s is None else float(budget_s)
        self.allow_memory_query = bool(allow_memory_query)
        self.state: Optional[LocalizationState] = None
        self._last_event: Optional[Dict[str, Any]] = None
        self.trace: List[Dict[str, Any]] = []

    def reset(self, sequence: str, *, start_frame: int = 0) -> Observation:
        frame_count = int(self.backend.num_frames(sequence))
        if not 0 <= int(start_frame) < frame_count:
            raise ValueError("start_frame is outside the sequence")
        self.state = LocalizationState(sequence=sequence, frame_index=int(start_frame),
                                       frame_count=frame_count)
        self._last_event = {"type": "reset", "causal": dict((key, False) for key in CAUSAL_FLAGS)}
        self.trace = [{
            "type": "reset", "sequence": sequence, "frame_index": int(start_frame),
            "causal": dict((key, False) for key in CAUSAL_FLAGS),
        }]
        return self._observation()

    def fork(self) -> "AgenticLocalizationEnv":
        """Copy causal state for evaluator-only counterfactual planning.

        The copy shares the immutable/model backend but clones all environment
        state.  It is deliberately not part of the policy observation API:
        deployable policies never receive or need a branching interface.
        """
        if self.state is None:
            raise RuntimeError("call reset before fork")
        clone = AgenticLocalizationEnv(
            self.backend, max_actions_per_frame=self.max_actions_per_frame,
            require_online_execution=self.require_online_execution,
            budget_s=self.budget_s, allow_memory_query=self.allow_memory_query)
        clone.state = copy.deepcopy(self.state)
        clone._last_event = copy.deepcopy(self._last_event)
        clone.trace = copy.deepcopy(self.trace)
        return clone

    def step(self, action: LocalizationAction) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("call reset before step")
        if self.state.done:
            raise RuntimeError("step called after environment termination")
        if self.state.frame_action_count >= self.max_actions_per_frame:
            raise InvalidAction("per-frame action budget exhausted; defer or accept earlier")
        old_action_count = self.state.action_count
        old_frame_action_count = self.state.frame_action_count
        old_frame_tool_cost_s = self.state.frame_tool_cost_s
        state_before = self._state_snapshot()
        self.state.action_count += 1
        self.state.frame_action_count += 1
        try:
            if action.kind == "localize_with_references":
                event = self._localize(action)
                reward = -1.0
            elif action.kind == "run_strong_geometry":
                event = self._strong_geometry(action)
                reward = -2.0
            elif action.kind == "retrieve_memory":
                event = self._retrieve_memory(action)
                reward = -0.2
            elif action.kind == "verify_candidate":
                event = self._verify()
                reward = 0.2 if event["accepted"] else -0.2
            elif action.kind == "accept_candidate":
                event = self._accept()
                reward = 1.0 if event["accepted"] else -0.5
            elif action.kind in ("reject_candidate", "defer_frame"):
                event = self._finish_frame(action.kind)
                reward = -0.1 if action.kind == "defer_frame" else 0.0
            else:  # pragma: no cover - guarded by LocalizationAction
                raise InvalidAction(action.kind)
        except Exception:
            # An invalid action must not silently consume the per-frame budget.
            # Backend-side effects are the backend's responsibility; the causal
            # environment itself remains unchanged on action failure.
            self.state.action_count = old_action_count
            self.state.frame_action_count = old_frame_action_count
            self.state.frame_tool_cost_s = old_frame_tool_cost_s
            raise
        # This is online, tool-reported elapsed time.  It is intentionally
        # distinct from evaluator wall time and makes remaining budget part of
        # the policy-visible state rather than an evaluator-only side channel.
        self.state.frame_tool_cost_s += self._event_elapsed_s(event)
        self._last_event = event
        observation = self._observation()
        info = {
            "event": _copy_jsonable(event),
            "causal": dict((key, False) for key in CAUSAL_FLAGS),
        }
        self.trace.append({
            "type": "step", "sequence": self.state.sequence,
            "frame_index_before": state_before["frame_index"],
            "action": {"kind": action.kind,
                       "reference_indices": list(action.reference_indices),
                       "memory_ids": list(action.memory_ids)},
            "state_before": state_before,
            "event": _copy_jsonable(event),
            "state_after": self._state_snapshot(),
            "causal": dict((key, False) for key in CAUSAL_FLAGS),
        })
        return observation, float(reward), bool(self.state.done), info

    def _state_snapshot(self) -> Dict[str, Any]:
        assert self.state is not None
        candidate = self.state.current_candidate
        return {
            "frame_index": self.state.frame_index,
            "accepted_frame_index": self.state.accepted_frame_index,
            "accepted_pose_available": self.state.accepted_pose is not None,
            "reject_streak": self.state.reject_streak,
            "memory_ids": [item.memory_id for item in self.state.memory],
            "memory_frame_indices": [item.frame_index for item in self.state.memory],
            "memory_size": len(self.state.memory),
            "candidate_source": None if candidate is None else candidate.source,
            "candidate_verified": False if candidate is None else candidate.verified,
            "action_count": self.state.action_count,
            "frame_action_count": self.state.frame_action_count,
            "frame_tool_cost_s": self.state.frame_tool_cost_s,
            "done": self.state.done,
        }

    def audit_trace(self) -> Dict[str, Any]:
        """Return a strict audit of the environment's own causal trace."""
        violations = []
        steps = [record for record in self.trace if record.get("type") == "step"]
        for index, record in enumerate(self.trace):
            try:
                _assert_no_evaluator_fields(record, f"trace[{index}]")
            except CausalViolation as error:
                violations.append(str(error))
            causal = record.get("causal", {})
            if any(bool(causal.get(key, True)) for key in CAUSAL_FLAGS):
                violations.append(f"trace[{index}] has non-false causal flags")
        for index, record in enumerate(steps):
            action = record["action"]["kind"]
            event = record["event"]
            before = record["state_before"]
            after = record["state_after"]
            if action in ("localize_with_references", "run_strong_geometry"):
                execution = event.get("execution", {})
                if execution.get("executed") is not True or execution.get("cache_hit") is True:
                    violations.append(f"step[{index}] acquisition lacks live execution receipt")
            if event.get("type") == "accept":
                if not before.get("candidate_verified", False):
                    violations.append(f"step[{index}] accepted an unverified candidate")
                if after["memory_size"] <= before["memory_size"]:
                    violations.append(f"step[{index}] accept did not create a memory record")
            if any(frame >= after["frame_index"] for frame in after["memory_frame_indices"]):
                violations.append(f"step[{index}] exposed current/future memory")
        return {
            "steps": len(steps),
            "violations": violations,
            "query_gt_read": False,
            "uses_future": False,
            "selection_uses_gt": False,
            "passed": not violations,
        }

    def _backend_state(self) -> Dict[str, Any]:
        assert self.state is not None
        anchor_age = (self.state.frame_index - self.state.accepted_frame_index
                      if self.state.accepted_frame_index is not None
                      else self.state.frame_index + 1)
        return {
            "frame_index": self.state.frame_index,
            "accepted_frame_index": self.state.accepted_frame_index,
            "accepted_pose": (None if self.state.accepted_pose is None
                               else self.state.accepted_pose.copy()),
            "anchor_age": int(anchor_age),
            "accepted_pose_available": self.state.accepted_pose is not None,
            "reject_streak": self.state.reject_streak,
            "memory_size": len(self.state.memory),
            "frame_tool_cost_s": float(self.state.frame_tool_cost_s),
        }

    @staticmethod
    def _event_elapsed_s(event: Mapping[str, Any]) -> float:
        """Read a backend receipt without ever charging evaluator wall time."""
        execution = event.get("execution", {})
        evidence = event.get("evidence", {})
        value = (execution.get("elapsed_s", evidence.get("elapsed_s", 0.0))
                 if isinstance(execution, Mapping) and isinstance(evidence, Mapping) else 0.0)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def _candidate_from_result(self, result: Mapping[str, Any], *, source: str,
                               references: Sequence[int], require_execution: bool = False) -> Candidate:
        assert self.state is not None
        if not isinstance(result, Mapping):
            raise CausalViolation("backend action did not return a mapping")
        _assert_no_evaluator_fields(result)
        audit = _causal_audit(result)
        execution = dict(result.get("execution", {}))
        if require_execution and self.require_online_execution:
            if execution.get("executed") is not True or execution.get("cache_hit") is True:
                raise CausalViolation(
                    "reference-conditioned action must report executed=True and cache_hit=False")
            if int(execution.get("query_frame_index", -1)) != self.state.frame_index:
                raise CausalViolation("execution receipt does not identify the current query frame")
            executed_refs = tuple(int(x) for x in execution.get("reference_indices", ()))
            if executed_refs != tuple(int(x) for x in references):
                raise CausalViolation(
                    "execution receipt reference_indices do not match the action")
            stages = execution.get("stages", ())
            if not stages:
                raise CausalViolation("execution receipt does not expose inference stages")
        candidate_references = tuple(int(x) for x in references)
        if not candidate_references and "reference_indices" in result:
            candidate_references = tuple(int(x) for x in result.get("reference_indices", ()))
        candidate = Candidate(
            pose=_as_pose(result.get("pose")),
            source=str(result.get("source", source)),
            frame_index=self.state.frame_index,
            reference_indices=candidate_references,
            evidence=dict(result.get("evidence", {})),
            execution=execution,
        )
        candidate.evidence["causal"] = audit
        if candidate.pose is None:
            candidate.evidence["available"] = False
        return candidate

    def _localize(self, action: LocalizationAction) -> Dict[str, Any]:
        assert self.state is not None
        if not action.reference_indices:
            raise InvalidAction("localize_with_references requires at least one reference")
        result = self.backend.localize(self.state.sequence, self.state.frame_index,
                                       action.reference_indices, self._backend_state())
        candidate = self._candidate_from_result(result, source="localization",
                                                 references=action.reference_indices,
                                                 require_execution=True)
        self.state.current_candidate = candidate
        return {"type": "localize", "source": candidate.source,
                "references": list(action.reference_indices),
                "pose_available": candidate.pose is not None,
                "execution": candidate.execution,
                "causal": dict((key, False) for key in CAUSAL_FLAGS)}

    def _strong_geometry(self, action: LocalizationAction) -> Dict[str, Any]:
        assert self.state is not None
        if len(action.reference_indices) < 2:
            raise InvalidAction("run_strong_geometry requires at least two references")
        result = self.backend.strong_geometry(
            self.state.sequence, self.state.frame_index,
            action.reference_indices, self._backend_state())
        candidate = self._candidate_from_result(
            result, source="strong_geometry", references=action.reference_indices,
            require_execution=True)
        self.state.current_candidate = candidate
        return {"type": "strong_geometry", "source": candidate.source,
                "references": list(action.reference_indices),
                "pose_available": candidate.pose is not None,
                "execution": candidate.execution,
                "causal": dict((key, False) for key in CAUSAL_FLAGS)}

    def _retrieve_memory(self, action: LocalizationAction) -> Dict[str, Any]:
        assert self.state is not None
        if not self.allow_memory_query:
            raise InvalidAction("memory query is disabled by the active action protocol")
        if action.memory_ids:
            known = {item.memory_id: item for item in self.state.memory}
            missing = [x for x in action.memory_ids if x not in known]
            if missing:
                raise InvalidAction("unknown memory ids: " + repr(missing))
            selected = [known[x] for x in action.memory_ids]
        else:
            selected = list(self.state.memory)
        memory = [{"memory_id": x.memory_id, "frame_index": x.frame_index,
                   "pose": x.pose, "source": x.source, "confidence": x.confidence,
                   "metadata": x.metadata} for x in selected]
        if any(item["frame_index"] >= self.state.frame_index for item in memory):
            raise CausalViolation("memory query attempted to access current/future frame")
        result = self.backend.retrieve_memory(self.state.sequence, self.state.frame_index,
                                              memory, self._backend_state())
        candidate = self._candidate_from_result(result, source="memory", references=())
        self.state.current_candidate = candidate
        return {"type": "retrieve_memory", "memory_ids": [x["memory_id"] for x in memory],
                "source": candidate.source, "pose_available": candidate.pose is not None,
                "execution": candidate.execution,
                "causal": dict((key, False) for key in CAUSAL_FLAGS)}

    def _verify(self) -> Dict[str, Any]:
        assert self.state is not None
        candidate = self.state.current_candidate
        if candidate is None or candidate.pose is None:
            raise InvalidAction("verify_candidate requires an available candidate")
        result = self.backend.verify(self.state.sequence, self.state.frame_index,
                                     candidate, self._backend_state())
        if not isinstance(result, Mapping):
            raise CausalViolation("verifier did not return a mapping")
        _assert_no_evaluator_fields(result)
        audit = _causal_audit(result)
        candidate.verified = bool(result.get("accepted", False))
        candidate.verification = dict(result.get("evidence", result.get("verification", {})))
        candidate.verification["causal"] = audit
        return {"type": "verify", "accepted": candidate.verified,
                "source": candidate.source, "evidence": candidate.verification,
                "causal": dict((key, False) for key in CAUSAL_FLAGS)}

    def _accept(self) -> Dict[str, Any]:
        assert self.state is not None
        candidate = self.state.current_candidate
        if candidate is None or candidate.pose is None or not candidate.verified:
            raise InvalidAction("accept_candidate requires a verified candidate")
        self.state.accepted_pose = candidate.pose.copy()
        self.state.accepted_frame_index = self.state.frame_index
        self.state.reject_streak = 0
        memory_id = len(self.state.memory)
        confidence = float(candidate.verification.get("confidence", 1.0))
        self.state.memory.append(MemoryRecord(
            memory_id=memory_id, frame_index=self.state.frame_index,
            pose=candidate.pose.copy(), source=candidate.source,
            confidence=max(0.0, min(1.0, confidence)),
            metadata={"references": list(candidate.reference_indices),
                      "verification": _copy_jsonable(candidate.verification)},
        ))
        event = {"type": "accept", "accepted": True, "memory_id": memory_id,
                 "source": candidate.source,
                 "causal": dict((key, False) for key in CAUSAL_FLAGS)}
        self._advance_frame()
        return event

    def _finish_frame(self, reason: str) -> Dict[str, Any]:
        assert self.state is not None
        self.state.current_candidate = None
        self.state.reject_streak += 1
        event = {"type": reason, "accepted": False, "causal": dict((key, False) for key in CAUSAL_FLAGS)}
        self._advance_frame()
        return event

    def _advance_frame(self) -> None:
        assert self.state is not None
        self.state.frame_index += 1
        self.state.frame_action_count = 0
        self.state.frame_tool_cost_s = 0.0
        self.state.current_candidate = None
        if self.state.frame_index >= self.state.frame_count:
            self.state.done = True

    def _observation(self) -> Observation:
        assert self.state is not None
        frame = {} if self.state.done else dict(self.backend.frame_observation(
            self.state.sequence, self.state.frame_index, self._backend_state()))
        _assert_no_evaluator_fields(frame, "observation.frame")
        candidate = self.state.current_candidate
        candidate_view = None
        if candidate is not None:
            candidate_view = {
                "source": candidate.source,
                "frame_index": candidate.frame_index,
                "reference_indices": list(candidate.reference_indices),
                "pose_available": candidate.pose is not None,
                "verified": candidate.verified,
                "evidence": _copy_jsonable(candidate.evidence),
                "verification": _copy_jsonable(candidate.verification),
                "execution": _copy_jsonable(candidate.execution),
            }
        memory_view = [{
            "memory_id": x.memory_id, "frame_index": x.frame_index,
            "source": x.source, "confidence": x.confidence,
            "metadata": _copy_jsonable(x.metadata),
        } for x in self.state.memory]
        state_view = {
            "accepted_pose_available": self.state.accepted_pose is not None,
            "accepted_frame_index": self.state.accepted_frame_index,
            "anchor_age": (self.state.frame_index - self.state.accepted_frame_index
                            if self.state.accepted_frame_index is not None
                            else self.state.frame_index + 1),
            "reject_streak": self.state.reject_streak,
            "memory_size": len(self.state.memory),
            "action_count": self.state.action_count,
            "frame_action_count": self.state.frame_action_count,
            "frame_tool_cost_s": float(self.state.frame_tool_cost_s),
        }
        if self.budget_s is not None:
            state_view["budget_s"] = self.budget_s
            state_view["remaining_budget_s"] = max(
                0.0, self.budget_s - float(self.state.frame_tool_cost_s))
        return Observation(self.state.sequence, self.state.frame_index,
                           self.state.frame_count, _copy_jsonable(frame), state_view,
                           candidate_view, memory_view,
                           _copy_jsonable(self._last_event), self.state.done)


__all__ = [
    "ACTION_KINDS", "AgenticLocalizationEnv", "CallableLocalizationBackend",
    "Candidate", "CausalViolation", "LocalizationAction", "LocalizationBackend",
    "LocalizationState", "MemoryRecord", "Observation", "InvalidAction",
]
