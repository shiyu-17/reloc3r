#!/usr/bin/env python3
"""Lightweight PoseAgent-style orchestration primitives.

This module deliberately separates three concerns that were previously mixed
inside one controller: pre-execution profiling/ranking, candidate verification,
and verification-guided fallback.  It is intentionally *not* a reimplementation
of PoseAgent's heavy visual PVN.  The first implementation uses only
policy-visible structured evidence and can therefore be trained and audited on
one RTX 4090 or CPU.

Ground-truth poses and terminal outcomes are forbidden from all public online
APIs here.  They may appear only in an offline training row under ``label``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

import numpy as np

from tools.agentic_localization_env import LocalizationAction


FORBIDDEN_ONLINE_FIELDS = {
    "query_gt", "query_gt_pose", "ground_truth", "ground_truth_pose", "gt_pose",
    "translation_error", "rotation_error", "success", "oracle", "outcome",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise TypeError("observation must be a mapping or expose as_dict()")
    return value


def _number(value: Any, default: float = 0.0, clip: float = 1e6) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return float(np.clip(value, -clip, clip)) if np.isfinite(value) else default


def _assert_online_safe(value: Any, path: str = "online") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_ONLINE_FIELDS:
                raise ValueError(f"forbidden evaluator-only field at {path}.{key}")
            _assert_online_safe(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_online_safe(child, f"{path}[{index}]")


@dataclass(frozen=True)
class PoseEstimatorTool:
    """A callable pose-estimation tool exposed through the causal environment."""

    tool_id: str
    action_kind: str
    min_references: int
    nominal_cost_s: float
    source_ids: tuple[str, ...]

    def action(self, references: Sequence[int]) -> LocalizationAction:
        refs = tuple(int(value) for value in references)
        if len(refs) < self.min_references:
            raise ValueError(f"{self.tool_id} requires {self.min_references} references")
        if self.action_kind == "localize_with_references":
            return LocalizationAction.localize(*refs)
        if self.action_kind == "run_strong_geometry":
            return LocalizationAction.geometry(*refs)
        raise ValueError(f"unsupported environment action kind: {self.action_kind}")


DEFAULT_TOOLS = (
    PoseEstimatorTool("reloc3r", "localize_with_references", 1, 1.0,
                      ("reloc3r_online", "localization")),
    PoseEstimatorTool("roma", "run_strong_geometry", 2, 4.0,
                      ("roma_online_geometry", "strong_geometry")),
)


@dataclass(frozen=True)
class RankedPoseAction:
    """One pre-execution tool/reference-group choice."""

    tool: PoseEstimatorTool
    group_index: int
    references: tuple[int, ...]
    profile: Mapping[str, float]
    score: float

    def action(self) -> LocalizationAction:
        return self.tool.action(self.references)

    @property
    def key(self) -> tuple[str, tuple[int, ...]]:
        return self.tool.tool_id, self.references


class PairProfiler:
    """Build cheap, reference-ID-free profiles for tool ranking.

    This is the resource-aware counterpart of PoseAgent's profiling stage.  It
    only consumes retrieval statistics and causal state now; visual profiling
    features can be appended later without changing ranker/controller APIs.
    """

    FEATURE_NAMES = (
        "group_rank", "group_size", "retrieval_mean", "retrieval_max",
        "retrieval_min", "retrieval_std", "map_spread_m", "anchor_age",
        "reject_streak", "memory_size", "has_anchor", "frame_action_count",
        "spent_budget_fraction", "remaining_budget_fraction", "tool_reloc3r",
        "tool_roma", "tool_nominal_cost_s",
    )

    @staticmethod
    def groups(observation: Any) -> list[Mapping[str, Any]]:
        value = _mapping(observation)
        groups = value.get("frame", {}).get("reference_groups", [])
        return [group for group in groups if group.get("reference_indices")]

    def profile(self, observation: Any, tool: PoseEstimatorTool,
                group_index: int, group: Mapping[str, Any]) -> Dict[str, float]:
        value = _mapping(observation)
        _assert_online_safe(value)
        scores = [_number(item) for item in group.get("retrieval_scores", ())]
        state = value.get("state", {})
        budget = max(_number(state.get("budget_s"), 1.0), 1e-6)
        spent = max(_number(state.get("frame_tool_cost_s")), 0.0)
        remaining = max(_number(state.get("remaining_budget_s"), budget), 0.0)
        return {
            "group_rank": float(group_index),
            "group_size": float(len(group.get("reference_indices", ()))),
            "retrieval_mean": _number(group.get("mean_retrieval_score"),
                                        float(np.mean(scores)) if scores else 0.0),
            "retrieval_max": max(scores, default=0.0),
            "retrieval_min": min(scores, default=0.0),
            "retrieval_std": float(np.std(scores)) if scores else 0.0,
            "map_spread_m": _number(group.get("map_spread_m")),
            "anchor_age": _number(state.get("anchor_age")),
            "reject_streak": _number(state.get("reject_streak")),
            "memory_size": _number(state.get("memory_size")),
            "has_anchor": float(bool(state.get("accepted_pose_available"))),
            "frame_action_count": _number(state.get("frame_action_count")),
            "spent_budget_fraction": min(spent / budget, 1.0),
            "remaining_budget_fraction": min(remaining / budget, 1.0),
            "tool_reloc3r": float(tool.tool_id == "reloc3r"),
            "tool_roma": float(tool.tool_id == "roma"),
            "tool_nominal_cost_s": float(tool.nominal_cost_s),
        }

    def vector(self, profile: Mapping[str, Any]) -> np.ndarray:
        return np.asarray([_number(profile.get(name)) for name in self.FEATURE_NAMES], dtype=np.float32)


class ActionRanker(Protocol):
    def score(self, profiles: Sequence[Mapping[str, float]]) -> np.ndarray:
        ...


@dataclass
class RuleRanker:
    """Fixed, transparent ranking baseline; no outcome labels are used."""

    retrieval_weight: float = 1.0
    group_rank_penalty: float = 0.03
    cost_penalty: float = 0.03
    geometry_penalty: float = 0.02

    def score(self, profiles: Sequence[Mapping[str, float]]) -> np.ndarray:
        values = []
        for profile in profiles:
            score = self.retrieval_weight * _number(profile.get("retrieval_mean"))
            score -= self.group_rank_penalty * _number(profile.get("group_rank"))
            score -= self.cost_penalty * _number(profile.get("tool_nominal_cost_s"))
            score -= self.geometry_penalty * _number(profile.get("tool_roma"))
            values.append(score)
        return np.asarray(values, dtype=np.float64)


@dataclass
class XGBRankerAdapter:
    """Inference wrapper for an offline-trained XGBRanker artifact."""

    model: Any
    profiler: PairProfiler = field(default_factory=PairProfiler)

    @classmethod
    def load(cls, model_path: Path, profiler: PairProfiler | None = None) -> "XGBRankerAdapter":
        try:
            import xgboost as xgb
        except ModuleNotFoundError as error:  # pragma: no cover - deployment dependency
            raise RuntimeError("xgboost is required to load a structured PoseAgent ranker") from error
        model = xgb.XGBRanker()
        model.load_model(Path(model_path))
        return cls(model, profiler or PairProfiler())

    def score(self, profiles: Sequence[Mapping[str, float]]) -> np.ndarray:
        matrix = np.stack([self.profiler.vector(profile) for profile in profiles])
        return np.asarray(self.model.predict(matrix), dtype=np.float64)


@dataclass
class StructuredRanker:
    """Small scene-disjoint action-success ranker trained on cheap profiles."""

    classifier: Any
    profiler: PairProfiler = field(default_factory=PairProfiler)
    cost_penalty: float = 0.02

    @classmethod
    def load(cls, directory: Path, profiler: PairProfiler | None = None) -> "StructuredRanker":
        import joblib
        directory = Path(directory)
        metadata = json.loads(directory.joinpath("metadata.json").read_text(encoding="utf-8"))
        if metadata.get("family") != "poseagent_structured_ranker":
            raise ValueError("not a structured PoseAgent ranker artifact")
        if tuple(metadata.get("feature_names", ())) != PairProfiler.FEATURE_NAMES:
            raise ValueError("ranker feature schema mismatch")
        return cls(joblib.load(directory / "success_classifier.joblib"), profiler or PairProfiler(),
                   float(metadata.get("cost_penalty", 0.02)))

    def score(self, profiles: Sequence[Mapping[str, float]]) -> np.ndarray:
        matrix = np.stack([self.profiler.vector(profile) for profile in profiles])
        probability = np.asarray(self.classifier.predict_proba(matrix)[:, 1], dtype=np.float64)
        costs = np.asarray([_number(profile.get("tool_nominal_cost_s")) for profile in profiles])
        return probability - self.cost_penalty * costs


@dataclass(frozen=True)
class VerificationDecision:
    accept: bool
    confidence: float
    predicted_translation_error_m: float | None
    predicted_rotation_error_deg: float | None
    verifier: str
    reason: str


class CandidateVerifier(Protocol):
    def evaluate(self, observation: Any) -> VerificationDecision:
        ...


class RuleVerifier:
    """Adapter around the existing source-aware backend verifier.

    The environment has already executed it through ``verify_candidate``.
    This wrapper makes its decision available to the common fallback
    controller without reimplementing, weakening, or silently bypassing its
    safety contract.
    """

    def evaluate(self, observation: Any) -> VerificationDecision:
        value = _mapping(observation)
        _assert_online_safe(value)
        candidate = value.get("candidate") or {}
        verification = candidate.get("verification") or {}
        accepted = bool(candidate.get("verified", False))
        confidence = float(np.clip(_number(verification.get("confidence")), 0.0, 1.0))
        return VerificationDecision(
            accept=accepted,
            confidence=confidence,
            predicted_translation_error_m=None,
            predicted_rotation_error_deg=None,
            verifier="rule",
            reason="backend_accepted" if accepted else "backend_rejected",
        )


class StructuredVerifierFeatures:
    """Fixed feature schema for the inexpensive learned verifier."""

    NAMES = (
        "source_reloc3r", "source_roma", "source_memory", "num_references",
        "translation_dispersion_m", "rotation_dispersion_deg", "inliers",
        "reprojection_error_px", "correspondences", "total_risk", "independent_risk",
        "previous_state_risk", "state_trust", "confidence", "heldout_translation_m",
        "heldout_rotation_deg", "heldout_inliers", "heldout_inlier_ratio",
        "heldout_epipolar_px", "cheap_valid_matches", "cheap_coverage_ratio",
        "cheap_spatial_entropy", "appearance_change_score", "temporal_translation_m",
        "temporal_rotation_deg", "anchor_age", "reject_streak", "memory_size",
        "frame_tool_cost_s", "remaining_budget_fraction",
    )

    @classmethod
    def vector(cls, observation: Any, candidate: Mapping[str, Any] | None = None) -> np.ndarray:
        value = _mapping(observation)
        _assert_online_safe(value)
        candidate = dict(candidate or value.get("candidate") or {})
        evidence = candidate.get("evidence") or {}
        verification = candidate.get("verification") or {}
        independent = verification.get("independent_verification") or {}
        cheap = evidence.get("cheap_correspondence") or {}
        appearance = evidence.get("appearance_change") or {}
        temporal = evidence.get("temporal_consistency") or {}
        state = value.get("state") or {}
        budget = max(_number(state.get("budget_s"), 1.0), 1e-6)
        source = str(candidate.get("source", ""))
        features = (
            float(source == "reloc3r_online"),
            float(source == "roma_online_geometry"),
            float(source == "memory"),
            _number(evidence.get("num_references")),
            _number(evidence.get("translation_dispersion_m")),
            _number(evidence.get("rotation_dispersion_deg")),
            _number(evidence.get("inliers")),
            _number(evidence.get("median_reprojection_px", evidence.get("reprojection_error_px"))),
            _number(evidence.get("correspondences")),
            _number(verification.get("total_risk")),
            _number(verification.get("independent_risk")),
            _number(verification.get("previous_state_risk")),
            _number(verification.get("state_trust")),
            _number(verification.get("confidence")),
            _number(independent.get("translation_disagreement_m")),
            _number(independent.get("rotation_disagreement_deg")),
            _number(independent.get("inliers")),
            _number(independent.get("inlier_ratio")),
            _number(independent.get("median_epipolar_px")),
            _number(cheap.get("valid_matches")),
            _number(cheap.get("coverage_ratio")),
            _number(cheap.get("spatial_entropy")),
            _number(appearance.get("change_score")),
            _number(temporal.get("translation_disagreement_m")),
            _number(temporal.get("rotation_disagreement_deg")),
            _number(state.get("anchor_age")),
            _number(state.get("reject_streak")),
            _number(state.get("memory_size")),
            _number(state.get("frame_tool_cost_s")),
            min(_number(state.get("remaining_budget_s"), budget) / budget, 1.0),
        )
        return np.asarray(features, dtype=np.float32)


@dataclass
class StructuredVerifier:
    """Calibrated XGBoost verifier trained only from evaluator-side labels.

    ``require_rule_accept`` is true by default: a learned score can make the
    system *more conservative*, but cannot turn a rule-rejected candidate into
    a Memory-writing acceptance before a dedicated safety experiment permits
    it.
    """

    success_model: Any
    translation_model: Any
    rotation_model: Any
    acceptance_threshold: float
    require_rule_accept: bool = True

    @classmethod
    def load(cls, model_dir: Path) -> "StructuredVerifier":
        root = Path(model_dir)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("feature_names") != list(StructuredVerifierFeatures.NAMES):
            raise ValueError("structured verifier feature schema does not match this code")
        backend = str(metadata.get("model_backend", "xgboost"))
        if backend == "xgboost":
            try:
                import xgboost as xgb
            except ModuleNotFoundError as error:  # pragma: no cover - deployment dependency
                raise RuntimeError("this structured verifier artifact requires xgboost") from error
            success = xgb.XGBClassifier(); success.load_model(root / "success_classifier.json")
            translation = xgb.XGBRegressor(); translation.load_model(root / "translation_regressor.json")
            rotation = xgb.XGBRegressor(); rotation.load_model(root / "rotation_regressor.json")
        elif backend == "sklearn_hist":
            try:
                import joblib
            except ModuleNotFoundError as error:  # pragma: no cover - deployment dependency
                raise RuntimeError("this structured verifier artifact requires joblib") from error
            success = joblib.load(root / "success_classifier.joblib")
            translation = joblib.load(root / "translation_regressor.joblib")
            rotation = joblib.load(root / "rotation_regressor.joblib")
        else:
            raise ValueError("unknown structured verifier model backend: " + backend)
        return cls(success, translation, rotation,
                   float(metadata["acceptance_threshold"]),
                   bool(metadata.get("require_rule_accept", True)))

    def evaluate(self, observation: Any) -> VerificationDecision:
        value = _mapping(observation)
        candidate = value.get("candidate") or {}
        if not candidate.get("pose_available", False):
            return VerificationDecision(False, 0.0, None, None, "structured", "missing_pose")
        x = StructuredVerifierFeatures.vector(value)[None]
        confidence = float(np.clip(self.success_model.predict_proba(x)[0, 1], 0.0, 1.0))
        translation = float(max(0.0, np.expm1(self.translation_model.predict(x)[0])))
        rotation = float(max(0.0, np.expm1(self.rotation_model.predict(x)[0])))
        rule_accepted = bool(candidate.get("verified", False))
        accept = confidence >= self.acceptance_threshold and (
            rule_accepted or not self.require_rule_accept)
        if self.require_rule_accept and not rule_accepted:
            reason = "rule_rejected"
        elif confidence < self.acceptance_threshold:
            reason = "below_calibrated_threshold"
        else:
            reason = "structured_accepted"
        return VerificationDecision(accept, confidence, translation, rotation,
                                    "structured", reason)


@dataclass(frozen=True)
class ControllerDecision:
    action: LocalizationAction
    metadata: Mapping[str, Any]


class PoseAgentFallbackController:
    """Verification-guided, budgeted fallback over estimator/reference actions."""

    def __init__(self, *, tools: Sequence[PoseEstimatorTool] = DEFAULT_TOOLS,
                 profiler: PairProfiler | None = None, ranker: ActionRanker | None = None,
                 verifier: CandidateVerifier | None = None, max_estimator_calls: int = 2):
        if max_estimator_calls < 1:
            raise ValueError("max_estimator_calls must be positive")
        self.tools = tuple(tools)
        self.profiler = profiler or PairProfiler()
        self.ranker = ranker or RuleRanker()
        self.verifier = verifier or RuleVerifier()
        self.max_estimator_calls = int(max_estimator_calls)
        self._frame_key: tuple[str, int] | None = None
        self._attempted: set[tuple[str, tuple[int, ...]]] = set()
        # Off by default: this is a registered parity ablation against the
        # feedback rule, not a silently changed default policy.
        self.geometry_after_verifier_reject = False

    def enable_geometry_after_verifier_reject(self) -> None:
        """Prioritize one strong-geometry retry after a rejected cheap pose.

        The trigger is entirely online: an existing verifier receipt rejects
        a pose-producing localization result.  No ground truth, learned
        score, or verifier threshold is changed.
        """
        self.geometry_after_verifier_reject = True

    def reset(self) -> None:
        self._frame_key = None
        self._attempted.clear()

    def _prepare_frame(self, observation: Mapping[str, Any]) -> None:
        key = (str(observation.get("sequence")), int(observation.get("frame_index", -1)))
        if key != self._frame_key:
            self._frame_key = key
            self._attempted.clear()

    def _ranked_actions(self, observation: Mapping[str, Any]) -> list[RankedPoseAction]:
        raw: list[tuple[PoseEstimatorTool, int, tuple[int, ...], Mapping[str, float]]] = []
        for group_index, group in enumerate(self.profiler.groups(observation)):
            references = tuple(int(item) for item in group.get("reference_indices", ()))
            for tool in self.tools:
                key = (tool.tool_id, references)
                if key in self._attempted or len(references) < tool.min_references:
                    continue
                raw.append((tool, group_index, references,
                            self.profiler.profile(observation, tool, group_index, group)))
        if not raw:
            return []
        scores = self.ranker.score([item[3] for item in raw])
        order = sorted(range(len(raw)), key=lambda index: (-float(scores[index]), raw[index][1], raw[index][0].tool_id))
        return [RankedPoseAction(*raw[index], score=float(scores[index])) for index in order]

    def choose(self, observation: Any) -> ControllerDecision:
        value = _mapping(observation)
        _assert_online_safe(value)
        self._prepare_frame(value)
        candidate = value.get("candidate") or {}
        if candidate.get("pose_available") and not candidate.get("verification"):
            return ControllerDecision(LocalizationAction.verify(), {"controller": "poseagent", "reason": "verify_candidate"})
        if candidate.get("pose_available") and candidate.get("verification"):
            verdict = self.verifier.evaluate(value)
            if verdict.accept and bool(candidate.get("verified", False)):
                return ControllerDecision(LocalizationAction.accept(), {
                    "controller": "poseagent", "reason": "verifier_accept", "verdict": verdict.__dict__,
                })
            if self.geometry_after_verifier_reject and not bool(candidate.get("verified", False)):
                refs = tuple(int(item) for item in candidate.get("reference_indices", ()))
                for tool in self.tools:
                    key = (tool.tool_id, refs)
                    if (tool.action_kind == "run_strong_geometry" and len(refs) >= tool.min_references and
                            key not in self._attempted and len(self._attempted) < self.max_estimator_calls):
                        self._attempted.add(key)
                        return ControllerDecision(tool.action(refs), {
                            "controller": "poseagent", "reason": "geometry_after_verifier_reject",
                            "tool": tool.tool_id, "references": list(refs),
                            "attempt_count": len(self._attempted),
                        })
        ranked = self._ranked_actions(value)
        if ranked and len(self._attempted) < self.max_estimator_calls:
            chosen = ranked[0]
            self._attempted.add(chosen.key)
            return ControllerDecision(chosen.action(), {
                "controller": "poseagent", "reason": "ranked_fallback", "tool": chosen.tool.tool_id,
                "group_index": chosen.group_index, "references": list(chosen.references),
                "rank_score": chosen.score, "profile": dict(chosen.profile),
                "attempt_count": len(self._attempted),
            })
        return ControllerDecision(LocalizationAction.defer(), {
            "controller": "poseagent", "reason": "budget_or_candidates_exhausted",
            "attempt_count": len(self._attempted),
        })


__all__ = [
    "ActionRanker", "CandidateVerifier", "ControllerDecision", "DEFAULT_TOOLS",
    "PairProfiler", "PoseAgentFallbackController", "PoseEstimatorTool", "RankedPoseAction",
    "RuleRanker", "RuleVerifier", "StructuredVerifier", "StructuredVerifierFeatures",
    "VerificationDecision", "XGBRankerAdapter", "StructuredRanker",
]
