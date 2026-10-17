#!/usr/bin/env python3
"""Deployable structured-evidence XGBoost controllers for Stage 3C.5.

The ranker is intentionally used only for ordering acquisition actions.  Its
raw ranking score is not a calibrated probability, so terminal control uses
the separately fitted recovery-probability and incremental-cost models.  This
keeps the XGBRanker experiment a test of acquisition ranking rather than an
uncontrolled change to terminal safety semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from tools.stage3b_policy import action_features, state_features
from tools.stage3c_recovery_policy import (ACQUISITION_KINDS,
                                           TERMINAL_EVIDENCE_SOURCES, _number)
from tools.stage3d_features import feature_vector, strip_post_geometry_fields
from tools.train_stage3c_xgboost_diagnostic import xgboost


@dataclass
class XGBRecoveryPolicy:
    """Explicit P-recover/cost controller or XGBRanker acquisition controller."""

    classifier: Any
    regressor: Any
    ranker: Any
    lambda_cost: float
    defer_threshold: float
    terminal_continue_margin: float
    acquisition_mode: str
    feature_level: str = "legacy"

    @classmethod
    def load(cls, model_dir: Path, *, mode: str, lambda_cost: float = 0.05,
             defer_threshold: float = 0.0, terminal_continue_margin: float = 0.05) -> "XGBRecoveryPolicy":
        if mode not in {"explicit", "ranker"}:
            raise ValueError(f"unknown XGB acquisition mode: {mode}")
        xgb = xgboost()
        classifier = xgb.XGBClassifier(); classifier.load_model(model_dir / "recovery_classifier.json")
        regressor = xgb.XGBRegressor(); regressor.load_model(model_dir / "incremental_cost_regressor.json")
        ranker = xgb.XGBRanker(); ranker.load_model(model_dir / "action_ranker.json")
        feature_level = "legacy"
        summary_path = model_dir / "diagnostic_summary.json"
        if summary_path.exists():
            feature_level = str(__import__("json").loads(summary_path.read_text()).get("feature_level", "legacy"))
        return cls(classifier, regressor, ranker, float(lambda_cost), float(defer_threshold),
                   float(terminal_continue_margin), mode, feature_level)

    def score_actions(self, observation: Any, actions: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        if not actions:
            raise ValueError("actions must be non-empty")
        if self.feature_level == "legacy":
            state = state_features(observation)
        else:
            # A second decision after a Geometry action must not consume its
            # post-action evidence, but the controller still needs a valid
            # pre-action vector.  Drop those fields explicitly at the online
            # boundary; training/evaluation feature builders remain fail-closed.
            candidate = (observation.as_dict() if hasattr(observation, "as_dict") else observation)
            candidate_evidence = ((candidate.get("candidate") or {}).get("evidence") or {})
            try:
                state = feature_vector(observation, self.feature_level)
            except ValueError:
                if not candidate_evidence:
                    raise
                state = feature_vector(strip_post_geometry_fields(candidate), self.feature_level)
        features = np.stack([np.concatenate((state, action_features(observation, action)))
                             for action in actions]).astype(np.float32)
        recovery = self.classifier.predict_proba(features)[:, 1].astype(np.float64)
        cost = np.expm1(self.regressor.predict(features)).clip(0.0, None).astype(np.float64)
        explicit_advantage = recovery - self.lambda_cost * cost
        rank_score = self.ranker.predict(features).astype(np.float64)
        chosen_score = rank_score if self.acquisition_mode == "ranker" else explicit_advantage
        return {"recovery_probability": recovery, "predicted_incremental_cost_s": cost,
                "explicit_advantage": explicit_advantage, "rank_score": rank_score,
                "selection_score": chosen_score}

    def choose(self, observation: Any, candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        acquisition_indices = [i for i, action in enumerate(candidates)
                               if str(action.get("kind")) in ACQUISITION_KINDS]
        accept_indices = [i for i, action in enumerate(candidates) if str(action.get("kind")) == "accept"]
        defer_indices = [i for i, action in enumerate(candidates)
                         if str(action.get("kind")) in {"defer", "reject"}]
        state = observation.as_dict() if hasattr(observation, "as_dict") else observation
        candidate = state.get("candidate") or {}
        verified = bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
        terminal_eligible = verified and str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES
        if not acquisition_indices:
            if accept_indices and terminal_eligible:
                return accept_indices[0], {"controller": "xgb_terminal", "reason": "verified_no_remaining_acquisition"}
            if defer_indices:
                return defer_indices[0], {"controller": "xgb_terminal",
                    "reason": "memory_requires_fresh_evidence" if verified and not terminal_eligible else "no_acquisition"}
            raise ValueError("candidate set has neither acquisition nor terminal action")

        acquisition = [candidates[i] for i in acquisition_indices]
        scores = self.score_actions(observation, acquisition)
        local = int(np.argmax(scores["selection_score"]))
        selected = acquisition_indices[local]
        # Ranker outputs are ordinal; only the calibrated explicit value can
        # be compared with a verifier confidence for terminal control.
        best_explicit_advantage = float(np.max(scores["explicit_advantage"]))
        confidence = float(np.clip(_number((candidate.get("verification") or {}).get("confidence")), 0.0, 1.0))
        metadata = {"controller": f"xgb_{self.acquisition_mode}",
                    "best_explicit_advantage": best_explicit_advantage,
                    "verifier_confidence": confidence,
                    **{key: value.tolist() for key, value in scores.items()}}
        if terminal_eligible and accept_indices and best_explicit_advantage <= confidence + self.terminal_continue_margin:
            return accept_indices[0], {**metadata, "reason": "accept_beats_continuation"}
        if not terminal_eligible and defer_indices and best_explicit_advantage <= self.defer_threshold:
            return defer_indices[0], {**metadata, "reason": "defer_beats_acquisition"}
        return selected, {**metadata, "reason": "acquire"}
