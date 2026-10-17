#!/usr/bin/env python3
"""Deployable Stage 3C.6 utility-consistent controller."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from tools.stage3b_policy import action_features, observation_dict, state_features
from tools.stage3c_recovery_policy import TERMINAL_EVIDENCE_SOURCES
from tools.stage3d_features import FEATURE_LEVELS, feature_vector, strip_post_geometry_fields
from tools.train_stage3c_xgboost_diagnostic import xgboost


def compose_utility(components: Mapping[str, np.ndarray], utility: Mapping[str, Any]) -> np.ndarray:
    return (
        float(utility["success_reward"]) * components["correct_accept"]
        - float(utility["false_accept_penalty"]) * components["false_accept"]
        - float(utility["cost_penalty_per_s"]) * components["incremental_cost_s"]
        - float(utility["recovery_penalty_per_frame"]) * components["recovery_time_frames"]
        - float(utility["memory_pollution_penalty"]) * components["memory_pollution"]
    )


def policy_state_features(observation: Any, feature_level: str) -> np.ndarray:
    """Build online features after crossing the object-to-dict boundary.

    ``Observation`` is a dataclass during rollout but training rows are plain
    dictionaries.  Projection must happen after normalization; otherwise a
    post-Geometry receipt such as ``inliers`` survives and the strict Stage 3D
    leakage check correctly aborts the second decision.
    """
    if feature_level == "legacy":
        return state_features(observation)
    normalized = observation_dict(observation)
    return feature_vector(strip_post_geometry_fields(normalized), feature_level)


@dataclass
class UtilityComponentPolicy:
    models: Dict[str, Any]
    utility: Dict[str, Any]
    feature_level: str = "legacy"

    @classmethod
    def load(cls, model_dir: Path) -> "UtilityComponentPolicy":
        xgb = xgboost()
        specifications = {
            "correct_accept": (xgb.XGBClassifier, "correct_accept_classifier.json"),
            "false_accept": (xgb.XGBClassifier, "false_accept_classifier.json"),
            "incremental_cost_s": (xgb.XGBRegressor, "incremental_cost_regressor.json"),
            "recovery_time_frames": (xgb.XGBRegressor, "recovery_time_regressor.json"),
            "memory_pollution": (xgb.XGBClassifier, "memory_pollution_classifier.json"),
        }
        models = {}
        for key, (constructor, filename) in specifications.items():
            model = constructor(); model.load_model(Path(model_dir) / filename); models[key] = model
        summary = json.loads((Path(model_dir) / "training_summary.json").read_text(encoding="utf-8"))
        if summary.get("family") != "stage3c6_utility_controller":
            raise ValueError("not a Stage 3C.6 utility controller")
        feature_level = str(summary.get("feature_level", "legacy"))
        if feature_level not in {"legacy", *FEATURE_LEVELS}:
            raise ValueError(f"unsupported Stage 3D feature level: {feature_level}")
        return cls(models=models, utility=dict(summary["utility"]), feature_level=feature_level)

    def score_actions(self, observation: Any,
                      actions: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        if not actions:
            raise ValueError("actions must be non-empty")
        state = policy_state_features(observation, self.feature_level)
        features = np.stack([np.concatenate((state, action_features(observation, action)))
                             for action in actions]).astype(np.float32)
        components = {
            "correct_accept": self.models["correct_accept"].predict_proba(features)[:, 1].astype(np.float64),
            "false_accept": self.models["false_accept"].predict_proba(features)[:, 1].astype(np.float64),
            "memory_pollution": self.models["memory_pollution"].predict_proba(features)[:, 1].astype(np.float64),
            "incremental_cost_s": np.expm1(
                self.models["incremental_cost_s"].predict(features)).clip(0.0, None).astype(np.float64),
            "recovery_time_frames": np.expm1(
                self.models["recovery_time_frames"].predict(features)).clip(0.0, None).astype(np.float64),
        }
        components["expected_utility"] = compose_utility(components, self.utility)
        return components

    def choose(self, observation: Any,
               candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        scores = self.score_actions(observation, candidates)
        expected = scores["expected_utility"].copy()
        obs = observation_dict(observation)
        candidate = obs.get("candidate") or {}
        terminal_eligible = (bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
                             and str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES)
        masked = []
        for index, action in enumerate(candidates):
            if str(action.get("kind")) == "accept" and not terminal_eligible:
                expected[index] = -np.inf; masked.append(index)
        if not np.isfinite(expected).any():
            raise ValueError("utility controller has no legal action")
        choice = int(np.argmax(expected))
        kinds = [str(action.get("kind")) for action in candidates]
        acquisition = [i for i, kind in enumerate(kinds) if kind in {"localize", "geometry", "memory"}]
        terminal = [i for i, kind in enumerate(kinds) if kind in {"accept", "defer", "reject"}
                    and i not in masked]
        metadata = {
            "controller": "stage3c6_component_utility", "reason": "maximum_expected_utility",
            "feature_level": self.feature_level,
            "terminal_eligible": terminal_eligible, "masked_action_indices": masked,
            "best_acquisition_utility": (None if not acquisition else float(max(expected[i] for i in acquisition))),
            "best_terminal_utility": (None if not terminal else float(max(expected[i] for i in terminal))),
            **{key: value.tolist() for key, value in scores.items()},
        }
        return choice, metadata
