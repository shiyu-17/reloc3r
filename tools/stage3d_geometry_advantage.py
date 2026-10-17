#!/usr/bin/env python3
"""Deployable direct Geometry-vs-cheap advantage controller for Stage 3D."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from tools.stage3b_policy import action_features, observation_dict
from tools.stage3c6_utility_policy import (TERMINAL_EVIDENCE_SOURCES,
                                            UtilityComponentPolicy,
                                            policy_state_features)
from tools.train_stage3c_xgboost_diagnostic import xgboost


CHEAP = {"localize", "memory"}


def _candidate_pairs(base: UtilityComponentPolicy, observation: Any,
                     actions: Sequence[Mapping[str, Any]]) -> tuple[int, int, np.ndarray] | None:
    """Choose the deployable Geometry/cheap pair using only frozen base scores."""
    scores = base.score_actions(observation, actions)["expected_utility"]
    kinds = [str(action.get("kind")) for action in actions]
    geometry = [index for index, kind in enumerate(kinds) if kind == "geometry"]
    cheap = [index for index, kind in enumerate(kinds) if kind in CHEAP]
    if not geometry or not cheap:
        return None
    return (max(geometry, key=lambda index: scores[index]),
            max(cheap, key=lambda index: scores[index]), scores)


def pair_features(base: UtilityComponentPolicy, observation: Any,
                  actions: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, int, int, np.ndarray] | None:
    """Feature a Geometry/cheap pair without Oracle or post-action information."""
    pair = _candidate_pairs(base, observation, actions)
    if pair is None:
        return None
    geometry, cheap, scores = pair
    state = policy_state_features(observation, base.feature_level)
    geom_action = action_features(observation, actions[geometry])
    cheap_action = action_features(observation, actions[cheap])
    values = np.concatenate((state, geom_action, cheap_action,
                             np.asarray([scores[geometry], scores[cheap], scores[geometry] - scores[cheap]],
                                        dtype=np.float32))).astype(np.float32)
    return values, geometry, cheap, scores


@dataclass
class GeometryAdvantagePolicy:
    """Terminal/admission remains frozen; only acquisition Geometry ranking changes."""

    base: UtilityComponentPolicy
    classifier: Any
    regressor: Any
    probability_threshold: float
    advantage_threshold: float
    summary: Dict[str, Any]

    @classmethod
    def load(cls, model_dir: Path) -> "GeometryAdvantagePolicy":
        model_dir = Path(model_dir)
        summary = json.loads((model_dir / "training_summary.json").read_text(encoding="utf-8"))
        if summary.get("family") != "stage3d_geometry_advantage":
            raise ValueError("not a Stage 3D Geometry advantage controller")
        base = UtilityComponentPolicy.load(Path(summary["base_checkpoint"]))
        xgb = xgboost()
        classifier = xgb.XGBClassifier(); classifier.load_model(model_dir / "geometry_better_classifier.json")
        regressor = xgb.XGBRegressor(); regressor.load_model(model_dir / "geometry_advantage_regressor.json")
        calibration = summary["calibration"]
        return cls(base=base, classifier=classifier, regressor=regressor,
                   probability_threshold=float(calibration["probability_threshold"]),
                   advantage_threshold=float(calibration["advantage_threshold"]), summary=summary)

    def choose(self, observation: Any,
               candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        # Preserve the frozen utility controller for terminal safety/admission.
        base_scores = self.base.score_actions(observation, candidates)["expected_utility"]
        kinds = [str(action.get("kind")) for action in candidates]
        masked_scores = base_scores.copy()
        acquisition = [index for index, kind in enumerate(kinds) if kind in {"localize", "geometry", "memory"}]
        terminal = [index for index, kind in enumerate(kinds) if kind in {"accept", "defer", "reject"}]
        obs = observation_dict(observation)
        candidate = obs.get("candidate") or {}
        terminal_eligible = (bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
                             and str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES)
        masked = []
        for index, kind in enumerate(kinds):
            if kind == "accept" and not terminal_eligible:
                masked_scores[index] = -np.inf
                masked.append(index)
        base_choice = int(np.argmax(masked_scores))
        terminal = [index for index in terminal if kinds[index] != "accept" or terminal_eligible]
        metadata = {
            "feature_level": self.base.feature_level,
            "terminal_eligible": terminal_eligible,
            "masked_action_indices": masked,
            "best_acquisition_utility": (None if not acquisition else float(max(masked_scores[index] for index in acquisition))),
            "best_terminal_utility": (None if not terminal else float(max(masked_scores[index] for index in terminal))),
        }
        best_acquisition = None if not acquisition else max(acquisition, key=lambda index: base_scores[index])
        best_terminal = None if not terminal else max(terminal, key=lambda index: base_scores[index])
        # Terminal/admission is intentionally not retrained in this stage.
        if best_acquisition is None or (best_terminal is not None and
                                        base_scores[best_terminal] >= base_scores[best_acquisition]):
            return base_choice, {**metadata, "controller": "stage3d_geometry_advantage",
                                 "reason": "frozen_terminal_or_admission", "advantage_applied": False}
        geometry = [index for index, kind in enumerate(kinds) if kind == "geometry"]
        cheap = [index for index, kind in enumerate(kinds) if kind in CHEAP]
        if not geometry or not cheap:
            return base_choice, {**metadata, "controller": "stage3d_geometry_advantage",
                                 "reason": "no_geometry_cheap_pair", "advantage_applied": False}
        geometry = max(geometry, key=lambda index: base_scores[index])
        cheap = max(cheap, key=lambda index: base_scores[index])
        state = policy_state_features(observation, self.base.feature_level)
        features = np.concatenate((state, action_features(observation, candidates[geometry]),
                                   action_features(observation, candidates[cheap]),
                                   np.asarray([base_scores[geometry], base_scores[cheap],
                                               base_scores[geometry] - base_scores[cheap]], dtype=np.float32)))
        probability = float(self.classifier.predict_proba(features[None, :])[:, 1][0])
        advantage = float(self.regressor.predict(features[None, :])[0])
        choose_geometry = (probability >= self.probability_threshold and
                           advantage >= self.advantage_threshold)
        choice = geometry if choose_geometry else cheap
        return choice, {**metadata, "controller": "stage3d_geometry_advantage",
                        "reason": "pairwise_geometry_advantage", "advantage_applied": True,
                        "base_choice": int(base_choice), "geometry_index": int(geometry), "cheap_index": int(cheap),
                        "geometry_better_probability": probability, "geometry_advantage": advantage,
                        "probability_threshold": self.probability_threshold,
                        "advantage_threshold": self.advantage_threshold}
