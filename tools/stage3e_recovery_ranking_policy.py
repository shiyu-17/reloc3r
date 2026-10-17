#!/usr/bin/env python3
"""Stage 3E pre-action recovery/cost acquisition controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from tools.stage3b_policy import action_features, observation_dict
from tools.stage3c6_utility_policy import TERMINAL_EVIDENCE_SOURCES, UtilityComponentPolicy, policy_state_features


ACQUISITION = {"localize", "geometry", "memory"}
CHEAP = {"localize", "memory"}


class RecoveryCostRanker(nn.Module):
    """A shared action encoder with recovery-probability and cost heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.log_cost_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = actions.shape[0]
        encoded = self.encoder(torch.cat((state[None].expand(count, -1), actions), dim=-1))
        return self.recovery_head(encoded).squeeze(-1), self.log_cost_head(encoded).squeeze(-1)


@dataclass
class RecoveryCostRankingPolicy:
    """Frozen terminal control plus one explicit acquisition score P-lambda*C."""

    model: RecoveryCostRanker
    base: UtilityComponentPolicy
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    lambda_cost: float
    geometry_margin_threshold: float
    device: torch.device

    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "cpu") -> "RecoveryCostRankingPolicy":
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("family") != "stage3e_recovery_cost_ranking":
            raise ValueError("not a Stage 3E recovery/cost ranking checkpoint")
        config = payload["model_config"]
        model = RecoveryCostRanker(**config)
        model.load_state_dict(payload["model_state"]); model.to(device).eval()
        norm = payload["normalizer"]
        return cls(
            model=model, base=UtilityComponentPolicy.load(Path(payload["base_checkpoint"])),
            state_mean=np.asarray(norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(norm["state_std"], dtype=np.float32),
            action_mean=np.asarray(norm["action_mean"], dtype=np.float32),
            action_std=np.asarray(norm["action_std"], dtype=np.float32),
            lambda_cost=float(payload["lambda_cost"]),
            geometry_margin_threshold=float(payload.get("geometry_margin_threshold", 0.0)), device=torch.device(device),
        )

    def score_acquisitions(self, observation: Any, actions: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        state = (policy_state_features(observation, self.base.feature_level) - self.state_mean) / self.state_std
        encoded = np.stack([(action_features(observation, action) - self.action_mean) / self.action_std
                            for action in actions]).astype(np.float32)
        with torch.no_grad():
            logits, log_cost = self.model(torch.from_numpy(state.astype(np.float32)).to(self.device),
                                          torch.from_numpy(encoded).to(self.device))
        recovery = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        cost = np.expm1(log_cost.cpu().numpy().astype(np.float64)).clip(0.0, None)
        return {"recovery_probability": recovery, "predicted_incremental_cost_s": cost,
                "explicit_advantage": recovery - self.lambda_cost * cost}

    def choose(self, observation: Any, candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, dict[str, Any]]:
        # The frozen Stage 3D component controller retains all verifier and
        # terminal/admission decisions.  Stage 3E is only allowed to rank a
        # Geometry/cheap acquisition choice after continuation is selected.
        base_scores = self.base.score_actions(observation, candidates)["expected_utility"]
        kinds = [str(action.get("kind")) for action in candidates]
        masked_scores = base_scores.copy()
        obs = observation_dict(observation); candidate = obs.get("candidate") or {}
        terminal_eligible = (bool(candidate.get("pose_available")) and bool(candidate.get("verified")) and
                             str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES)
        masked = []
        for index, kind in enumerate(kinds):
            if kind == "accept" and not terminal_eligible:
                masked_scores[index] = -np.inf; masked.append(index)
        base_choice = int(np.argmax(masked_scores))
        acquisitions = [i for i, kind in enumerate(kinds) if kind in ACQUISITION]
        terminals = [i for i, kind in enumerate(kinds)
                     if kind in {"accept", "defer", "reject"} and (kind != "accept" or terminal_eligible)]
        metadata = {
            "controller": "stage3e_recovery_cost_ranking", "feature_level": self.base.feature_level,
            "terminal_eligible": terminal_eligible, "masked_action_indices": masked,
            "base_choice": base_choice,
        }
        if not acquisitions:
            return base_choice, {**metadata, "reason": "no_acquisition", "ranking_applied": False}
        best_acquisition = max(acquisitions, key=lambda i: base_scores[i])
        best_terminal = None if not terminals else max(terminals, key=lambda i: base_scores[i])
        if best_terminal is not None and base_scores[best_terminal] >= base_scores[best_acquisition]:
            return base_choice, {**metadata, "reason": "frozen_terminal_or_admission", "ranking_applied": False}
        if not any(kinds[i] == "geometry" for i in acquisitions) or not any(kinds[i] in CHEAP for i in acquisitions):
            return base_choice, {**metadata, "reason": "no_geometry_cheap_pair", "ranking_applied": False}
        action_rows = [candidates[i] for i in acquisitions]
        scores = self.score_acquisitions(observation, action_rows)
        local_choice = int(np.argmax(scores["explicit_advantage"]))
        geometry = [j for j, index in enumerate(acquisitions) if kinds[index] == "geometry"]
        cheap = [j for j, index in enumerate(acquisitions) if kinds[index] in CHEAP]
        geometry_margin = None
        if geometry and cheap:
            best_geometry = max(geometry, key=lambda j: scores["explicit_advantage"][j])
            best_cheap = max(cheap, key=lambda j: scores["explicit_advantage"][j])
            geometry_margin = float(scores["explicit_advantage"][best_geometry] - scores["explicit_advantage"][best_cheap])
            if local_choice == best_geometry and geometry_margin < self.geometry_margin_threshold:
                local_choice = best_cheap
        choice = acquisitions[local_choice]
        return choice, {**metadata, "reason": "explicit_recovery_cost_ranking", "ranking_applied": True,
                        "acquisition_indices": acquisitions, "selected_acquisition_index": choice,
                        "geometry_margin": geometry_margin, "geometry_margin_threshold": self.geometry_margin_threshold,
                        **{key: value.tolist() for key, value in scores.items()}}
