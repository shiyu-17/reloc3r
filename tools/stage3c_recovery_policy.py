#!/usr/bin/env python3
"""Cost-aware recovery policy for Stage 3C.

Unlike the Stage 3B scalar-Q policy, this policy separately predicts the
finite-horizon probability of recovering a correct terminal pose and the
additional tool cost needed to do so.  An explicit controller combines them;
GT-derived targets never enter the online observation or action selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from tools.stage3b_policy import ACTION_DIM, STATE_DIM, action_features, state_features


ACQUISITION_KINDS = frozenset({"localize", "geometry", "memory"})
# A memory entry is retrieval context, not independently re-established pose
# evidence.  It may be stale (or, after a prior error, polluted), so it cannot
# be accepted directly.  A fresh localizer or geometry receipt must establish
# terminal evidence first.
TERMINAL_EVIDENCE_SOURCES = frozenset({"reloc3r_online", "roma_online_geometry"})


class RecoveryCostMLP(nn.Module):
    """Shared structured encoder with calibrated recovery and log-cost heads."""

    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.log_cost_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        batch, count, _ = actions.shape
        state = states[:, None, :].expand(batch, count, states.shape[-1])
        hidden = self.backbone(torch.cat((state, actions), dim=-1))
        recovery_logits = self.recovery_head(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        log_cost = self.log_cost_head(hidden).squeeze(-1)
        return recovery_logits, log_cost


def create_model(config: Mapping[str, Any]) -> RecoveryCostMLP:
    return RecoveryCostMLP(hidden_dim=int(config.get("hidden_dim", 96)))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


@dataclass
class RecoveryCostPolicy:
    model: RecoveryCostMLP
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    lambda_cost: float
    defer_threshold: float
    terminal_continue_margin: float
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: Path, *, device: str = "cpu") -> "RecoveryCostPolicy":
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("family") != "stage3c_recovery":
            raise ValueError("not a Stage 3C recovery-policy checkpoint")
        model = create_model(payload["model_config"])
        model.load_state_dict(payload["model_state"])
        model.to(device).eval()
        normalizer = payload["normalizer"]
        controller = payload["controller"]
        return cls(
            model=model,
            state_mean=np.asarray(normalizer["state_mean"], dtype=np.float32),
            state_std=np.asarray(normalizer["state_std"], dtype=np.float32),
            action_mean=np.asarray(normalizer["action_mean"], dtype=np.float32),
            action_std=np.asarray(normalizer["action_std"], dtype=np.float32),
            lambda_cost=float(controller["lambda_cost"]),
            defer_threshold=float(controller["defer_threshold"]),
            terminal_continue_margin=float(controller["terminal_continue_margin"]),
            device=torch.device(device),
        )

    def score_actions(self, observation: Any, actions: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        """Return online-predictable recovery, incremental cost and advantage."""
        if not actions:
            raise ValueError("actions must be non-empty")
        state = (state_features(observation) - self.state_mean) / self.state_std
        encoded_actions = np.stack([
            (action_features(observation, action) - self.action_mean) / self.action_std
            for action in actions
        ]).astype(np.float32)
        with torch.no_grad():
            logits, log_cost = self.model(
                torch.from_numpy(state.astype(np.float32))[None].to(self.device),
                torch.from_numpy(encoded_actions)[None].to(self.device),
                torch.ones((1, len(actions)), dtype=torch.bool, device=self.device),
            )
        recovery = torch.sigmoid(logits[0]).detach().cpu().numpy().astype(np.float64)
        # Training target is log1p(incremental seconds), so this is always non-negative.
        cost = np.expm1(log_cost[0].detach().cpu().numpy().astype(np.float64)).clip(0.0, None)
        advantage = recovery - self.lambda_cost * cost
        return {"recovery_probability": recovery, "predicted_incremental_cost_s": cost,
                "advantage": advantage}

    def choose(self, observation: Any, candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """Verifier-gated terminal control plus explicit cost-aware acquisition.

        Defer has an explicit zero recovery/zero new-cost baseline.  A verified
        candidate may be accepted, but the controller can continue only where
        its predicted recovery advantage exceeds verifier confidence by a
        frozen margin.  This prevents automatic verifier->Accept while keeping
        the safety evidence entirely policy-visible.
        """
        acquisition_indices = [index for index, action in enumerate(candidates)
                               if str(action.get("kind")) in ACQUISITION_KINDS]
        accept_indices = [index for index, action in enumerate(candidates)
                          if str(action.get("kind")) == "accept"]
        defer_indices = [index for index, action in enumerate(candidates)
                         if str(action.get("kind")) in {"defer", "reject"}]
        candidate = (observation.as_dict() if hasattr(observation, "as_dict") else observation).get("candidate") or {}
        verified = bool(candidate.get("pose_available")) and bool(candidate.get("verified"))
        terminal_eligible = verified and str(candidate.get("source")) in TERMINAL_EVIDENCE_SOURCES
        if not acquisition_indices:
            if accept_indices and terminal_eligible:
                return accept_indices[0], {"controller": "verifier_gated_terminal",
                                           "reason": "verified_no_remaining_acquisition"}
            if defer_indices:
                return defer_indices[0], {
                    "controller": "defer_baseline",
                    "reason": ("memory_requires_fresh_evidence" if verified and not terminal_eligible
                               else "no_acquisition"),
                }
            raise ValueError("candidate set has neither acquisition nor terminal action")

        acquisition = [candidates[index] for index in acquisition_indices]
        scores = self.score_actions(observation, acquisition)
        local = int(np.argmax(scores["advantage"]))
        chosen_index = acquisition_indices[local]
        chosen_advantage = float(scores["advantage"][local])
        confidence = float(np.clip(_number((candidate.get("verification") or {}).get("confidence")), 0.0, 1.0))

        if terminal_eligible and accept_indices:
            # Continuing must beat the verifier-supported terminal candidate,
            # not merely be positive relative to Defer.
            if chosen_advantage <= confidence + self.terminal_continue_margin:
                return accept_indices[0], {
                    "controller": "verifier_gated_terminal", "reason": "accept_beats_continuation",
                    "verifier_confidence": confidence, "best_acquisition_advantage": chosen_advantage,
                    **{key: value.tolist() for key, value in scores.items()},
                }
        elif defer_indices and chosen_advantage <= self.defer_threshold:
            return defer_indices[0], {
                "controller": "cost_aware_advantage", "reason": "defer_beats_acquisition",
                "best_acquisition_advantage": chosen_advantage,
                **{key: value.tolist() for key, value in scores.items()},
            }

        return chosen_index, {
            "controller": "cost_aware_advantage", "reason": "acquire",
            "best_acquisition_advantage": chosen_advantage,
            **{key: value.tolist() for key, value in scores.items()},
        }
