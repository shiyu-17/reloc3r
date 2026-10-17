#!/usr/bin/env python3
"""Structured policy/value models for Stage 3B.

The module deliberately consumes only policy-visible structured evidence.  It
does not load images, query GT, Oracle labels, or reference identities at
inference time.  Oracle utility is used only by :mod:`train_stage3b_policy`
to supervise the policy and value heads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from tools.active_evidence_protocol import legal_actions as protocol_legal_actions
from tools.active_evidence_protocol import is_deployment_legal as protocol_is_deployment_legal


ACTION_KINDS = ("localize", "geometry", "memory", "accept", "reject", "defer")
SOURCE_KINDS = ("reloc3r_online", "roma_online_geometry", "memory")
STATE_DIM = 26
ACTION_DIM = 12


def observation_dict(observation: Any) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    method = getattr(observation, "as_dict", None)
    if callable(method):
        return method()
    raise TypeError("observation must be a mapping or expose as_dict()")


def _number(value: Any, default: float = 0.0, clip: float = 1000.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return float(np.clip(result, -clip, clip))


def _groups(observation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    groups = observation.get("frame", {}).get("reference_groups", [])
    return [item for item in groups if len(item.get("reference_indices", [])) >= 2]


def state_features(observation: Any) -> np.ndarray:
    """Fixed-dimensional, reference-ID-free feature vector."""
    obs = observation_dict(observation)
    frame = obs.get("frame", {})
    state = obs.get("state", {})
    candidate = obs.get("candidate") or {}
    evidence = candidate.get("evidence", {})
    verification = candidate.get("verification", {})
    groups = _groups(obs)
    scores = [_number(group.get("mean_retrieval_score")) for group in groups]
    spreads = [_number(group.get("map_spread_m")) for group in groups]
    source = str(candidate.get("source", ""))
    budget = max(_number(state.get("budget_s"), 30.0), 1e-6)
    values = [
        min(len(groups), 8) / 8.0,
        max(scores, default=0.0),
        float(np.mean(scores)) if scores else 0.0,
        float(np.std(scores)) if scores else 0.0,
        (scores[0] - scores[1]) if len(scores) >= 2 else 0.0,
        max(spreads, default=0.0) / 30.0,
        float(candidate != {}),
        float(bool(candidate.get("pose_available"))),
        float(bool(candidate.get("verified"))),
        *[float(source == name) for name in SOURCE_KINDS],
        _number(evidence.get("num_references")) / 8.0,
        _number(evidence.get("translation_dispersion_m")) / 10.0,
        _number(evidence.get("rotation_dispersion_deg")) / 45.0,
        _number(evidence.get("inliers")) / 100.0,
        _number(evidence.get("reprojection_error_px")) / 20.0,
        _number(verification.get("total_risk", verification.get("independent_risk"))) / 10.0,
        _number(verification.get("confidence")),
        float(bool(state.get("accepted_pose_available"))),
        _number(state.get("anchor_age")) / 30.0,
        _number(state.get("reject_streak")) / 10.0,
        _number(state.get("memory_size")) / 20.0,
        _number(state.get("frame_action_count")) / 8.0,
        _number(state.get("frame_tool_cost_s")) / budget,
        _number(state.get("remaining_budget_s"), budget) / budget,
    ]
    output = np.asarray(values, dtype=np.float32)
    assert output.shape == (STATE_DIM,), output.shape
    return output


def action_features(observation: Any, action: Mapping[str, Any]) -> np.ndarray:
    """Action descriptors use group statistics, never reference IDs."""
    obs = observation_dict(observation)
    kind = str(action.get("kind", ""))
    # Reject and Defer used to be duplicate frame-ending actions.  Keep the
    # legacy feature slot/checkpoint shape readable, but canonicalize all new
    # negative terminal decisions to Defer.
    if kind == "reject":
        kind = "defer"
    references = tuple(int(item) for item in action.get("references", []))
    groups = _groups(obs)
    matched_index = next((index for index, group in enumerate(groups)
                          if tuple(int(item) for item in group.get("reference_indices", [])) == references), None)
    group = groups[matched_index] if matched_index is not None else {}
    raw_scores = group.get("retrieval_scores", [])
    scores = [_number(value) for value in raw_scores]
    values = [
        *[float(kind == name) for name in ACTION_KINDS],
        float(bool(action.get("is_terminal_action", kind in {"accept", "reject", "defer"}))),
        min(len(references), 8) / 8.0,
        _number(group.get("mean_retrieval_score")),
        float(np.std(scores)) if scores else 0.0,
        _number(group.get("map_spread_m")) / 30.0,
        ((matched_index + 1) / 8.0) if matched_index is not None else 0.0,
    ]
    output = np.asarray(values, dtype=np.float32)
    assert output.shape == (ACTION_DIM,), output.shape
    return output


def legal_actions(observation: Any, remaining_group_indices: Sequence[int], *,
                  geometry_used: bool, memory_used: bool = False,
                  allow_acquisition: bool = True) -> List[Dict[str, Any]]:
    """Compatibility wrapper for the versioned shared action contract."""
    del memory_used
    return protocol_legal_actions(observation, remaining_group_indices,
                                  geometry_used=geometry_used,
                                  allow_acquisition=allow_acquisition)


def is_deployment_legal(observation: Any, action: Mapping[str, Any]) -> bool:
    """Apply the shared v2 deployment mask to offline labels."""
    return protocol_is_deployment_legal(observation, action)


class MLPPolicyValue(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 96):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        batch, count, _ = actions.shape
        expanded = states[:, None, :].expand(batch, count, states.shape[-1])
        hidden = self.backbone(torch.cat((expanded, actions), dim=-1))
        logits = self.policy_head(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class TransformerPolicyValue(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        batch, count, _ = actions.shape
        expanded = states[:, None, :].expand(batch, count, states.shape[-1])
        tokens = self.token(torch.cat((expanded, actions), dim=-1))
        hidden = self.encoder(tokens, src_key_padding_mask=~mask)
        logits = self.policy_head(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def create_model(model_type: str, config: Mapping[str, Any]) -> nn.Module:
    hidden_dim = int(config.get("hidden_dim", 96))
    if model_type == "mlp":
        return MLPPolicyValue(hidden_dim=hidden_dim)
    if model_type == "transformer":
        return TransformerPolicyValue(hidden_dim=hidden_dim,
                                      layers=int(config.get("layers", 2)),
                                      heads=int(config.get("heads", 4)))
    raise ValueError("unknown model_type: " + model_type)


@dataclass
class StructuredPolicy:
    model: nn.Module
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    value_mean: float
    value_std: float
    device: torch.device
    decision: str = "policy"

    @classmethod
    def load(cls, checkpoint_path: Path, *, device: str = "cpu", decision: str = "policy") -> "StructuredPolicy":
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = create_model(payload["model_type"], payload["model_config"])
        model.load_state_dict(payload["model_state"])
        model.to(device).eval()
        normalizer = payload["normalizer"]
        return cls(model=model,
                   state_mean=np.asarray(normalizer["state_mean"], dtype=np.float32),
                   state_std=np.asarray(normalizer["state_std"], dtype=np.float32),
                   action_mean=np.asarray(normalizer["action_mean"], dtype=np.float32),
                   action_std=np.asarray(normalizer["action_std"], dtype=np.float32),
                   value_mean=float(normalizer["value_mean"]),
                   value_std=float(normalizer["value_std"]),
                   device=torch.device(device), decision=decision)

    def choose(self, observation: Any, candidates: Sequence[Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        state = (state_features(observation) - self.state_mean) / self.state_std
        actions = np.stack([(action_features(observation, item) - self.action_mean) / self.action_std
                            for item in candidates])
        with torch.no_grad():
            states = torch.from_numpy(state)[None].to(self.device)
            action_tensor = torch.from_numpy(actions)[None].to(self.device)
            mask = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device)
            logits, values = self.model(states, action_tensor, mask)
            terminal_only = all(bool(item.get("is_terminal_action")) or
                                str(item.get("kind")) in {"accept", "defer", "reject"}
                                for item in candidates)
            use_value = self.decision == "value" or (
                self.decision == "terminal_value" and terminal_only)
            scores = values[0] if use_value else logits[0]
            choice = int(torch.argmax(scores).item())
        return choice, {
            "decision": self.decision,
            "score_source": "value" if use_value else "policy",
            "terminal_only_candidates": terminal_only,
            "policy_logits": logits[0].detach().cpu().tolist(),
            "predicted_q": (values[0].detach().cpu().numpy() * self.value_std + self.value_mean).tolist(),
        }


def save_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
