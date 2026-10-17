#!/usr/bin/env python3
"""Pre-Geometry feature ladder for Stage 3D.

Every feature in this module is available before the *next* strong-geometry
action.  Optional backend-produced groups (``cheap_correspondence``,
``temporal_consistency`` and ``appearance_change``) are read only from the
current observation.  RoMa/PnP/epipolar/inlier/reprojection fields are rejected
instead of silently becoming leakage.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tools.stage3b_policy import observation_dict, _groups, _number


FEATURE_LEVELS = ("f0", "f1", "f2", "f3")
FORBIDDEN_PRE_GEOMETRY_KEYS = frozenset({
    "roma", "roma_match", "roma_matches", "geometry_backend", "strong_geometry",
    "pnp", "inliers", "reprojection", "epipolar", "triangulated", "geometry_inlier",
})


def _safe(value: Any, default: float = 0.0) -> float:
    return _number(value, default=default, clip=1e9)


def _entropy(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.maximum(values - values.min(), 0.0) + 1e-6
    values /= values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(max(2, len(values))))


def _assert_pre_geometry(value: Any, path: str = "observation") -> None:
    """Fail closed if an optional feature payload contains post-Geometry data."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PRE_GEOMETRY_KEYS or any(
                    token in normalized for token in ("roma", "reprojection", "epipolar", "triangulat", "pnp_inlier")):
                raise ValueError(f"post-Geometry field is not a Stage 3D feature: {path}.{key}")
            _assert_pre_geometry(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_pre_geometry(child, f"{path}[{index}]")


def strip_post_geometry_fields(value: Any) -> Any:
    """Project an online state onto the pre-Geometry observation boundary.

    A state reached *after* a RoMa action may still contain the receipt and
    verifier evidence from that action.  The next acquisition decision must
    not consume those post-action fields, but it must remain possible to make
    a second decision.  Training/evaluation builders still call
    :func:`_assert_pre_geometry` directly; this projection is only for the
    explicit online state transition at the controller boundary.
    """
    if isinstance(value, Mapping):
        return {str(key): strip_post_geometry_fields(child)
                for key, child in value.items()
                if str(key).lower() not in FORBIDDEN_PRE_GEOMETRY_KEYS
                and not any(token in str(key).lower()
                            for token in ("roma", "reprojection", "epipolar", "triangulat", "pnp_inlier"))}
    if isinstance(value, list):
        return [strip_post_geometry_fields(child) for child in value]
    if isinstance(value, tuple):
        return tuple(strip_post_geometry_fields(child) for child in value)
    return value


def _base_features(obs: Mapping[str, Any]) -> np.ndarray:
    """F0: retrieval, verifier, temporal state and budget, without RoMa fields."""
    frame, state = obs.get("frame", {}), obs.get("state", {})
    candidate = obs.get("candidate") or {}
    evidence = candidate.get("evidence") or {}
    source = str(candidate.get("source", ""))
    verification = candidate.get("verification") or {}
    groups = _groups(obs)
    scores = [_safe(group.get("mean_retrieval_score")) for group in groups]
    spreads = [_safe(group.get("map_spread_m", group.get("map_spread"))) for group in groups]
    budget = max(_safe(state.get("budget_s"), 30.0), 1e-6)
    values = [
        min(len(groups), 8) / 8.0,
        max(scores, default=0.0), float(np.mean(scores)) if scores else 0.0,
        float(np.std(scores)) if scores else 0.0,
        (scores[0] - scores[1]) if len(scores) >= 2 else 0.0,
        float(candidate != {}), float(bool(candidate.get("pose_available"))),
        float(bool(candidate.get("verified"))),
        *[float(source == name) for name in ("reloc3r_online", "roma_online_geometry", "memory")],
        # These are already emitted by the cheap localizer/retrieval stage.
        # They were inadvertently omitted from the original F0, even though
        # they are legal pre-Geometry evidence on most recovery states.
        _safe(evidence.get("num_references")) / 8.0,
        _safe(evidence.get("translation_dispersion_m")) / 10.0,
        _safe(evidence.get("rotation_dispersion_deg")) / 45.0,
        max(spreads, default=0.0) / 30.0,
        _safe(verification.get("total_risk", verification.get("independent_risk"))) / 10.0,
        _safe(verification.get("confidence")),
        float(bool(state.get("accepted_pose_available"))),
        _safe(state.get("anchor_age")) / 30.0,
        _safe(state.get("reject_streak")) / 10.0,
        _safe(state.get("memory_size")) / 20.0,
        _safe(state.get("frame_action_count")) / 8.0,
        _safe(state.get("frame_tool_cost_s")) / budget,
        _safe(state.get("remaining_budget_s"), budget) / budget,
    ]
    return np.asarray(values, dtype=np.float32)


def _f1_features(obs: Mapping[str, Any]) -> list[float]:
    """Cheap retrieval/correspondence proxies available after localize."""
    frame, candidate = obs.get("frame", {}), obs.get("candidate") or {}
    groups = _groups(obs)
    scores = [_safe(x) for x in frame.get("retrieval_scores", [])]
    group_means = [_safe(g.get("mean_retrieval_score")) for g in groups]
    cheap = candidate.get("evidence", {}).get("cheap_correspondence") or {}
    _assert_pre_geometry(cheap, "candidate.evidence.cheap_correspondence")
    # If a backend exposes actual cheap matches, use them; otherwise retain a
    # deterministic retrieval-only proxy and an availability mask.
    actual = [cheap.get(key) for key in ("coverage_ratio", "spatial_entropy", "concentration",
                                          "valid_matches", "median_pixel_radius")]
    has_actual = float(any(value is not None for value in actual))
    return [
        _entropy(scores), float(np.max(scores) - np.min(scores)) if scores else 0.0,
        float(np.mean(group_means)) if group_means else 0.0,
        float(np.std(group_means)) if group_means else 0.0,
        _safe(cheap.get("coverage_ratio")) if has_actual else 0.0,
        _safe(cheap.get("spatial_entropy")) if has_actual else 0.0,
        _safe(cheap.get("concentration")) if has_actual else 0.0,
        _safe(cheap.get("valid_matches")) / 1000.0 if has_actual else 0.0,
        _safe(cheap.get("median_pixel_radius")) / 1000.0 if has_actual else 0.0,
        has_actual,
    ]


def _f2_features(obs: Mapping[str, Any]) -> list[float]:
    """Temporal/anchor consistency that exists before a new Geometry call."""
    candidate = obs.get("candidate") or {}
    verification = candidate.get("verification") or {}
    temporal = candidate.get("evidence", {}).get("temporal_consistency") or {}
    _assert_pre_geometry(temporal, "candidate.evidence.temporal_consistency")
    return [
        _safe(verification.get("previous_state_risk")) / 10.0,
        _safe(verification.get("state_trust")),
        _safe(temporal.get("translation_disagreement_m")) / 10.0,
        _safe(temporal.get("rotation_disagreement_deg")) / 90.0,
        _safe(temporal.get("anchor_retrieval_consistency")),
        float(any(value is not None for value in temporal.values())),
    ]


def _f3_features(obs: Mapping[str, Any]) -> list[float]:
    """Appearance/change cues, with an explicit missing-feature mask."""
    frame = obs.get("frame", {})
    candidate = obs.get("candidate") or {}
    change = (frame.get("appearance_change") or
              (candidate.get("evidence", {}).get("appearance_change") or {}))
    _assert_pre_geometry(change, "appearance_change")
    fields = ("query_reference_cosine", "change_score", "dynamic_ratio", "occlusion_ratio",
              "viewpoint_diversity", "brightness_change")
    present = float(any(change.get(field) is not None for field in fields))
    return [_safe(change.get(field)) for field in fields] + [present]


def feature_vector(observation: Any, level: str = "f0") -> np.ndarray:
    """Return the fixed-dimensional state vector for one ladder level."""
    if level not in FEATURE_LEVELS:
        raise ValueError(f"feature level must be one of {FEATURE_LEVELS}")
    obs = observation_dict(observation)
    # The same observation may later be reused for an action decision.  Fail
    # closed if a caller tries to train/evaluate a pre-Geometry feature vector
    # from a candidate whose evidence already contains strong-geometry output.
    candidate = obs.get("candidate") or {}
    _assert_pre_geometry(candidate.get("evidence") or {}, "candidate.evidence")
    base = _base_features(obs)
    values = [base]
    if level in {"f1", "f2", "f3"}:
        values.append(np.asarray(_f1_features(obs), dtype=np.float32))
    if level in {"f2", "f3"}:
        values.append(np.asarray(_f2_features(obs), dtype=np.float32))
    if level == "f3":
        values.append(np.asarray(_f3_features(obs), dtype=np.float32))
    return np.concatenate(values).astype(np.float32)


def feature_metadata(level: str) -> Dict[str, Any]:
    if level not in FEATURE_LEVELS:
        raise ValueError(f"feature level must be one of {FEATURE_LEVELS}")
    dimensions = {"f0": len(_base_features({})), "f1": len(_base_features({})) + 10,
                  "f2": len(_base_features({})) + 16, "f3": len(_base_features({})) + 23}
    return {"schema_version": 2, "level": level, "dimension": dimensions[level],
            "online_boundary": "all fields are available before the next Geometry action; post-Geometry fields fail closed"}
