#!/usr/bin/env python3
"""Select recovery/boundary states for Stage 3D targeted data collection.

The output remains grouped by state, so XGBRanker still sees natural within-
state action comparisons.  Positive/boundary recovery states are oversampled
only for training; evaluation must use the unfiltered source distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _advantage(action: Mapping[str, Any]) -> float:
    return float(action.get("oracle_advantage", 0.0))


def geometry_margin(actions: List[Mapping[str, Any]]) -> float | None:
    geometry = [item for item in actions if str(item.get("kind")) == "geometry"]
    cheap = [item for item in actions if str(item.get("kind")) in {"localize", "memory"}]
    if not geometry or not cheap:
        return None
    return max(map(_advantage, geometry)) - max(map(_advantage, cheap))


def classify(row: Mapping[str, Any], boundary: float) -> str | None:
    observation = row.get("observation") or {}
    candidate = observation.get("candidate") or {}
    margin = geometry_margin(row.get("actions", []))
    if margin is None:
        return None
    if not candidate:
        return "root_background"
    if margin >= 0.0:
        return "geometry_positive_recovery"
    if abs(margin) <= boundary:
        return "geometry_boundary_recovery"
    if str(candidate.get("source")) == "memory":
        return "memory_negative_recovery"
    return "recovery_background"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-margin", type=float, default=0.15)
    parser.add_argument("--negative-keep-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if not 0.0 <= args.negative_keep_probability <= 1.0 or args.boundary_margin < 0:
        parser.error("invalid sampling parameters")
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        reason = classify(row, args.boundary_margin)
        if reason is None:
            continue
        if reason in {"root_background", "recovery_background"} and rng.random() > args.negative_keep_probability:
            continue
        output = dict(row)
        output["stage3d_sampling_reason"] = reason
        output["stage3d_geometry_margin"] = geometry_margin(row.get("actions", []))
        # Keep a bounded multiplier so a targeted corpus cannot erase the
        # natural action prior when used with weighted training.
        output["stage3d_training_weight"] = 3.0 if reason == "geometry_positive_recovery" else (
            2.0 if reason == "geometry_boundary_recovery" else 1.0)
        selected.append(output); reasons[reason] += 1
    selected.sort(key=lambda row: (str(row.get("sequence")), int(row.get("frame_index", 0)), str(row.get("state_id", ""))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "stage3d_recovery_opportunities.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "purpose": "Stage 3D targeted recovery/opportunity corpus",
        "source": str(args.input), "states": len(selected), "reason_counts": dict(sorted(reasons.items())),
        "boundary_margin": args.boundary_margin,
        "negative_keep_probability": args.negative_keep_probability,
        "training_only_weights": "targeted weights are bounded and must not be used for natural-distribution evaluation",
        "online_boundary": "selection uses evaluator-only Oracle labels after policy-visible observations; features are added separately with stage3d_features.py",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
