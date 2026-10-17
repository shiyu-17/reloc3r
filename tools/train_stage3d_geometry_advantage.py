#!/usr/bin/env python3
"""Train a direct, pairwise Geometry-vs-best-cheap advantage model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.stage3b_policy import action_features
from tools.stage3c6_utility_policy import (UtilityComponentPolicy, compose_utility,
                                            policy_state_features)
from tools.train_stage3c_xgboost_diagnostic import xgboost
from tools.train_stage3c6_utility_controller import read_records, temporal_split


def records_to_pairs(records: list[Mapping[str, Any]], base: UtilityComponentPolicy):
    # Batch all frozen-component predictions.  Calling XGBoost once per state
    # makes dataset assembly dominate training time without changing labels.
    candidates, feature_rows, spans = [], [], []
    for row in records:
        actions = row["actions"]
        kinds = [str(action.get("kind")) for action in actions]
        geometry = [index for index, kind in enumerate(kinds) if kind == "geometry"]
        cheap = [index for index, kind in enumerate(kinds) if kind in {"localize", "memory"}]
        if not geometry or not cheap:
            continue
        state = policy_state_features(row["observation"], base.feature_level)
        begin = len(feature_rows)
        action_rows = [action_features(row["observation"], action) for action in actions]
        feature_rows.extend(np.concatenate((state, action)) for action in action_rows)
        spans.append((begin, len(feature_rows)))
        candidates.append((row, actions, geometry, cheap, state, action_rows))
    if not candidates:
        raise ValueError("no states have both Geometry and a cheap acquisition action")
    action_matrix = np.asarray(feature_rows, dtype=np.float32)
    components = {
        "correct_accept": base.models["correct_accept"].predict_proba(action_matrix)[:, 1],
        "false_accept": base.models["false_accept"].predict_proba(action_matrix)[:, 1],
        "memory_pollution": base.models["memory_pollution"].predict_proba(action_matrix)[:, 1],
        "incremental_cost_s": np.expm1(base.models["incremental_cost_s"].predict(action_matrix)).clip(0.0, None),
        "recovery_time_frames": np.expm1(base.models["recovery_time_frames"].predict(action_matrix)).clip(0.0, None),
    }
    expected = compose_utility(components, base.utility)
    values, margins, labels, weights, metadata = [], [], [], [], []
    for (row, actions, geometry_indices, cheap_indices, state, action_rows), (begin, end) in zip(candidates, spans):
        scores = expected[begin:end]
        geometry = max(geometry_indices, key=lambda index: scores[index])
        cheap = max(cheap_indices, key=lambda index: scores[index])
        features = np.concatenate((state, action_rows[geometry], action_rows[cheap],
                                   np.asarray([scores[geometry], scores[cheap], scores[geometry] - scores[cheap]],
                                              dtype=np.float32))).astype(np.float32)
        advantage = float(actions[geometry]["oracle_incremental_utility"] -
                          actions[cheap]["oracle_incremental_utility"])
        values.append(features); margins.append(advantage); labels.append(float(advantage >= 0.0))
        targeted = bool(set(row.get("target_tags", [])) & {"geometry_optimal", "high_oracle_regret"})
        weights.append(2.0 if targeted else 1.0)
        metadata.append((str(row["sequence"]), int(row["frame_index"]), geometry, cheap))
    return (np.asarray(values, dtype=np.float32), np.asarray(margins, dtype=np.float32),
            np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float32), metadata)


def metrics(probability: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    sign = probability >= 0.5
    truth = target >= 0.0
    return {
        "states": int(len(target)), "positive_geometry_advantage": int(truth.sum()),
        "pairwise_sign_accuracy": float(np.mean(sign == truth)),
        "advantage_mae": float(np.mean(np.abs(prediction - target))),
        "advantage_correlation": (None if len(target) < 2 or np.std(prediction) == 0 or np.std(target) == 0
                                  else float(np.corrcoef(prediction, target)[0, 1])),
        "geometry_recall": None if not truth.any() else float(np.sum(sign & truth) / np.sum(truth)),
        "geometry_precision": None if not sign.any() else float(np.sum(sign & truth) / np.sum(sign)),
        "confusion": {"tp": int(np.sum(sign & truth)), "fp": int(np.sum(sign & ~truth)),
                      "tn": int(np.sum(~sign & ~truth)), "fn": int(np.sum(~sign & truth))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=240)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1777)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    base = UtilityComponentPolicy.load(args.base_checkpoint)
    records = read_records(args.dataset)
    train, validation, validation_frames = temporal_split(records, args.validation_fraction)
    tx, ty, tl, tw, _ = records_to_pairs(train, base)
    vx, vy, vl, _, _ = records_to_pairs(validation, base)
    xgb = xgboost()
    common = {"n_estimators": args.estimators, "max_depth": args.max_depth,
              "learning_rate": args.learning_rate, "subsample": 0.9, "colsample_bytree": 0.9,
              "random_state": args.seed, "tree_method": "hist", "device": args.device, "n_jobs": 4}
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    classifier.fit(tx, tl, sample_weight=tw)
    regressor = xgb.XGBRegressor(objective="reg:squarederror", **common)
    regressor.fit(tx, ty, sample_weight=tw)
    probability = classifier.predict_proba(vx)[:, 1]
    prediction = regressor.predict(vx)
    summary = {
        "schema_version": 1, "family": "stage3d_geometry_advantage",
        "dataset": str(args.dataset), "base_checkpoint": str(args.base_checkpoint),
        "feature_level": base.feature_level,
        "training": {"train_pairs": int(len(tx)), "validation_pairs": int(len(vx)),
                     "validation_frames": validation_frames, "seed": args.seed,
                     "estimators": args.estimators, "max_depth": args.max_depth,
                     "learning_rate": args.learning_rate, "targeted_positive_weight": 2.0},
        "calibration": {"probability_threshold": 0.5, "advantage_threshold": 0.0,
                        "selection": "fixed before fresh-split evaluation; no Stage 3D final frame is used"},
        "validation": metrics(probability, prediction, vy),
        "online_boundary": "pair features contain frozen controller scores, current observation and legal actions only; Oracle advantage is a training label only",
        "controller_boundary": "frozen component controller retains verifier/terminal/admission; this head only ranks Geometry against the frozen best cheap acquisition",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(args.output_dir / "geometry_better_classifier.json")
    regressor.save_model(args.output_dir / "geometry_advantage_regressor.json")
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
