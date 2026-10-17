#!/usr/bin/env python3
"""Train the resource-light structured PoseAgent verifier on scene splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.poseagent_style import StructuredVerifierFeatures


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("verifier dataset is empty")
    for row in rows:
        if row.get("offline_only") is not True:
            raise ValueError("verifier dataset must be explicitly marked offline_only")
    return rows


def matrix(rows: list[dict[str, Any]]):
    features = np.stack([StructuredVerifierFeatures.vector(row["observation"], row["candidate"])
                         for row in rows])
    success = np.asarray([bool(row["label"]["success"]) for row in rows], dtype=np.int64)
    translation = np.log1p(np.asarray([max(0.0, float(row["label"]["translation_error_m"])) for row in rows]))
    rotation = np.log1p(np.asarray([max(0.0, float(row["label"]["rotation_error_deg"])) for row in rows]))
    return features, success, translation, rotation


def threshold_at_far(probability: np.ndarray, success: np.ndarray, maximum_far: float) -> tuple[float, dict[str, float]]:
    candidates = sorted(set([0.0, 1.0, *[float(value) for value in probability]]))
    feasible = []
    for threshold in candidates:
        accepted = probability >= threshold
        count = int(accepted.sum())
        false = int(np.logical_and(accepted, success == 0).sum())
        far = false / count if count else 0.0
        recall = int(np.logical_and(accepted, success == 1).sum()) / max(1, int((success == 1).sum()))
        if far <= maximum_far:
            feasible.append((recall, -threshold, threshold, {"accepted": count, "false_accepts": false,
                                                              "false_accept_rate": far, "success_recall": recall}))
    if not feasible:
        return 1.0, {"accepted": 0, "false_accepts": 0, "false_accept_rate": 0.0, "success_recall": 0.0}
    _, _, threshold, metrics = max(feasible)
    return threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-scenes", nargs="+", required=True)
    parser.add_argument("--validation-scenes", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-validation-far", type=float, default=0.05)
    parser.add_argument("--backend", choices=("sklearn_hist", "xgboost"), default="sklearn_hist",
                        help="sklearn_hist is dependency-free in reloc3r; xgboost is optional")
    parser.add_argument("--estimators", type=int, default=160,
                        help="boosting rounds for xgboost, ignored by sklearn_hist")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="used only by the optional xgboost backend")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--allow-rule-rescue", action="store_true",
                        help="unsafe by default: permit learned acceptance after the rule verifier rejected")
    args = parser.parse_args()
    if not 0.0 <= args.max_validation_far < 1.0:
        parser.error("--max-validation-far must be in [0, 1)")
    rows = [row for row in read_rows(args.dataset) if row.get("candidate")]
    if not rows:
        raise ValueError("verifier dataset has no pose-producing candidates")
    train_scenes, validation_scenes = set(args.train_scenes), set(args.validation_scenes)
    if train_scenes & validation_scenes:
        parser.error("train and validation scenes must be disjoint")
    train = [row for row in rows if row["scene_id"] in train_scenes]
    validation = [row for row in rows if row["scene_id"] in validation_scenes]
    if not train or not validation:
        raise ValueError("scene split yielded an empty train or validation set")
    tx, ty, tt, tr = matrix(train)
    vx, vy, vt, vr = matrix(validation)
    if len(np.unique(ty)) < 2:
        raise ValueError("training set needs both successful and unsuccessful candidates")
    if args.backend == "xgboost":
        from tools.train_stage3c_xgboost_diagnostic import xgboost
        xgb = xgboost()
        common = {"n_estimators": args.estimators, "max_depth": args.max_depth,
                  "learning_rate": args.learning_rate, "subsample": 0.9,
                  "colsample_bytree": 0.9, "random_state": args.seed,
                  "tree_method": "hist", "device": args.device, "n_jobs": 4}
        classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        translation = xgb.XGBRegressor(objective="reg:squarederror", **common)
        rotation = xgb.XGBRegressor(objective="reg:squarederror", **common)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        common = {"learning_rate": args.learning_rate, "max_leaf_nodes": 2 ** args.max_depth,
                  "l2_regularization": 1e-3, "random_state": args.seed}
        classifier = HistGradientBoostingClassifier(**common)
        translation = HistGradientBoostingRegressor(**common)
        rotation = HistGradientBoostingRegressor(**common)
    classifier.fit(tx, ty); translation.fit(tx, tt); rotation.fit(tx, tr)
    probability = classifier.predict_proba(vx)[:, 1]
    threshold, threshold_metrics = threshold_at_far(probability, vy, args.max_validation_far)
    summary = {
        "schema_version": 1, "family": "poseagent_structured_verifier",
        "feature_names": list(StructuredVerifierFeatures.NAMES),
        "train_scenes": sorted(train_scenes), "validation_scenes": sorted(validation_scenes),
        "train_rows": len(train), "validation_rows": len(validation),
        "acceptance_threshold": threshold, "validation_threshold_metrics": threshold_metrics,
        "validation_brier": float(np.mean((probability - vy) ** 2)),
        "validation_translation_mae_m": float(np.mean(np.abs(np.expm1(translation.predict(vx)) - np.expm1(vt)))),
        "validation_rotation_mae_deg": float(np.mean(np.abs(np.expm1(rotation.predict(vx)) - np.expm1(vr)))),
        "model_backend": args.backend,
        "require_rule_accept": not args.allow_rule_rescue,
        "boundary": "GT labels are offline-only; online inference uses structured policy-visible evidence only.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "xgboost":
        classifier.save_model(args.output_dir / "success_classifier.json")
        translation.save_model(args.output_dir / "translation_regressor.json")
        rotation.save_model(args.output_dir / "rotation_regressor.json")
    else:
        import joblib
        joblib.dump(classifier, args.output_dir / "success_classifier.joblib")
        joblib.dump(translation, args.output_dir / "translation_regressor.joblib")
        joblib.dump(rotation, args.output_dir / "rotation_regressor.joblib")
    (args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
