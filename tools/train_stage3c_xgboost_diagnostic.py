#!/usr/bin/env python3
"""XGBoost recovery/cost/ranking diagnostic for Stage 3C.

This is deliberately a diagnostic baseline, not the deployed sequential
controller.  It answers whether the current structured observation has enough
signal before a neural policy or language model is made more complex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from tools.stage3b_policy import action_features, state_features
from tools.stage3d_features import feature_metadata, feature_vector


def xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ModuleNotFoundError as error:
        bundled = Path(__file__).resolve().parents[1] / "third_party" / "stage3c_xgboost"
        if bundled.exists() and str(bundled) not in sys.path:
            sys.path.insert(0, str(bundled))
            try:
                import xgboost as xgb
                return xgb
            except ModuleNotFoundError:
                pass
        raise RuntimeError(
            "xgboost is required for the Stage 3C diagnostic. Install it in the reloc3r environment; "
            "the MLP path remains independent of this optional baseline.") from error


def read_records(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("recovery dataset is empty")
    return rows


def split(records: Sequence[Mapping[str, Any]], validation_fraction: float):
    frames: Dict[str, List[int]] = {}
    for row in records:
        if row.get("dagger_diagnostics"):
            continue
        frames.setdefault(str(row["sequence"]), []).append(int(row["frame_index"]))
    validation_frames = {}
    for sequence, values in frames.items():
        unique = sorted(set(values)); count = max(1, int(round(len(unique) * validation_fraction)))
        validation_frames[sequence] = set(unique[-count:])
    train = [row for row in records if row.get("dagger_diagnostics") or int(row["frame_index"]) not in validation_frames[str(row["sequence"])]]
    validation = [row for row in records if not row.get("dagger_diagnostics") and int(row["frame_index"]) in validation_frames[str(row["sequence"])]]
    if not train or not validation:
        raise ValueError("empty temporal partition")
    return train, validation, {key: sorted(value) for key, value in validation_frames.items()}


def flatten(records: Sequence[Mapping[str, Any]], feature_level: str = "legacy"):
    x, recovery, log_cost, advantage, weights, group, group_weights, kinds = [], [], [], [], [], [], [], []
    for row in records:
        state = (state_features(row["observation"]) if feature_level == "legacy"
                 else feature_vector(row["observation"], feature_level))
        group.append(len(row["actions"]))
        # XGBRanker defines sample weights per query group, whereas the
        # classifier/regressor consume one weight per action row.
        row_weight = float(row.get("stage3d_training_weight", row.get("training_weight", 1.0)))
        group_weights.append(row_weight)
        for action in row["actions"]:
            x.append(np.concatenate((state, action_features(row["observation"], action))))
            recovery.append(float(action["recovery_target"]))
            log_cost.append(np.log1p(max(0.0, float(action["incremental_cost_s"]))))
            advantage.append(float(action["oracle_advantage"]))
            weights.append(row_weight)
            kinds.append(str(action["kind"]))
    return (np.asarray(x, dtype=np.float32), np.asarray(recovery), np.asarray(log_cost),
            np.asarray(advantage), np.asarray(weights), group, np.asarray(group_weights), kinds)


def rank_metrics(predicted: np.ndarray, target: np.ndarray, group: Sequence[int], kinds: Sequence[str]):
    pair_correct = pair_total = geometry_correct = geometry_total = 0; offset = 0
    for size in group:
        p, t, k = predicted[offset:offset + size], target[offset:offset + size], kinds[offset:offset + size]
        for i in range(size):
            for j in range(i + 1, size):
                if abs(float(t[i] - t[j])) <= 1e-7:
                    continue
                pair_total += 1; pair_correct += int((p[i] - p[j]) * (t[i] - t[j]) > 0)
        if "geometry" in k and size > 1:
            g = k.index("geometry"); alternatives = [i for i in range(size) if i != g]
            geometry_total += 1
            geometry_correct += int((p[g] >= max(p[i] for i in alternatives)) == (t[g] >= max(t[i] for i in alternatives)))
        offset += size
    return {"pairwise_sign_accuracy": pair_correct / max(1, pair_total),
            "geometry_margin_sign_accuracy": geometry_correct / max(1, geometry_total),
            "geometry_margin_states": geometry_total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-cost", type=float, required=True)
    parser.add_argument("--estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="XGBoost device; GPU is the default when CUDA is available")
    parser.add_argument("--feature-level", choices=("legacy", "f0", "f1", "f2", "f3"), default="legacy",
                        help="Stage 3D pre-action feature ladder level; legacy preserves Stage 3C dimensions")
    args = parser.parse_args()
    xgb = xgboost()
    records = read_records(args.dataset)
    train_rows, validation_rows, validation_frames = split(records, args.validation_fraction)
    tx, ty, tc, ta, tw, tg, tgw, _ = flatten(train_rows, args.feature_level)
    vx, vy, vc, va, vw, vg, vgw, vk = flatten(validation_rows, args.feature_level)
    common = {"n_estimators": args.estimators, "max_depth": args.max_depth, "learning_rate": args.learning_rate,
              "subsample": 0.9, "colsample_bytree": 0.9, "random_state": args.seed, "tree_method": "hist",
              "device": args.device, "n_jobs": 4}
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    regressor = xgb.XGBRegressor(objective="reg:squarederror", **common)
    ranker = xgb.XGBRanker(objective="rank:pairwise", eval_metric="ndcg", **common)
    classifier.fit(tx, ty, sample_weight=tw); regressor.fit(tx, tc, sample_weight=tw)
    # XGBRanker requires contiguous actions grouped by state, which flatten preserves.
    ranker.fit(tx, ta, group=tg, sample_weight=tgw)
    probability = classifier.predict_proba(vx)[:, 1]
    predicted_cost = np.expm1(regressor.predict(vx)).clip(0.0, None)
    explicit_advantage = probability - args.lambda_cost * predicted_cost
    direct_rank = ranker.predict(vx)
    summary = {
        "schema_version": 1, "family": "stage3c_xgboost_diagnostic", "dataset": str(args.dataset),
        "feature_level": args.feature_level,
        "feature_metadata": ({"level": "legacy", "dimension": int(tx.shape[1] - 12)}
                             if args.feature_level == "legacy" else feature_metadata(args.feature_level)),
        "validation_frames": validation_frames, "train_states": len(train_rows), "validation_states": len(validation_rows),
        "lambda_cost": args.lambda_cost,
        "classifier_brier": float(np.mean((probability - vy) ** 2)),
        "cost_mae_s": float(np.mean(np.abs(predicted_cost - np.expm1(vc)))),
        "explicit_controller_ranking": rank_metrics(explicit_advantage, va, vg, vk),
        "xgbranker_ranking": rank_metrics(direct_rank, va, vg, vk),
        "implementation_boundary": "XGBoost is an offline structured-evidence diagnostic. Its outputs are not used by the deployed policy checkpoint.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(args.output_dir / "recovery_classifier.json")
    regressor.save_model(args.output_dir / "incremental_cost_regressor.json")
    ranker.save_model(args.output_dir / "action_ranker.json")
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
