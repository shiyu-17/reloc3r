#!/usr/bin/env python3
"""Train the Stage 3C.6 utility-consistent component controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from tools.stage3b_policy import action_features, state_features
from tools.stage3d_features import FEATURE_LEVELS, feature_metadata, feature_vector, strip_post_geometry_fields
from tools.train_stage3c_xgboost_diagnostic import xgboost


COMPONENTS = ("correct_accept", "false_accept", "incremental_cost_s",
              "recovery_time_frames", "memory_pollution")


def read_records(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("utility dataset is empty")
    return rows


def temporal_split(records: Sequence[Mapping[str, Any]], validation_fraction: float):
    frames: Dict[tuple[str, str], List[int]] = {}
    for row in records:
        source = "dagger" if row.get("dagger_diagnostics") else "oracle"
        frames.setdefault((str(row["sequence"]), source), []).append(int(row["frame_index"]))
    validation_frames: Dict[str, Dict[str, set[int]]] = {}
    for (sequence, source), values in frames.items():
        unique = sorted(set(values))
        count = max(1, int(round(len(unique) * validation_fraction)))
        validation_frames.setdefault(sequence, {})[source] = set(unique[-count:])
    def is_validation(row):
        source = "dagger" if row.get("dagger_diagnostics") else "oracle"
        return int(row["frame_index"]) in validation_frames[str(row["sequence"])][source]
    train = [row for row in records if not is_validation(row)]
    validation = [row for row in records if is_validation(row)]
    if not train or not validation:
        raise ValueError("empty temporal partition")
    return train, validation, {sequence: {source: sorted(values) for source, values in sources.items()}
                               for sequence, sources in validation_frames.items()}


def _state_features(observation: Any, feature_level: str) -> np.ndarray:
    if feature_level == "legacy":
        return state_features(observation)
    # States reached after a Geometry receipt may still be eligible for a
    # second acquisition.  Project only the receipt itself out of the policy
    # view, rather than leaking RoMa output into a pre-Geometry controller.
    return feature_vector(strip_post_geometry_fields(observation), feature_level)


def flatten(records: Sequence[Mapping[str, Any]], feature_level: str = "legacy",
            targeted_state_weight: float = 1.0):
    x, targets, utilities, kinds, groups, weights = [], {key: [] for key in COMPONENTS}, [], [], [], []
    for row in records:
        state = _state_features(row["observation"], feature_level)
        actions = row["actions"]
        groups.append(len(actions))
        targeted = bool(set(row.get("target_tags", [])) & {
            "geometry_optimal", "memory_optimal", "premature_accept", "high_oracle_regret"})
        # Preserve the natural action prior for calibration.  The multiplier
        # only counteracts sparse, policy-induced recovery opportunities and
        # is bounded at the *state*, never individual-action, level.
        state_weight = float(targeted_state_weight) if targeted else 1.0
        per_action_weight = state_weight / max(1, len(actions))
        for action in actions:
            x.append(np.concatenate((state, action_features(row["observation"], action))))
            for key in COMPONENTS:
                targets[key].append(float(action[key]))
            utilities.append(float(action["oracle_incremental_utility"]))
            kinds.append(str(action["kind"]))
            weights.append(per_action_weight)
    return (np.asarray(x, dtype=np.float32),
            {key: np.asarray(value, dtype=np.float32) for key, value in targets.items()},
            np.asarray(utilities, dtype=np.float32), kinds, groups, np.asarray(weights, dtype=np.float32))


def compose(predicted: Mapping[str, np.ndarray], utility: Mapping[str, Any]) -> np.ndarray:
    return (
        float(utility["success_reward"]) * predicted["correct_accept"]
        - float(utility["false_accept_penalty"]) * predicted["false_accept"]
        - float(utility["cost_penalty_per_s"]) * predicted["incremental_cost_s"]
        - float(utility["recovery_penalty_per_frame"]) * predicted["recovery_time_frames"]
        - float(utility["memory_pollution_penalty"]) * predicted["memory_pollution"]
    )


def decision_metrics(predicted: np.ndarray, target: np.ndarray, groups: Sequence[int],
                     kinds: Sequence[str]) -> Dict[str, Any]:
    offset = correct = count = geometry_correct = geometry_count = admission_correct = admission_count = 0
    regret = 0.0
    by_oracle: Dict[str, Dict[str, int]] = {}
    for size in groups:
        p = predicted[offset:offset + size]; t = target[offset:offset + size]
        k = kinds[offset:offset + size]
        pi, ti = int(np.argmax(p)), int(np.argmax(t))
        correct += int(abs(float(t[pi] - t[ti])) <= 1e-7); count += 1
        regret += float(t[ti] - t[pi])
        oracle_kind = k[ti]
        bucket = by_oracle.setdefault(oracle_kind, {"states": 0, "chosen_optimal": 0})
        bucket["states"] += 1; bucket["chosen_optimal"] += int(abs(float(t[pi] - t[ti])) <= 1e-7)
        g = [i for i, kind in enumerate(k) if kind == "geometry"]
        alt = [i for i, kind in enumerate(k) if kind != "geometry"]
        if g and alt:
            geometry_count += 1
            geometry_correct += int((p[g[0]] >= max(p[i] for i in alt)) ==
                                    (t[g[0]] >= max(t[i] for i in alt)))
        acq = [i for i, kind in enumerate(k) if kind in {"localize", "geometry", "memory"}]
        terminal = [i for i, kind in enumerate(k) if kind in {"accept", "defer", "reject"}]
        if acq and terminal:
            admission_count += 1
            admission_correct += int((max(p[i] for i in acq) > max(p[i] for i in terminal)) ==
                                     (max(t[i] for i in acq) > max(t[i] for i in terminal)))
        offset += size
    for value in by_oracle.values():
        value["accuracy"] = value["chosen_optimal"] / max(1, value["states"])
    return {
        "states": count, "top1_oracle_action_accuracy": correct / max(1, count),
        "mean_oracle_regret": regret / max(1, count),
        "geometry_margin_sign_accuracy": geometry_correct / max(1, geometry_count),
        "geometry_margin_states": geometry_count,
        "acquisition_admission_accuracy": admission_correct / max(1, admission_count),
        "acquisition_admission_states": admission_count,
        "by_oracle_action": by_oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=240)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--feature-level", choices=("legacy", *FEATURE_LEVELS), default="legacy",
                        help="Stage 3D observation ladder; legacy preserves the Stage 3C.6 state vector.")
    parser.add_argument("--targeted-state-weight", type=float, default=1.0,
                        help="bounded multiplier for tagged recovery/opportunity states; 1 disables targeting")
    args = parser.parse_args()
    records = read_records(args.dataset)
    train_rows, validation_rows, validation_frames = temporal_split(records, args.validation_fraction)
    if not 1.0 <= args.targeted_state_weight <= 3.0:
        parser.error("--targeted-state-weight must be in [1, 3]")
    tx, ty, tu, _, _, tw = flatten(train_rows, args.feature_level, args.targeted_state_weight)
    vx, vy, vu, vk, vg, _ = flatten(validation_rows, args.feature_level, 1.0)
    utility = json.loads((args.dataset.parent / "summary.json").read_text(encoding="utf-8"))["utility"]
    xgb = xgboost()
    common = {"n_estimators": args.estimators, "max_depth": args.max_depth,
              "learning_rate": args.learning_rate, "subsample": 0.9, "colsample_bytree": 0.9,
              "random_state": args.seed, "tree_method": "hist", "device": args.device, "n_jobs": 4}
    models: Dict[str, Any] = {}
    for key in ("correct_accept", "false_accept", "memory_pollution"):
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        model.fit(tx, ty[key], sample_weight=tw)
        models[key] = model
    for key in ("incremental_cost_s", "recovery_time_frames"):
        model = xgb.XGBRegressor(objective="reg:squarederror", **common)
        model.fit(tx, np.log1p(ty[key]), sample_weight=tw)
        models[key] = model
    predicted = {
        "correct_accept": models["correct_accept"].predict_proba(vx)[:, 1],
        "false_accept": models["false_accept"].predict_proba(vx)[:, 1],
        "memory_pollution": models["memory_pollution"].predict_proba(vx)[:, 1],
        "incremental_cost_s": np.expm1(models["incremental_cost_s"].predict(vx)).clip(0.0, None),
        "recovery_time_frames": np.expm1(models["recovery_time_frames"].predict(vx)).clip(0.0, None),
    }
    predicted_utility = compose(predicted, utility)
    component_metrics = {}
    for key in COMPONENTS:
        if key in {"correct_accept", "false_accept", "memory_pollution"}:
            component_metrics[key + "_brier"] = float(np.mean((predicted[key] - vy[key]) ** 2))
        else:
            component_metrics[key + "_mae"] = float(np.mean(np.abs(predicted[key] - vy[key])))
    summary = {
        "schema_version": 1, "family": "stage3c6_utility_controller",
        "dataset": str(args.dataset), "utility": utility,
        "feature_level": args.feature_level,
        "feature_metadata": (None if args.feature_level == "legacy" else feature_metadata(args.feature_level)),
        "training": {"train_states": len(train_rows), "validation_states": len(validation_rows),
                     "validation_frames": validation_frames, "seed": args.seed,
                     "estimators": args.estimators, "max_depth": args.max_depth,
                     "learning_rate": args.learning_rate,
                     "targeted_state_weight": args.targeted_state_weight},
        "component_metrics": component_metrics,
        "utility_mae": float(np.mean(np.abs(predicted_utility - vu))),
        "utility_correlation": float(np.corrcoef(predicted_utility, vu)[0, 1]),
        "decision_metrics": decision_metrics(predicted_utility, vu, vg, vk),
        "controller_boundary": "all legal actions share the exact incremental Oracle utility; verifier is only a hard Accept eligibility mask",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "correct_accept": "correct_accept_classifier.json",
        "false_accept": "false_accept_classifier.json",
        "incremental_cost_s": "incremental_cost_regressor.json",
        "recovery_time_frames": "recovery_time_regressor.json",
        "memory_pollution": "memory_pollution_classifier.json",
    }
    for key, model in models.items():
        model.save_model(args.output_dir / filenames[key])
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
