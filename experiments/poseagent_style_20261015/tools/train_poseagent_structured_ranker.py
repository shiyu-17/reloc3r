#!/usr/bin/env python3
"""Train/validate the lightweight PoseAgent action ranker scene-disjointly."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

from tools.poseagent_style import PairProfiler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True,
                    help="offline-only rows from build_poseagent_verifier_dataset.py")
parser.add_argument("--train-scenes", nargs="+", required=True)
parser.add_argument("--validation-scenes", nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--cost-penalty", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=777)
args = parser.parse_args()
rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows or any(row.get("offline_only") is not True for row in rows):
    raise ValueError("dataset must contain offline_only labelled rows")
train_scenes, validation_scenes = set(args.train_scenes), set(args.validation_scenes)
if train_scenes & validation_scenes:
    raise ValueError("scene splits overlap")
train = [row for row in rows if row["scene_id"] in train_scenes and row.get("ranker_profile")]
validation = [row for row in rows if row["scene_id"] in validation_scenes and row.get("ranker_profile")]
if not train or not validation:
    raise ValueError("scene split has no ranker rows; regenerate candidates")
profile = PairProfiler()
def xy(values):
    return (np.stack([profile.vector(row["ranker_profile"]) for row in values]),
            np.asarray([bool(row["label"]["success"]) for row in values], dtype=np.int64))
tx, ty = xy(train); vx, vy = xy(validation)
if len(np.unique(ty)) < 2:
    raise ValueError("training ranker needs both success classes")
model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=16,
                                       l2_regularization=1e-3, random_state=args.seed).fit(tx, ty)
probability = model.predict_proba(vx)[:, 1]
# Candidate-level diagnostic. Episode success is evaluated later by the causal controller.
summary = {"schema_version": 1, "family": "poseagent_structured_ranker",
           "feature_names": list(PairProfiler.FEATURE_NAMES), "train_scenes": sorted(train_scenes),
           "validation_scenes": sorted(validation_scenes), "train_rows": len(train),
           "validation_rows": len(validation), "cost_penalty": args.cost_penalty,
           "validation_brier": float(np.mean((probability - vy) ** 2)),
           "validation_auc_proxy_pairwise": float(np.mean((probability[vy == 1, None] > probability[None, vy == 0]))
                                                   if np.any(vy == 1) and np.any(vy == 0) else 0.0),
           "boundary": "success labels are offline-only; online ranker receives PairProfiler features only."}
args.output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, args.output_dir / "success_classifier.joblib")
(args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
