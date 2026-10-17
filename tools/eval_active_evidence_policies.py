#!/usr/bin/env python3
"""Pilot evaluation of active evidence acquisition policies.

The online policies only observe map-side retrieval/group evidence and tool
outputs.  Query GT is used only by the offline calibration/evaluator paths;
it is never passed into the online environment or action-selection policy.  The CPU ORB backend is
used when CUDA/Reloc3r is unavailable; it executes the same causal environment
contract and is explicitly an engineering pilot, not the paper baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend
from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical,
    pose_error,
    read_manifest,
    tum_camera_poses,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
POLICIES = ("fixed", "random", "rule", "light")


def audit_flags() -> Dict[str, bool]:
    return {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False}


def load_query_gt(data_root: Path, gt_root: Path, sequence: str):
    _, _, query_times, _, query_root = read_manifest(data_root, sequence)
    poses, _ = tum_camera_poses(
        gt_root / f"{sequence}_gt_poses.txt", query_times,
        base_to_front_rgb_optical(query_root), 0.1)
    return poses


def success_pose(pose: Any, gt: np.ndarray) -> bool:
    if pose is None:
        return False
    t, r = pose_error(np.asarray(pose, dtype=np.float64), gt)
    return bool(t <= 0.25 and r <= 5.0)


def group_features(group: Mapping[str, Any]) -> np.ndarray:
    scores = np.asarray(group.get("retrieval_scores", []), dtype=np.float64)
    return np.asarray([
        float(group.get("mean_retrieval_score", scores.mean() if len(scores) else 0.0)),
        float(scores.std() if len(scores) else 0.0),
        float(group.get("map_spread_m", 0.0)),
    ], dtype=np.float64)


def groups_from_observation(observation) -> List[Dict[str, Any]]:
    groups = observation.frame.get("reference_groups", [])
    if groups:
        return [dict(group) for group in groups if len(group.get("reference_indices", [])) >= 2]
    refs = observation.frame.get("retrieval_references", [])
    scores = observation.frame.get("retrieval_scores", [])
    return [{"reference_indices": refs[start:start + 3],
             "retrieval_scores": scores[start:start + 3],
             "mean_retrieval_score": float(np.mean(scores[start:start + 3]))}
            for start in range(0, len(refs), 3) if len(refs[start:start + 3]) >= 2]


def load_policy_model(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    # Stored by this script as a small numpy/json artifact; retrain is cheap,
    # so the default path is an in-memory model and no pickle is trusted.
    payload = json.loads(path.read_text())
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=500))
    model.fit(np.asarray(payload["features"], dtype=np.float64), np.asarray(payload["labels"], dtype=int))
    return model


def policy_order(name: str, groups: Sequence[Mapping[str, Any]], *, rng: random.Random,
                 model=None) -> List[int]:
    indices = list(range(len(groups)))
    if name == "fixed":
        return indices
    if name == "random":
        rng.shuffle(indices)
        return indices
    if name == "rule":
        # Map-side evidence only: prefer high retrieval score and a useful
        # spatial baseline, without looking at pose correctness.
        return sorted(indices, key=lambda i: (
            float(groups[i].get("mean_retrieval_score", 0.0)) +
            0.02 * float(groups[i].get("map_spread_m", 0.0))), reverse=True)
    if name == "light":
        if model is None:
            return policy_order("rule", groups, rng=rng)
        features = np.stack([group_features(groups[i]) for i in indices])
        probabilities = model.predict_proba(features)[:, 1]
        return [index for _, index in sorted(zip(probabilities, indices), reverse=True)]
    raise ValueError(name)


def make_backend(args, sequence: str):
    return CpuOrbLocalizationBackend(
        args.data_root, args.mapping_gt, sequence,
        args.retrieval_root / sequence / "retrieval.npz",
        nfeatures=args.nfeatures,
    )


def calibrate_light_policy(args, sequences: Sequence[str], calibration_end: int,
                           gt_by_sequence: Mapping[str, np.ndarray], output_dir: Path):
    features, labels = [], []
    rows = []
    for sequence in sequences:
        backend = make_backend(args, sequence)
        for frame_index in range(min(calibration_end, backend.num_frames(sequence))):
            observation = type("Observation", (), {
                "frame": backend.frame_observation(sequence, frame_index, {})})()
            groups = groups_from_observation(observation)
            for group_index, group in enumerate(groups):
                result = backend.localize(sequence, frame_index,
                                          group["reference_indices"], {})
                pose = result.get("pose")
                features.append(group_features(group))
                labels.append(int(success_pose(pose, gt_by_sequence[sequence][frame_index])))
                rows.append({"sequence": sequence, "frame_index": frame_index,
                             "group_index": group_index, "label": labels[-1],
                             "features": features[-1].tolist(),
                             "causal": audit_flags()})
    if len(set(labels)) < 2:
        # A degenerate calibration split is still recorded, but the light
        # policy falls back to the map-side rule without using test outcomes.
        model = None
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=500))
        model.fit(np.asarray(features), np.asarray(labels))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "light_calibration.json").write_text(json.dumps({
        "features": [x.tolist() for x in features], "labels": labels,
        "rows": rows, "model_available": model is not None,
        "calibration_end": calibration_end,
        "online_selection_uses_gt": False,
        "gt_used_for_calibration_labels": True,
    }, indent=2) + "\n")
    return model


def run_policy(args, sequence: str, policy_name: str, start: int, end: int,
               model=None) -> Dict[str, Any]:
    backend = make_backend(args, sequence)
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=2 * args.max_evidence_actions + 3)
    observation = env.reset(sequence, start_frame=start)
    records = []
    stable_offset = sum((index + 1) * ord(char)
                        for index, char in enumerate(sequence + ":" + policy_name))
    rng = random.Random(args.seed + stable_offset)
    while not observation.done and observation.frame_index < end:
        frame_index = observation.frame_index
        groups = groups_from_observation(observation)
        order = policy_order(policy_name, groups, rng=rng, model=model)
        frame_start = time.perf_counter()
        accepted_pose = None
        accepted_source = None
        attempts = 0
        verification_attempts = 0
        memory_attempted = 0
        for group_index in order[:args.max_evidence_actions]:
            refs = groups[group_index]["reference_indices"]
            observation, _, _, _ = env.step(LocalizationAction.localize(*refs))
            attempts += 1
            if not observation.candidate or not observation.candidate.get("pose_available"):
                continue
            verification_attempts += 1
            observation, _, _, _ = env.step(LocalizationAction.verify())
            if observation.candidate and observation.candidate.get("verified"):
                accepted_source = observation.candidate.get("source")
                observation, _, _, _ = env.step(LocalizationAction.accept())
                accepted_pose = None if env.state.accepted_pose is None else env.state.accepted_pose.copy()
                break
        if accepted_pose is None:
            # All policies get the same causal historical-memory option after
            # their evidence budget.  It is not an oracle and cannot see GT.
            memory_attempted = 1
            observation, _, _, _ = env.step(LocalizationAction.retrieve())
            if observation.candidate and observation.candidate.get("pose_available"):
                verification_attempts += 1
                observation, _, _, _ = env.step(LocalizationAction.verify())
                if observation.candidate and observation.candidate.get("verified"):
                    accepted_source = observation.candidate.get("source")
                    observation, _, _, _ = env.step(LocalizationAction.accept())
                    accepted_pose = None if env.state.accepted_pose is None else env.state.accepted_pose.copy()
            if accepted_pose is None:
                observation, _, _, _ = env.step(LocalizationAction.defer())
        records.append({
            "sequence": sequence, "frame_index": frame_index,
            "policy": policy_name, "attempts": attempts,
            "verification_attempts": verification_attempts,
            "memory_attempted": memory_attempted,
            "total_tool_actions": attempts + verification_attempts + memory_attempted,
            "accepted": accepted_pose is not None,
            "pose": None if accepted_pose is None else accepted_pose.tolist(),
            "pose_source": accepted_source,
            "wall_time_s": time.perf_counter() - frame_start,
            "causal": audit_flags(),
        })
    audit = env.audit_trace()
    if not audit["passed"]:
        raise RuntimeError(f"{sequence}/{policy_name} causal audit failed: {audit}")
    return {"sequence": sequence, "policy": policy_name,
            "records": records, "audit": audit,
            "query_gt_read": False, "uses_future": False,
            "selection_uses_gt": False}


def summarize(records: Sequence[Mapping[str, Any]], gt: np.ndarray) -> Dict[str, Any]:
    accepted = [row for row in records if row["accepted"]]
    correct = sum(success_pose(row["pose"], gt[int(row["frame_index"])]) for row in accepted)
    return {
        "frames": len(records), "accepted": len(accepted),
        "coverage": len(accepted) / len(records) if records else 0.0,
        "accepted_correct": int(correct),
        "accepted_accuracy": correct / len(accepted) if accepted else 0.0,
        "false_accept_rate": (len(accepted) - correct) / len(accepted) if accepted else 0.0,
        "full_frame_success": correct / len(records) if records else 0.0,
        "mean_evidence_attempts": float(np.mean([r["attempts"] for r in records])) if records else 0.0,
        "mean_verification_attempts": float(np.mean([r.get("verification_attempts", 0) for r in records])) if records else 0.0,
        "mean_memory_attempts": float(np.mean([r.get("memory_attempted", 0) for r in records])) if records else 0.0,
        "mean_total_tool_actions": float(np.mean([r.get("total_tool_actions", r.get("attempts", 0)) for r in records])) if records else 0.0,
        "mean_wall_time_s": float(np.mean([r["wall_time_s"] for r in records])) if records else 0.0,
    }


def evaluate_roma(args, sequence: str, start: int, end: int, gt: np.ndarray):
    path = args.roma_root / f"{sequence}_top100_merged.jsonl"
    if not path.exists():
        return None
    rows = {int(json.loads(line)["query_index"]): json.loads(line)
            for line in path.read_text().splitlines() if line.strip()}
    selected = []
    for index in range(start, end):
        row = rows.get(index, {})
        selected.append({"pose": row.get("pose"), "accepted": row.get("pose") is not None,
                         "frame_index": index, "attempts": 1,
                         "wall_time_s": float(row.get("elapsed_s", 0.0))})
    return summarize(selected, gt)


def oracle_summary(args, sequence: str, start: int, end: int, gt: np.ndarray):
    backend = make_backend(args, sequence)
    rows = []
    for frame_index in range(start, min(end, backend.num_frames(sequence))):
        observation = type("Observation", (), {
            "frame": backend.frame_observation(sequence, frame_index, {})})()
        groups = groups_from_observation(observation)
        outcomes = []
        for group in groups:
            result = backend.localize(sequence, frame_index, group["reference_indices"], {})
            outcomes.append(result.get("pose"))
        correct = [pose for pose in outcomes if success_pose(pose, gt[frame_index])]
        chosen = correct[0] if correct else None
        rows.append({"frame_index": frame_index, "pose": chosen,
                     "accepted": chosen is not None,
                     "attempts": 1 if chosen is not None else len(outcomes),
                     "wall_time_s": 0.0})
    summary = summarize(rows, gt)
    summary["oracle_uses_gt"] = True
    summary["interpretation"] = "finite-depth counterfactual upper bound; not deployable"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--roma-root", type=Path, default=Path("outputs/lilocbench/AVL-004/always_geometry/roma"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lilocbench/active_evidence_pilot"))
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--max-frames", type=int, default=8,
                        help="pilot prefix per sequence; split into calibration/test")
    parser.add_argument("--calibration-fraction", type=float, default=0.4)
    parser.add_argument("--max-evidence-actions", type=int, default=2)
    parser.add_argument("--nfeatures", type=int, default=500)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if args.max_frames < 3:
        raise ValueError("max-frames must leave both calibration and test frames")
    split = max(1, int(math.ceil(args.max_frames * args.calibration_fraction)))
    gt = {sequence: load_query_gt(args.data_root, args.gt_root, sequence)
          for sequence in args.sequences}
    model = calibrate_light_policy(args, args.sequences, split, gt, args.output_dir)
    output = {"config": {
        "sequences": args.sequences, "max_frames": args.max_frames,
        "calibration_end": split, "max_evidence_actions": args.max_evidence_actions,
        "nfeatures": args.nfeatures, "seed": args.seed,
        "online_policy_causal": True,
    }, "policies": {}, "oracle": {}, "fixed_roma": {}}
    for sequence in args.sequences:
        end = min(args.max_frames, len(gt[sequence]))
        for policy in POLICIES:
            run = run_policy(args, sequence, policy, split, end, model=model)
            output["policies"].setdefault(policy, {})[sequence] = {
                "summary": summarize(run["records"], gt[sequence]),
                "audit": run["audit"],
            }
            path = args.output_dir / "traces" / f"{sequence}_{policy}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in run["records"]))
        output["oracle"][sequence] = oracle_summary(args, sequence, split, end, gt[sequence])
        output["fixed_roma"][sequence] = evaluate_roma(args, sequence, split, end, gt[sequence])
    output["online_audit"] = audit_flags()
    output["calibration"] = {
        "gt_used_for_labels": True,
        "online_selection_uses_gt": False,
        "split": "calibration_prefix_only",
    }
    output["interpretation"] = {
        "fixed_random_rule_light": "online policies; no query GT in action selection",
        "light": "logistic policy trained only on calibration-prefix evaluator labels",
        "oracle": "counterfactual GT upper bound, not deployable",
        "fixed_roma": "historical fixed strong-tool reference; cost is not directly comparable to CPU ORB",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
