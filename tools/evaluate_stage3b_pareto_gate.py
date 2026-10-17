#!/usr/bin/env python3
"""Evaluate the Stage 3B three-objective Pareto Gate on paired frames.

The gate has three independently auditable objectives: success
non-inferiority, false-accept safety, and measured-tool-cost efficiency.
It never changes a policy or utility and uses GT only in this offline
evaluator to score already-written poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from tools.eval_active_evidence_policies import load_query_gt, success_pose


FrameKey = Tuple[str, int]


def load_traces(root: Path, policy: str) -> Dict[FrameKey, Dict[str, Any]]:
    paths = sorted(root.glob(f"*/traces/*_{policy}.jsonl"))
    direct = root / "traces"
    if direct.is_dir():
        paths.extend(sorted(direct.glob(f"*_{policy}.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no *_{policy}.jsonl traces beneath {root}")
    rows: Dict[FrameKey, Dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sequence"]), int(row["frame_index"]))
            if key in rows:
                raise ValueError(f"duplicate frame {key} at {path}:{line_number}")
            rows[key] = row
    return rows


def load_policy_audits(root: Path, policy: str) -> Dict[str, bool]:
    """Read per-sequence causal audit status from completed evaluator summaries."""
    paths = sorted(root.glob("*/summary.json"))
    direct = root / "summary.json"
    if direct.exists():
        paths.append(direct)
    audits: Dict[str, bool] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sequence, entry in data.get("policies", {}).get(policy, {}).items():
            passed = bool(entry.get("audit", {}).get("passed", False))
            if sequence in audits and audits[sequence] != passed:
                raise ValueError(f"conflicting causal audit status for {policy}/{sequence}")
            audits[str(sequence)] = passed
    return audits


def retain_audited(rows: Mapping[FrameKey, Dict[str, Any]],
                    audits: Mapping[str, bool]) -> Tuple[Dict[FrameKey, Dict[str, Any]], Dict[str, str]]:
    excluded: Dict[str, str] = {}
    retained: Dict[FrameKey, Dict[str, Any]] = {}
    for key, row in rows.items():
        sequence = key[0]
        if sequence not in audits:
            excluded[sequence] = "missing_completed_summary_audit"
        elif not audits[sequence]:
            excluded[sequence] = "causal_audit_failed"
        else:
            retained[key] = row
    return retained, excluded


def frame_arrays(rows: Mapping[FrameKey, Mapping[str, Any]], *, data_root: Path,
                 gt_root: Path) -> Dict[str, np.ndarray]:
    gt_by_sequence = {
        sequence: load_query_gt(data_root, gt_root, sequence)
        for sequence in sorted({key[0] for key in rows})
    }
    success, accepted, cost, geometry = [], [], [], []
    for (sequence, frame_index), row in rows.items():
        pose = row.get("pose")
        is_accepted = bool(row.get("accepted")) and pose is not None
        is_success = bool(is_accepted and success_pose(pose, gt_by_sequence[sequence][frame_index]))
        accepted.append(is_accepted)
        success.append(is_success)
        cost.append(float(row.get("measured_tool_cost_s", 0.0)))
        geometry.append(any(action.get("kind") == "geometry"
                            for action in row.get("acquisitions", row.get("program", []))))
    return {
        "success": np.asarray(success, dtype=np.float64),
        "accepted": np.asarray(accepted, dtype=np.float64),
        "cost": np.asarray(cost, dtype=np.float64),
        "geometry": np.asarray(geometry, dtype=np.float64),
    }


def metrics(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    frames = int(len(arrays["success"]))
    accepted = int(arrays["accepted"].sum())
    correct = int(arrays["success"].sum())
    false_accepts = accepted - correct
    return {
        "frames": frames,
        "accepted": accepted,
        "coverage": accepted / frames if frames else 0.0,
        "accepted_correct": correct,
        "full_frame_success": correct / frames if frames else 0.0,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / accepted if accepted else 0.0,
        "mean_measured_tool_cost_s": float(arrays["cost"].mean()) if frames else 0.0,
        "geometry_frames": int(arrays["geometry"].sum()),
        "geometry_frame_rate": float(arrays["geometry"].mean()) if frames else 0.0,
    }


def _far(success: np.ndarray, accepted: np.ndarray) -> float:
    count = float(accepted.sum())
    return float((accepted.sum() - success.sum()) / count) if count else 0.0


def paired_bootstrap(learned: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], *,
                     samples: int, confidence: float, seed: int) -> Dict[str, Any]:
    count = len(learned["success"])
    rng = np.random.default_rng(seed)
    differences = {"success": [], "false_accept_rate": [], "cost_s": []}
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        differences["success"].append(float(
            learned["success"][indices].mean() - baseline["success"][indices].mean()))
        differences["false_accept_rate"].append(
            _far(learned["success"][indices], learned["accepted"][indices])
            - _far(baseline["success"][indices], baseline["accepted"][indices]))
        differences["cost_s"].append(float(
            learned["cost"][indices].mean() - baseline["cost"][indices].mean()))
    alpha = (1.0 - confidence) / 2.0
    output: Dict[str, Any] = {}
    for name, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        output[name + "_difference"] = {
            "learned_minus_baseline": float(array.mean()),
            "confidence_interval": [float(np.quantile(array, alpha)),
                                    float(np.quantile(array, 1.0 - alpha))],
        }
    return output


def pareto_relation(learned: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    learned_values = (float(learned["full_frame_success"]),
                      -float(learned["false_accept_rate"]),
                      -float(learned["mean_measured_tool_cost_s"]))
    baseline_values = (float(baseline["full_frame_success"]),
                       -float(baseline["false_accept_rate"]),
                       -float(baseline["mean_measured_tool_cost_s"]))
    learned_dominates = all(a >= b for a, b in zip(learned_values, baseline_values)) and any(
        a > b for a, b in zip(learned_values, baseline_values))
    baseline_dominates = all(b >= a for a, b in zip(learned_values, baseline_values)) and any(
        b > a for a, b in zip(learned_values, baseline_values))
    if learned_dominates:
        return "learned_dominates_baseline"
    if baseline_dominates:
        return "baseline_dominates_learned"
    return "non_dominated_tradeoff"


def evaluate_gate(learned: Mapping[str, Any], baseline: Mapping[str, Any],
                  paired: Mapping[str, Any], config: Mapping[str, Any], *,
                  common_frames: int, common_sequences: Sequence[str]) -> Dict[str, Any]:
    margin = float(config["success"]["noninferiority_margin"])
    success_difference = (float(learned["full_frame_success"])
                          - float(baseline["full_frame_success"]))
    success_point = success_difference >= -margin
    ci_low = float(paired["success_difference"]["confidence_interval"][0])
    ci_required = bool(config["success"].get("require_ci_noninferiority", False))
    success_ci = ci_low >= -margin
    success_pass = success_point and (success_ci or not ci_required)

    far_limit = float(config["safety"]["max_false_accept_rate"])
    safety_pass = float(learned["false_accept_rate"]) <= far_limit
    cost_limit = float(config["efficiency"]["max_mean_tool_cost_s"])
    absolute_cost_pass = float(learned["mean_measured_tool_cost_s"]) < cost_limit
    relative_required = bool(config["efficiency"].get("require_lower_cost_than_baseline", True))
    relative_cost_pass = (float(learned["mean_measured_tool_cost_s"])
                          < float(baseline["mean_measured_tool_cost_s"]))
    efficiency_pass = absolute_cost_pass and (relative_cost_pass or not relative_required)

    coverage_config = config["coverage"]
    required_sequences = set(map(str, coverage_config["required_sequences"]))
    sequences_pass = required_sequences.issubset(set(common_sequences))
    frames_pass = common_frames >= int(coverage_config["minimum_common_frames"])
    coverage_pass = sequences_pass and frames_pass
    objectives = {
        "success_noninferiority": {
            "pass": success_pass, "difference": success_difference,
            "margin": margin, "point_estimate_pass": success_point,
            "ci_required": ci_required, "ci_pass": success_ci,
        },
        "false_accept_safety": {
            "pass": safety_pass, "observed": learned["false_accept_rate"],
            "maximum": far_limit,
        },
        "tool_cost_efficiency": {
            "pass": efficiency_pass,
            "observed_s": learned["mean_measured_tool_cost_s"],
            "absolute_maximum_s": cost_limit,
            "baseline_s": baseline["mean_measured_tool_cost_s"],
            "absolute_pass": absolute_cost_pass,
            "lower_than_baseline_pass": relative_cost_pass,
        },
    }
    return {
        "pass": coverage_pass and all(item["pass"] for item in objectives.values()),
        "coverage_pass": coverage_pass,
        "coverage": {
            "common_frames": common_frames,
            "minimum_common_frames": int(coverage_config["minimum_common_frames"]),
            "common_sequences": sorted(common_sequences),
            "required_sequences": sorted(required_sequences),
            "sequences_pass": sequences_pass, "frames_pass": frames_pass,
        },
        "objectives": objectives,
        "pareto_relation": pareto_relation(learned, baseline),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/stage3b_pareto_gate_v1.json"))
    parser.add_argument("--learned-policy", default="learned")
    parser.add_argument("--baseline-policy", default="feedback_rule")
    parser.add_argument("--evaluation-split", choices=("development", "final_blind"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    learned_rows, learned_excluded = retain_audited(
        load_traces(args.learned_root, args.learned_policy),
        load_policy_audits(args.learned_root, args.learned_policy))
    baseline_rows, baseline_excluded = retain_audited(
        load_traces(args.baseline_root, args.baseline_policy),
        load_policy_audits(args.baseline_root, args.baseline_policy))
    common_keys = sorted(set(learned_rows) & set(baseline_rows))
    if not common_keys:
        raise ValueError("learned and baseline traces have no common frames")
    learned_only = sorted(set(learned_rows) - set(baseline_rows))
    baseline_only = sorted(set(baseline_rows) - set(learned_rows))
    learned = frame_arrays({key: learned_rows[key] for key in common_keys},
                           data_root=args.data_root, gt_root=args.gt_root)
    baseline = frame_arrays({key: baseline_rows[key] for key in common_keys},
                            data_root=args.data_root, gt_root=args.gt_root)
    learned_metrics, baseline_metrics = metrics(learned), metrics(baseline)
    statistics = config["statistics"]
    paired = paired_bootstrap(
        learned, baseline, samples=int(statistics["bootstrap_samples"]),
        confidence=float(statistics["confidence_level"]), seed=int(statistics["seed"]))
    sequences = sorted({key[0] for key in common_keys})
    gate = evaluate_gate(learned_metrics, baseline_metrics, paired, config,
                         common_frames=len(common_keys), common_sequences=sequences)
    output = {
        "schema_version": 1,
        "evaluation_split": args.evaluation_split,
        "claim_scope": ("development_diagnostic_only" if args.evaluation_split == "development"
                        else "final_blind_gate"),
        "config_path": str(args.config), "config": config,
        "inputs": {"learned_root": str(args.learned_root),
                   "baseline_root": str(args.baseline_root),
                   "learned_policy": args.learned_policy,
                   "baseline_policy": args.baseline_policy},
        "pairing": {"common_frames": len(common_keys), "common_sequences": sequences,
                    "learned_only_frames": [list(key) for key in learned_only],
                    "baseline_only_frames": [list(key) for key in baseline_only],
                    "excluded_unaudited_sequences": {
                        "learned": learned_excluded, "baseline": baseline_excluded}},
        "metrics": {"learned": learned_metrics, "baseline": baseline_metrics},
        "paired_bootstrap": {
            "samples": int(statistics["bootstrap_samples"]),
            "confidence_level": float(statistics["confidence_level"]), **paired},
        "gate": gate,
        "interpretation": (
            "Passing a development gate authorizes the next experiment only; it is not a final held-out claim."
            if args.evaluation_split == "development" else
            "A final-blind pass is claim-eligible only if the split was frozen before policy selection."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": output["metrics"], "gate": gate,
                      "paired_bootstrap": output["paired_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
