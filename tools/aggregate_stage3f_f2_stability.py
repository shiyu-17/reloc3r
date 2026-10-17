#!/usr/bin/env python3
"""Audit frozen F2 stability against feedback-rule across fixed time blocks.

Success/FAR are read from completed causal collector traces.  Cost is reported
separately because historical multi-GPU runs are not a controlled latency
comparison; a Stage 3F crossover replay supplies that evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluate_stage3b_pareto_gate import frame_arrays, metrics, paired_bootstrap
from tools.evaluate_stage3d_pareto_gate import load_rollouts


DEFAULT_BLOCKS = {
    "100_119": (
        "outputs/lilocbench/stage3d_v2_fresh_collect_r1/f2",
        "outputs/lilocbench/stage3d_v2_feedback_rule_fresh_baseline_r1",
    ),
    "120_139": (
        "outputs/lilocbench/stage3d_geometry_advantage_fresh_base_r1/f2",
        "outputs/lilocbench/stage3d_geometry_advantage_fresh_eval_r1/feedback_rule",
    ),
    "140_154": (
        "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/f2_base_15/f2",
        "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/feedback_rule_15",
    ),
}


def aligned_arrays(f2_root: Path, rule_root: Path, data_root: Path, gt_root: Path):
    f2_rows, rule_rows = load_rollouts(f2_root), load_rollouts(rule_root)
    if set(f2_rows) != set(rule_rows):
        raise ValueError(f"unaligned block: F2-only={set(f2_rows)-set(rule_rows)}, "
                         f"rule-only={set(rule_rows)-set(f2_rows)}")
    common = dict(sorted(f2_rows.items()))
    return (frame_arrays(common, data_root=data_root, gt_root=gt_root),
            frame_arrays({key: rule_rows[key] for key in common}, data_root=data_root, gt_root=gt_root),
            sorted(common))


def aggregate_summaries(items):
    rows = [item["summary"] for item in items]
    frames = sum(int(row["frames"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    correct = sum(int(row["accepted_correct"]) for row in rows)
    false_accepts = sum(int(row["false_accepts"]) for row in rows)
    geometry = sum(int(row["geometry_calls"]) for row in rows)
    weighted_cost = sum(float(row["mean_measured_tool_cost_s"]) * int(row["frames"])
                          for row in rows)
    return {
        "frames": frames, "accepted": accepted, "accepted_correct": correct,
        "success": correct / frames if frames else 0.0,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / accepted if accepted else 0.0,
        "geometry_calls": geometry, "geometry_call_rate": geometry / frames if frames else 0.0,
        "mean_measured_tool_cost_s": weighted_cost / frames if frames else 0.0,
    }


def crossover_cost_audit(root: Path):
    blocks = {}
    for block in DEFAULT_BLOCKS:
        summaries = sorted((root / block).glob("*/summary.json"))
        if not summaries:
            raise FileNotFoundError(f"missing crossover summaries below {root / block}")
        entries = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
        f2, rule = [], []
        order = {"f2_first": [], "rule_first": []}
        for entry in entries:
            policies = entry["config"]["policies"]
            if policies not in (["learned", "feedback_rule"], ["feedback_rule", "learned"]):
                raise ValueError(f"unexpected crossover order: {policies}")
            sequence = entry["config"]["sequences"][0]
            item_f2 = {"sequence": sequence, "summary": entry["policies"]["learned"][sequence]["summary"]}
            item_rule = {"sequence": sequence, "summary": entry["policies"]["feedback_rule"][sequence]["summary"]}
            f2.append(item_f2)
            rule.append(item_rule)
            order["f2_first" if policies[0] == "learned" else "rule_first"].append((item_f2, item_rule))
        order_result = {}
        for name, pairs in order.items():
            if not pairs:
                continue
            left, right = zip(*pairs)
            left_total, right_total = aggregate_summaries(left), aggregate_summaries(right)
            order_result[name] = {
                "f2_cost_s": left_total["mean_measured_tool_cost_s"],
                "feedback_rule_cost_s": right_total["mean_measured_tool_cost_s"],
                "f2_minus_rule_cost_s": (left_total["mean_measured_tool_cost_s"] -
                                           right_total["mean_measured_tool_cost_s"]),
            }
        f2_total, rule_total = aggregate_summaries(f2), aggregate_summaries(rule)
        blocks[block] = {
            "sequences": sorted(item["sequence"] for item in f2),
            "f2": f2_total, "feedback_rule": rule_total,
            "f2_minus_rule_cost_s": (f2_total["mean_measured_tool_cost_s"] -
                                       rule_total["mean_measured_tool_cost_s"]),
            "order_breakdown": order_result,
        }
    return blocks


def paired_crossover_cost(root: Path, reverse_root: Path):
    """Average AB and BA costs for the identical (block, sequence) unit."""
    blocks = {}
    for block in DEFAULT_BLOCKS:
        def read_entries(base):
            result = {}
            for path in sorted((base / block).glob("*/summary.json")):
                entry = json.loads(path.read_text(encoding="utf-8"))
                sequence = entry["config"]["sequences"][0]
                result[sequence] = entry
            return result
        first, second = read_entries(root), read_entries(reverse_root)
        if set(first) != set(second):
            raise ValueError(f"AB/BA sequence mismatch in {block}: {set(first)} vs {set(second)}")
        rows = []
        for sequence in sorted(first):
            a, b = first[sequence], second[sequence]
            order_a, order_b = a["config"]["policies"], b["config"]["policies"]
            if order_a[0] == order_b[0]:
                raise ValueError(f"orders are not reversed for {block}/{sequence}: {order_a}, {order_b}")
            def cost(entry, policy):
                return float(entry["policies"][policy][sequence]["summary"]["mean_measured_tool_cost_s"])
            f2_a, f2_b = cost(a, "learned"), cost(b, "learned")
            rule_a, rule_b = cost(a, "feedback_rule"), cost(b, "feedback_rule")
            rows.append({
                "sequence": sequence,
                "f2_ab_cost_s": f2_a, "f2_ba_cost_s": f2_b,
                "feedback_ab_cost_s": rule_a, "feedback_ba_cost_s": rule_b,
                "f2_mean_cost_s": (f2_a + f2_b) / 2.0,
                "feedback_mean_cost_s": (rule_a + rule_b) / 2.0,
            })
        f2_mean = float(np.mean([row["f2_mean_cost_s"] for row in rows]))
        rule_mean = float(np.mean([row["feedback_mean_cost_s"] for row in rows]))
        blocks[block] = {
            "per_sequence": rows, "f2_mean_cost_s": f2_mean,
            "feedback_rule_mean_cost_s": rule_mean,
            "f2_minus_rule_cost_s": f2_mean - rule_mean,
        }
    all_rows = [row for block in blocks.values() for row in block["per_sequence"]]
    f2_mean = float(np.mean([row["f2_mean_cost_s"] for row in all_rows]))
    rule_mean = float(np.mean([row["feedback_mean_cost_s"] for row in all_rows]))
    return {"blocks": blocks, "pooled": {"f2_mean_cost_s": f2_mean,
            "feedback_rule_mean_cost_s": rule_mean, "f2_minus_rule_cost_s": f2_mean - rule_mean}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--crossover-root", type=Path,
                        help="same-GPU F2/rule crossover output root; required for latency conclusion")
    parser.add_argument("--crossover-reverse-root", type=Path,
                        help="AB/BA reverse-order crossover root; removes order warm-up bias")
    args = parser.parse_args()

    all_f2, all_rule, blocks = [], [], {}
    for index, (name, (f2_path, rule_path)) in enumerate(DEFAULT_BLOCKS.items()):
        f2, rule, keys = aligned_arrays(Path(f2_path), Path(rule_path), args.data_root, args.gt_root)
        blocks[name] = {
            "f2_root": f2_path, "feedback_rule_root": rule_path,
            "frames": len(keys), "sequences": sorted({key[0] for key in keys}),
            "f2": metrics(f2), "feedback_rule": metrics(rule),
            "paired_bootstrap": paired_bootstrap(f2, rule, samples=args.samples,
                                                    confidence=0.95, seed=1777 + index),
            "success_difference_pp": 100.0 * float(f2["success"].mean() - rule["success"].mean()),
        }
        all_f2.append(f2)
        all_rule.append(rule)
    merged_f2 = {key: np.concatenate([part[key] for part in all_f2]) for key in all_f2[0]}
    merged_rule = {key: np.concatenate([part[key] for part in all_rule]) for key in all_rule[0]}
    differences = [row["success_difference_pp"] for row in blocks.values()]
    report = {
        "schema_version": 1,
        "purpose": "frozen F2 temporal stability audit; no policy tuning",
        "blocks": blocks,
        "pooled": {
            "f2": metrics(merged_f2), "feedback_rule": metrics(merged_rule),
            "paired_bootstrap": paired_bootstrap(merged_f2, merged_rule, samples=args.samples,
                                                    confidence=0.95, seed=2026),
            "success_difference_pp": 100.0 * float(merged_f2["success"].mean() - merged_rule["success"].mean()),
        },
        "success_stability": {
            "per_block_difference_pp": differences,
            "same_direction": len({np.sign(value) for value in differences}) == 1,
            "interpretation": "success advantage is stable only when all held-out temporal blocks have the same sign",
        },
        "latency_boundary": "historical costs are descriptive only; use the same-GPU crossover replay for latency claims",
    }
    if args.crossover_root is not None:
        report["same_gpu_crossover_cost"] = crossover_cost_audit(args.crossover_root)
    if args.crossover_reverse_root is not None:
        if args.crossover_root is None:
            parser.error("--crossover-reverse-root requires --crossover-root")
        report["order_debiased_same_gpu_cost"] = paired_crossover_cost(
            args.crossover_root, args.crossover_reverse_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
