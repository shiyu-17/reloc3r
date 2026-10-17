#!/usr/bin/env python3
"""Create the frozen Stage 3 closeout summary from completed artifacts.

This script is deliberately dependency-light and read-only with respect to
existing experiment outputs.  It validates the three aligned temporal blocks
and writes one compact summary used by the next-stage protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


BLOCKS = {
    "100_119": {
        "f2": "outputs/lilocbench/stage3d_v2_fresh_collect_r1/f2",
        "legacy_rule": "outputs/lilocbench/stage3d_v2_feedback_rule_fresh_baseline_r1",
    },
    "120_139": {
        "f2": "outputs/lilocbench/stage3d_geometry_advantage_fresh_base_r1/f2",
        "legacy_rule": "outputs/lilocbench/stage3d_geometry_advantage_fresh_eval_r1/feedback_rule",
    },
    "140_154": {
        "f2": "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/f2_base_15/f2",
        "legacy_rule": "outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/feedback_rule_15",
    },
}
ALIGNED_ROOT = "outputs/lilocbench/stage3f_terminal_aligned_feedback_r1"
SEQUENCES = {"static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"}


def load_rows(root: Path) -> list[Dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rollout = data.get("rollout_summary", {})
        if len(rollout) != 1:
            raise ValueError(f"expected one sequence summary in {path}")
        sequence, row = next(iter(rollout.items()))
        rows.append({"sequence": str(sequence), **dict(row)})
    if {row["sequence"] for row in rows} != SEQUENCES:
        raise ValueError(f"incomplete sequence set below {root}")
    return rows


def aggregate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    frames = sum(int(row["frames"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    correct = sum(int(row["accepted_correct"]) for row in rows)
    false_accepts = sum(int(row["false_accepts"]) for row in rows)
    geometry = sum(int(row["geometry_calls"]) for row in rows)
    weighted_cost = sum(float(row["mean_measured_tool_cost_s"]) * int(row["frames"])
                        for row in rows)
    return {
        "frames": frames, "accepted": accepted, "accepted_correct": correct,
        "success": correct / frames, "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / accepted if accepted else 0.0,
        "geometry_calls": geometry, "geometry_call_rate": geometry / frames,
        "descriptive_mean_tool_cost_s": weighted_cost / frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pooled: Dict[str, list[Dict[str, Any]]] = {
        "f2": [], "legacy_rule": [], "terminal_aligned_rule": []}
    blocks = {}
    for block, roots in BLOCKS.items():
        block_rows = {
            "f2": load_rows(Path(roots["f2"])),
            "legacy_rule": load_rows(Path(roots["legacy_rule"])),
            "terminal_aligned_rule": load_rows(Path(ALIGNED_ROOT) / block),
        }
        expected_frames = 60 if block == "140_154" else 80
        metrics = {name: aggregate(rows) for name, rows in block_rows.items()}
        if any(row["frames"] != expected_frames for row in metrics.values()):
            raise ValueError(f"unaligned frame count in {block}: {metrics}")
        blocks[block] = metrics
        for name, rows in block_rows.items():
            pooled[name].extend(rows)
    pooled_metrics = {name: aggregate(rows) for name, rows in pooled.items()}
    report = {
        "schema_version": 1,
        "status": "stage3_closed",
        "blocks": blocks,
        "pooled": pooled_metrics,
        "terminal_interface_confound": {
            "legacy_rule_minus_f2_correct_frames": (
                pooled_metrics["legacy_rule"]["accepted_correct"] -
                pooled_metrics["f2"]["accepted_correct"]),
            "terminal_aligned_rule_minus_f2_correct_frames": (
                pooled_metrics["terminal_aligned_rule"]["accepted_correct"] -
                pooled_metrics["f2"]["accepted_correct"]),
            "interpretation": (
                "The apparent legacy-rule success advantage disappears after terminal alignment; "
                "Stage 3 does not justify more Geometry/controller tuning on these splits."),
        },
        "decision": {
            "freeze": ["F2", "feedback_rule", "source-aware verifier", "current tool backends"],
            "stop": ["same-split controller tuning", "unconstrained Geometry-rate optimization",
                     "Qwen/Qwen-VL/RL escalation"],
            "next": "active-evidence realizable-value validation on failure-triggered episodes",
        },
        "latency_boundary": (
            "Costs are descriptive only because the three policies were not rerun in one "
            "counterbalanced same-process experiment."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pooled": report["pooled"],
                      "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
