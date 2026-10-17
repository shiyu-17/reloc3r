#!/usr/bin/env python3
"""Aggregate a Stage 3D feature-ladder fresh-split evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.analyze_stage3c6_decisions import audit


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_level(level_dir: Path) -> dict:
    summaries = sorted(level_dir.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no summaries under {level_dir}")
    totals = {
        "frames": 0,
        "accepted": 0,
        "accepted_correct": 0,
        "false_accepts": 0,
        "geometry_calls": 0,
        "tool_actions": 0.0,
        "measured_tool_cost_s": 0.0,
    }
    sequences = {}
    decisions = []
    for summary_path in summaries:
        summary = load_json(summary_path)
        if len(summary["rollout_summary"]) != 1:
            raise ValueError(f"expected one sequence in {summary_path}")
        sequence, row = next(iter(summary["rollout_summary"].items()))
        frames = int(row["frames"])
        sequences[sequence] = row
        totals["frames"] += frames
        totals["accepted"] += int(row["accepted"])
        totals["accepted_correct"] += int(row["accepted_correct"])
        totals["false_accepts"] += int(row["false_accepts"])
        totals["geometry_calls"] += int(row["geometry_calls"])
        totals["tool_actions"] += float(row["mean_tool_actions"]) * frames
        totals["measured_tool_cost_s"] += float(row["mean_measured_tool_cost_s"]) * frames
        rollout_path = summary_path.parent / f"{sequence}_rollout.jsonl"
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            frame = json.loads(line)
            decisions.extend(
                decision for decision in frame.get("decisions", [])
                if decision.get("oracle_label") is not None
            )
    frames = totals["frames"]
    accepted = totals["accepted"]
    return {
        "sequence_count": len(summaries),
        "sequences": sequences,
        "aggregate": {
            **totals,
            "success": totals["accepted_correct"] / frames,
            "false_accept_rate": None if accepted == 0 else totals["false_accepts"] / accepted,
            "coverage": accepted / frames,
            "accepted_accuracy": None if accepted == 0 else totals["accepted_correct"] / accepted,
            "geometry_call_rate": totals["geometry_calls"] / frames,
            "mean_tool_actions": totals["tool_actions"] / frames,
            "mean_measured_tool_cost_s": totals["measured_tool_cost_s"] / frames,
        },
        "opportunity_audit": audit(decisions),
    }


def assert_aligned(levels: dict[str, dict]) -> None:
    """Reject comparisons that do not evaluate exactly the same frames.

    A sequential policy can alter its own state distribution, but it must still
    be evaluated over the same named sequence and frame count as every
    comparator.  Without this check, an accidental ``--max-frames`` mismatch
    can silently become an invalid aggregate comparison.
    """
    names = list(levels)
    if len(names) < 2:
        return
    reference_name = names[0]
    reference = levels[reference_name]["sequences"]
    reference_frames = {name: int(row["frames"]) for name, row in reference.items()}
    for name in names[1:]:
        candidate = levels[name]["sequences"]
        candidate_frames = {sequence: int(row["frames"]) for sequence, row in candidate.items()}
        if candidate_frames != reference_frames:
            raise ValueError(
                "unaligned evaluation protocol: "
                f"{reference_name}={reference_frames}, {name}={candidate_frames}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", action="append", default=[], metavar="NAME=SUBDIR",
                        help="Optional level mapping relative to root; repeatable. Defaults to F0--F3.")
    parser.add_argument("--require-aligned", action="store_true",
                        help="Fail unless every level has identical sequence names and frame counts.")
    args = parser.parse_args()
    mappings = args.level or [f"{name}={name}" for name in ("f0", "f1", "f2", "f3")]
    levels = {}
    for mapping in mappings:
        if "=" not in mapping:
            parser.error(f"--level must be NAME=SUBDIR, got {mapping!r}")
        name, subdir = mapping.split("=", 1)
        if not name:
            parser.error("--level name cannot be empty")
        levels[name] = summarize_level(args.root / subdir)
    if args.require_aligned:
        assert_aligned(levels)
    report = {
        "schema_version": 1,
        "purpose": "Stage 3D fresh-split feature-ladder aggregate and opportunity audit",
        "root": str(args.root),
        "levels": levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
