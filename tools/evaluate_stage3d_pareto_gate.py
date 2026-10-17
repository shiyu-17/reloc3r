#!/usr/bin/env python3
"""Evaluate Stage 3D policies and feedback-rule on paired collector rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from tools.evaluate_stage3b_pareto_gate import (
    evaluate_gate, frame_arrays, metrics, paired_bootstrap,
)


FrameKey = Tuple[str, int]


def load_rollouts(root: Path) -> Dict[FrameKey, Dict[str, Any]]:
    rows: Dict[FrameKey, Dict[str, Any]] = {}
    paths = sorted(root.glob("*/*_rollout.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no collector rollout files below {root}")
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sequence"]), int(row["frame_index"]))
            if key in rows:
                raise ValueError(f"duplicate frame {key} at {path}:{number}")
            causal = row.get("causal", {})
            if any(bool(causal.get(name, False)) for name in
                   ("query_gt_read", "uses_future", "selection_uses_gt")):
                raise ValueError(f"online causal boundary violated at {key}")
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage3b_pareto_gate_v1.json"))
    parser.add_argument("--evaluation-split", choices=("development", "final_blind"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    learned_rows, baseline_rows = load_rollouts(args.learned_root), load_rollouts(args.baseline_root)
    common = sorted(set(learned_rows) & set(baseline_rows))
    if not common:
        raise ValueError("learned and feedback-rule rollouts have no common frames")
    learned_arrays = frame_arrays({key: learned_rows[key] for key in common},
                                  data_root=args.data_root, gt_root=args.gt_root)
    baseline_arrays = frame_arrays({key: baseline_rows[key] for key in common},
                                   data_root=args.data_root, gt_root=args.gt_root)
    learned, baseline = metrics(learned_arrays), metrics(baseline_arrays)
    statistics = config["statistics"]
    paired = paired_bootstrap(learned_arrays, baseline_arrays,
                              samples=int(statistics["bootstrap_samples"]),
                              confidence=float(statistics["confidence_level"]),
                              seed=int(statistics["seed"]))
    sequences = sorted({key[0] for key in common})
    gate = evaluate_gate(learned, baseline, paired, config,
                         common_frames=len(common), common_sequences=sequences)
    report = {
        "schema_version": 1,
        "purpose": "paired Stage 3D collector Pareto Gate",
        "evaluation_split": args.evaluation_split,
        "inputs": {"learned_root": str(args.learned_root), "baseline_root": str(args.baseline_root)},
        "pairing": {"common_frames": len(common), "common_sequences": sequences,
                    "learned_only_frames": [list(key) for key in sorted(set(learned_rows) - set(baseline_rows))],
                    "baseline_only_frames": [list(key) for key in sorted(set(baseline_rows) - set(learned_rows))]},
        "metrics": {"learned": learned, "baseline": baseline},
        "paired_bootstrap": {"samples": int(statistics["bootstrap_samples"]),
                             "confidence_level": float(statistics["confidence_level"]), **paired},
        "gate": gate,
        "causal_audit": "all collector records passed online causal-field checks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "paired_bootstrap": report["paired_bootstrap"],
                      "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
