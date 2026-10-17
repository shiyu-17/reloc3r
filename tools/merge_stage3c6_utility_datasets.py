#!/usr/bin/env python3
"""Merge live-cost component datasets without erasing repeated measurements.

The identical policy-visible state can be reached in separate GPU trajectories
while a fresh Geometry call has a slightly different measured latency.  Its
Oracle utility is therefore a sample from the deployment-time cost
distribution, not a contradictory deterministic label.  This merger keeps
those samples separate, records their provenance, and gives each row a unique
training ID.  It must not be used to merge a train split with an evaluation
split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    sources = []
    utility: Dict[str, Any] | None = None
    for source_index, path in enumerate(args.input):
        source_rows = read(path)
        summary_path = path.parent / "summary.json"
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_utility = dict(source_summary["utility"])
        if utility is None:
            utility = source_utility
        elif utility != source_utility:
            raise ValueError(f"utility contract differs for {path}")
        sources.append({"path": str(path), "rows": len(source_rows)})
        for row_index, row in enumerate(source_rows):
            item = dict(row)
            item["state_id"] = f"{row.get('state_id', 'state')}::live_source_{source_index}::{row_index}"
            item["live_cost_source"] = str(path)
            rows.append(item)
    if not rows:
        raise ValueError("no component rows to merge")
    rows.sort(key=lambda row: (str(row["sequence"]), int(row["frame_index"]), str(row["state_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {"schema_version": 1, "name": "live_cost_component_dataset_merge",
               "output": str(args.output), "states": len(rows), "sources": sources,
               "utility": utility,
               "deduplication": "none; repeated live-cost Oracle measurements are preserved",
               "online_boundary": "component labels are evaluator-only; online policy receives only pre-action observation/action features"}
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
