#!/usr/bin/env python3
"""Freeze the no-GT Manager geometry-call indices for AVL-004 A.5."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manager-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for trace_path in sorted(args.manager_dir.glob("*/manager_trace.jsonl")):
        seq = trace_path.parent.name
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        indices = [int(row["query_index"]) for row in rows if row["action"] == "call_geometry"]
        out = args.output_dir / f"{seq}_geometry_indices.txt"
        out.write_text(" ".join(str(i) for i in indices) + "\n", encoding="utf-8")
        shard_paths = []
        for shard in range(args.shards):
            part = indices[shard::args.shards]
            shard_path = args.output_dir / f"{seq}_geometry_indices_shard{shard:02d}_of_{args.shards:02d}.txt"
            shard_path.write_text(" ".join(str(i) for i in part) + "\n", encoding="utf-8")
            shard_paths.append(str(shard_path))
        summary[seq] = {
            "frames": len(rows), "geometry_frames": len(indices),
            "indices_path": str(out), "selection_uses_gt": False,
            "source_trace": str(trace_path), "shards": shard_paths,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
