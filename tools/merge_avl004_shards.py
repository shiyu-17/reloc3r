#!/usr/bin/env python3
"""Merge completed AVL-004 route JSONL shards with a strict query audit."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", required=True)
    p.add_argument("--expected-indices", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = {int(x) for x in args.expected_indices.read_text().split()}
    rows = []
    for path in sorted(Path().glob(args.input_glob)):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_index = {}
    duplicates = []
    for row in rows:
        qi = int(row["query_index"])
        if qi in by_index:
            duplicates.append(qi)
        by_index[qi] = row
    missing = sorted(expected - set(by_index))
    unexpected = sorted(set(by_index) - expected)
    if duplicates or missing or unexpected:
        raise SystemExit(json.dumps({"duplicates": duplicates, "missing": missing, "unexpected": unexpected}))
    merged = [by_index[i] for i in sorted(expected)]
    for row in merged:
        if row.get("query_gt_read") is not False or row.get("selection_uses_gt") is not False:
            raise SystemExit(f"GT audit failed for query {row['query_index']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged), encoding="utf-8")
    print(json.dumps({"frames": len(merged), "output": str(args.output),
                      "query_gt_read": False, "selection_uses_gt": False}, indent=2))


if __name__ == "__main__":
    main()
