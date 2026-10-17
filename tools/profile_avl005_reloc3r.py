#!/usr/bin/env python3
"""Fixed, no-GT Reloc3r timing microbenchmark for AVL-005 cost accounting.

This measures image preparation and the relative-pose network separately on
frozen mapping-only Top-10 pairs.  It deliberately reports sampled per-query
costs: retrieval and geometry are accounted for by their own full-frame runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.eval_lilocbench_reloc3r import (
    make_batch,
    read_manifest,
    resize_for_reloc3r,
)
from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def sample_indices(count: int, requested: int) -> list[int]:
    # Deterministic, endpoint-inclusive temporal coverage; no GT involved.
    return np.unique(np.linspace(0, count - 1, min(count, requested), dtype=int)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-sequence", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    _, mapping, _, intrinsics, _ = read_manifest(args.data_root, "mapping")
    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    model.eval()
    report = {"protocol": "frozen mapping-only Top-10; query GT not loaded",
              "samples_per_sequence": args.samples_per_sequence,
              "batch_size": args.batch_size, "sequences": {}}
    for sequence in SEQUENCES:
        _, query, _, _, _ = read_manifest(args.data_root, sequence)
        retrieved = np.load(args.retrieval_root / sequence / "retrieval.npz")
        rows = []
        for query_index in sample_indices(len(query), args.samples_per_sequence):
            pairs = [(mapping[int(ref)]["path"], query[query_index]["path"])
                     for ref in retrieved["mapping_indices"][query_index]]
            preprocess_s = 0.0
            inference_s = 0.0
            for start in range(0, len(pairs), args.batch_size):
                t0 = time.perf_counter()
                batch = make_batch(pairs[start:start + args.batch_size], intrinsics, device)
                preprocess_s += time.perf_counter() - t0
                if args.warmup and not rows and start == 0:
                    for _ in range(args.warmup):
                        inference_relpose(batch, model, device, use_amp=True)
                    torch.cuda.synchronize(device)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                inference_relpose(batch, model, device, use_amp=True)
                torch.cuda.synchronize(device)
                inference_s += time.perf_counter() - t0
            rows.append({"query_index": query_index, "pairs": len(pairs),
                         "preprocess_s": preprocess_s, "inference_s": inference_s,
                         "query_gt_read": False})
        report["sequences"][sequence] = {"rows": rows,
            "median_preprocess_s": float(np.median([r["preprocess_s"] for r in rows])),
            "median_inference_s": float(np.median([r["inference_s"] for r in rows])),
            "mean_preprocess_s": float(np.mean([r["preprocess_s"] for r in rows])),
            "mean_inference_s": float(np.mean([r["inference_s"] for r in rows])),
            "query_gt_read": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("median_preprocess_s", "median_inference_s")}
                      for k, v in report["sequences"].items()}, indent=2))


if __name__ == "__main__":
    main()
