#!/usr/bin/env python3
"""Compute causal adjacent-frame Reloc3r edges for AVL-009.

This stage is deliberately GT-free.  For a query sequence it evaluates only
ordered pairs ``(I[t-1], I[t])`` and writes the relative transform returned by
the existing Reloc3r relpose model.  The convention is the same one used by
``eval_lilocbench_reloc3r.py``: if ``P_db`` is a world-from-camera pose, the
query pose candidate is ``P_db @ relative``.  The sequence evaluator later
propagates ``P_world[t-1] @ relative[t]`` without reading query GT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_lilocbench_reloc3r import make_batch, read_manifest  # noqa: E402
from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_pose(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    return bool(np.linalg.norm(pose[3] - np.array([0.0, 0.0, 0.0, 1.0])) < 1e-3)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    _, frames, _, intrinsics, _ = read_manifest(args.data_root, args.query_sequence)
    current_indices = np.arange(1, len(frames), dtype=np.int32)
    selected = current_indices[current_indices % args.num_shards == args.shard_index]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    estimates = []
    batch_rows = []
    total_start = time.perf_counter()
    for start in range(0, len(selected), args.batch_size):
        batch_indices = selected[start:start + args.batch_size]
        pairs = [(frames[int(index) - 1]["path"], frames[int(index)]["path"])
                 for index in batch_indices]
        batch = make_batch(pairs, intrinsics, device)
        batch_start = time.perf_counter()
        with torch.inference_mode():
            pose = inference_relpose(batch, model, device, use_amp=args.amp)
        batch_elapsed = time.perf_counter() - batch_start
        pose_np = pose.detach().cpu().numpy().astype(np.float64)
        if pose_np.shape != (len(batch_indices), 4, 4):
            raise RuntimeError(f"Unexpected Reloc3r edge shape {pose_np.shape}")
        estimates.append(pose_np)
        batch_rows.append({
            "first_query_index": int(batch_indices[0]),
            "last_query_index": int(batch_indices[-1]),
            "count": int(len(batch_indices)),
            "elapsed_s": float(batch_elapsed),
        })

    if estimates:
        relative = np.concatenate(estimates, axis=0)
    else:
        relative = np.empty((0, 4, 4), dtype=np.float64)
    valid = np.asarray([finite_pose(pose) for pose in relative], dtype=bool)
    total_elapsed = time.perf_counter() - total_start
    np.savez_compressed(
        output_dir / f"edges_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz",
        query_indices=selected.astype(np.int32),
        relative_poses=relative,
        valid=valid,
    )
    manifest = {
        "experiment": "AVL-009",
        "query_sequence": args.query_sequence,
        "query_frames": len(frames),
        "edge_definition": "(I[t-1], I[t])",
        "relative_pose_convention": "world_pose[t] = world_pose[t-1] @ relative[t]",
        "mapping_or_query_gt_read": False,
        "query_gt_read": False,
        "future_frame_read": False,
        "seed": args.seed,
        "model": "Reloc3rRelpose(img_size=512)",
        "checkpoint": "siyan824/reloc3r-512",
        "device": str(device),
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "edge_count": int(len(selected)),
        "valid_edge_count": int(valid.sum()),
        "invalid_edge_count": int((~valid).sum()),
        "elapsed_s": float(total_elapsed),
        "batches": batch_rows,
    }
    (output_dir / f"edges_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--query-sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    run(args)


if __name__ == "__main__":
    main()
