#!/usr/bin/env python3
"""AVL-015 Q3: causal Historical-keyframe Relocalization.

Reliable query frames from a prior causal trace form a temporary local map.
The current image is matched to two or more historical keyframes; their stored
global poses triangulate 3-D tracks, and PnP estimates the current global pose.
No query GT, future frame, or future descriptor is used by the route.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import read_manifest  # noqa: E402
from tools.run_avl004_roma_recovery import (  # noqa: E402
    camera_matrix,
    intersect_query,
    intrinsics,
    roma_pair,
    solve_pose,
    triangulate,
)
from romatch import roma_indoor  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_trace(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_keyframes(trace, embeddings, frame_paths, qi, current_embedding, num_refs, min_gap):
    candidates = []
    q = current_embedding.astype(np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    for idx in range(qi):
        row = trace[idx]
        pose = row.get("pose")
        if not row.get("state_updated", False) or pose is None or not row.get("accepted", False):
            continue
        d = embeddings[idx].astype(np.float32)
        d /= max(float(np.linalg.norm(d)), 1e-12)
        candidates.append({"frame_index": idx, "similarity": float(np.dot(q, d)),
                           "pose": pose, "path": str(frame_paths[idx])})
    candidates.sort(key=lambda x: (-x["similarity"], x["frame_index"]))
    selected = []
    for candidate in candidates:
        if all(abs(candidate["frame_index"] - x["frame_index"]) >= min_gap for x in selected):
            selected.append(candidate)
        if len(selected) >= num_refs:
            break
    if len(selected) < num_refs:
        used = {x["frame_index"] for x in selected}
        selected.extend(x for x in candidates if x["frame_index"] not in used)
        selected = selected[:num_refs]
    return selected


def estimate_pose(model, query_path, refs, K, dist, device, args):
    pair_matches = []
    matching_start = time.perf_counter()
    for ref in refs:
        pair_matches.append(roma_pair(model, query_path, Path(ref["path"]), device, args.num_matches))
    matching_s = time.perf_counter() - matching_start
    all3d, all2d, allscore = [], [], []
    pair_stats = []
    tri_start = time.perf_counter()
    for i, j in combinations(range(len(refs)), 2):
        qxy, xa, xb, score = intersect_query(pair_matches[i], pair_matches[j],
                                               radius=args.query_radius,
                                               max_points=args.max_intersections)
        Pi = camera_matrix(np.asarray(refs[i]["pose"], dtype=np.float64), K)
        Pj = camera_matrix(np.asarray(refs[j]["pose"], dtype=np.float64), K)
        xyz, valid = triangulate(Pi, Pj, xa, xb)
        if len(xyz):
            all3d.append(xyz)
            all2d.append(qxy[valid])
            allscore.append(score[valid])
        pair_stats.append({"historical_indices": [refs[i]["frame_index"], refs[j]["frame_index"]],
                           "similarities": [refs[i]["similarity"], refs[j]["similarity"]],
                           "intersections": int(len(qxy)), "triangulated": int(len(xyz))})
    triangulation_s = time.perf_counter() - tri_start
    if not all3d:
        return {"pose": None, "inliers": 0, "median_reprojection_px": None,
                "correspondences": 0, "pair_stats": pair_stats,
                "matching_s": matching_s, "triangulation_s": triangulation_s, "pnp_s": 0.0}
    xyz = np.concatenate(all3d)
    qxy = np.concatenate(all2d)
    scores = np.concatenate(allscore)
    keep = np.argsort(-scores, kind="stable")[:args.max_pnp_points]
    pnp_start = time.perf_counter()
    pose, inliers, reproj = solve_pose(xyz[keep], qxy[keep], K, dist)
    pnp_s = time.perf_counter() - pnp_start
    return {"pose": None if pose is None else pose.tolist(), "inliers": int(inliers),
            "median_reprojection_px": None if not np.isfinite(reproj) else float(reproj),
            "correspondences": int(len(keep)), "pair_stats": pair_stats,
            "matching_s": matching_s, "triangulation_s": triangulation_s, "pnp_s": pnp_s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    p.add_argument("--sequence", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, default=Path("outputs/lilocbench/AVL-007/embeddings"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--indices", nargs="+", type=int)
    p.add_argument("--indices-file", type=Path)
    p.add_argument("--all", action="store_true", help="process every query frame in causal order")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-refs", type=int, default=4)
    p.add_argument("--min-gap", type=int, default=5)
    p.add_argument("--num-matches", type=int, default=6000)
    p.add_argument("--max-intersections", type=int, default=1800)
    p.add_argument("--max-pnp-points", type=int, default=3000)
    p.add_argument("--query-radius", type=float, default=2.0)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    supplied = int(args.indices is not None) + int(args.indices_file is not None) + int(args.all)
    if supplied != 1:
        p.error("provide exactly one of --indices, --indices-file, or --all")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = roma_indoor(device=device, coarse_res=560, upsample_res=864, use_custom_corr=False)
    model.eval()
    _, frames, _, _, seq_root = read_manifest(args.data_root, args.sequence)
    indices = (args.indices if args.indices is not None else
               [int(x) for x in args.indices_file.read_text().split()] if args.indices_file is not None else
               list(range(len(frames))))
    trace = load_trace(args.trace)
    embeddings = np.load(args.embeddings / f"{args.sequence}_cls.npy", mmap_mode="r")
    K, dist = intrinsics(args.data_root / f"{args.sequence}/camera_front/color/intrinsics.yaml")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qi in sorted(indices):
        start = time.perf_counter()
        qpath = frames[qi]["path"]
        refs = load_keyframes(trace, embeddings, [f["path"] for f in frames], qi,
                              embeddings[qi], args.num_refs, args.min_gap)
        if len(refs) < 2:
            result = {"pose": None, "inliers": 0, "median_reprojection_px": None,
                      "correspondences": 0, "pair_stats": [], "matching_s": 0.0,
                      "triangulation_s": 0.0, "pnp_s": 0.0}
        else:
            result = estimate_pose(model, qpath, refs, K, dist, device, args)
        rows.append({"sequence": args.sequence, "query_index": qi,
                     "query_path": str(qpath), "pose_source": "historical_keyframe_relocalization",
                     "pose": result.pop("pose"), "historical_keyframes": refs,
                     "causal": {"query_gt_read": False, "uses_future": False,
                                "selection_uses_gt": False, "max_historical_index": max((r["frame_index"] for r in refs), default=-1)},
                     "evidence": result, "elapsed_s": time.perf_counter() - start})
        print(json.dumps({"query_index": qi, "keyframes": [r["frame_index"] for r in refs],
                          "pose": rows[-1]["pose"] is not None, "inliers": result["inliers"]}), flush=True)
    args.output.write_text("\n".join(json.dumps(x) for x in rows) + "\n")


if __name__ == "__main__":
    main()
