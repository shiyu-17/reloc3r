#!/usr/bin/env python3
"""Map-based 2D/3D recovery with RoMa correspondences for AVL-004.

This is an auditable recovery feasibility test, not a standard HLoc model
because LILocBench does not ship a COLMAP reconstruction.  Mapping camera
poses are map assets; query GT is never loaded.  For each query, RoMa matches
the query to several retrieved mapping images.  Matching pixels in two map
views are triangulated with the known mapping poses, then a query PnP pose is
estimated.  The evaluator consumes the saved poses later and is the only code
that reads query GT.
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
import yaml
from scipy.spatial import cKDTree

from romatch import roma_indoor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical,
    read_manifest,
    tum_camera_poses,
)
from tools.run_avl003_ablations import load_relative  # noqa: E402


def seed_everything(seed: int) -> None:
    """Make sampling and OpenCV RANSAC auditable across repeated runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def intrinsics(path: Path):
    data = yaml.safe_load(path.read_text())
    K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(data.get("distortion_coefficients", []), dtype=np.float64).reshape(-1)
    return K, d


def camera_matrix(c2w, K):
    r = c2w[:3, :3].T
    t = -r @ c2w[:3, 3]
    return K @ np.column_stack([r, t])


@torch.inference_mode()
def roma_pair(model, query_path, ref_path, device, num_matches):
    warp, certainty = model.match(str(query_path), str(ref_path), device=device)
    matches, score = model.sample(warp, certainty, num=num_matches)
    q = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if q is None or r is None:
        raise FileNotFoundError(f"missing image: {query_path} or {ref_path}")
    qxy, rxy = model.to_pixel_coordinates(matches, q.shape[0], q.shape[1],
                                            r.shape[0], r.shape[1])
    return (qxy.detach().cpu().numpy().astype(np.float32),
            rxy.detach().cpu().numpy().astype(np.float32),
            score.detach().cpu().numpy().astype(np.float32))


def intersect_query(a, b, radius=2.0, max_points=2000):
    """Intersect two q->map correspondence sets by query pixel."""
    if not len(a[0]) or not len(b[0]):
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty(0, np.float32)
    tree = cKDTree(b[0])
    dist, idx = tree.query(a[0], distance_upper_bound=radius)
    valid = np.isfinite(dist) & (idx < len(b[0]))
    if not np.any(valid):
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty(0, np.float32)
    order = np.argsort(-(a[2][valid] * b[2][idx[valid]]), kind="stable")[:max_points]
    qa = a[0][valid][order]
    rb = b[1][idx[valid][order]]
    ra = a[1][valid][order]
    scores = (a[2][valid][order] * b[2][idx[valid][order]])
    return qa, ra, rb, scores


def triangulate(Pa, Pb, xa, xb):
    if len(xa) < 2:
        return np.empty((0, 3), np.float64), np.zeros(len(xa), dtype=bool)
    homog = cv2.triangulatePoints(Pa, Pb, xa.T, xb.T).T
    good = np.abs(homog[:, 3]) > 1e-8
    xyz = homog[good, :3] / homog[good, 3:4]
    # Reject numerically extreme points; PnP/RANSAC handles remaining noise.
    finite = np.all(np.isfinite(xyz), axis=1) & (np.linalg.norm(xyz, axis=1) < 100.0)
    selected = np.zeros(len(xa), dtype=bool)
    selected[np.flatnonzero(good)[finite]] = True
    return xyz[finite], selected


def solve_pose(points3d, points2d, K, dist):
    if len(points3d) < 6:
        return None, 0, 0.0
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d.astype(np.float64), points2d.astype(np.float64), K, dist,
        iterationsCount=2000, reprojectionError=3.0, confidence=0.999,
        # ITERATIVE is slower than SQPNP but avoids the frequent mirrored
        # solution on the largely planar indoor surfaces in this benchmark.
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < 6:
        return None, 0, float("inf")
    R, _ = cv2.Rodrigues(rvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = (-R.T @ tvec).reshape(3)
    proj, _ = cv2.projectPoints(points3d[inliers[:, 0]], rvec, tvec, K, dist)
    err = np.linalg.norm(proj[:, 0, :] - points2d[inliers[:, 0]], axis=1)
    return c2w, int(len(inliers)), float(np.median(err))


def run_one(model, args, mapping_frames, mapping_pose, K, dist, rel_item, qpath, refs, device):
    matching_start = time.perf_counter()
    pair_matches = []
    for ref_idx in refs:
        qxy = mapping_frames[int(ref_idx)]["path"]
        pair_matches.append(roma_pair(model, qpath, qxy, device, args.num_matches))
    matching_s = time.perf_counter() - matching_start
    P = [camera_matrix(mapping_pose[int(ref)], K) for ref in refs]
    all3d, all2d, allscore = [], [], []
    pair_stats = []
    triangulation_start = time.perf_counter()
    for i, j in combinations(range(len(refs)), 2):
        qxy, xa, xb, score = intersect_query(pair_matches[i], pair_matches[j], args.query_radius, args.max_intersections)
        xyz, valid = triangulate(P[i], P[j], xa, xb)
        if len(xyz):
            all3d.append(xyz); all2d.append(qxy[valid]); allscore.append(score[valid])
        pair_stats.append({"reference_indices": [int(refs[i]), int(refs[j])], "intersections": int(len(qxy)), "triangulated": int(len(xyz))})
    triangulation_s = time.perf_counter() - triangulation_start
    if not all3d:
        return {"pose": None, "inliers": 0, "median_reprojection_px": None, "correspondences": 0,
                "pair_stats": pair_stats, "matching_s": matching_s,
                "triangulation_s": triangulation_s, "pnp_s": 0.0}
    xyz = np.concatenate(all3d); qxy = np.concatenate(all2d); scores = np.concatenate(allscore)
    # Prefer high-confidence tracks while retaining enough spatial coverage.
    keep = np.argsort(-scores, kind="stable")[:args.max_pnp_points]
    pnp_start = time.perf_counter()
    pose, inliers, reproj = solve_pose(xyz[keep], qxy[keep], K, dist)
    pnp_s = time.perf_counter() - pnp_start
    return {"pose": None if pose is None else pose.tolist(), "inliers": inliers,
            "median_reprojection_px": None if not np.isfinite(reproj) else reproj,
            "correspondences": int(len(keep)), "pair_stats": pair_stats,
            "matching_s": matching_s, "triangulation_s": triangulation_s,
            "pnp_s": pnp_s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--mapping-gt", type=Path, required=True)
    p.add_argument("--manager-trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--indices", nargs="+", type=int)
    p.add_argument("--indices-file", type=Path)
    p.add_argument("--override-refs", nargs="+", type=int,
                   help="diagnostic-only explicit map refs; never used by Manager")
    p.add_argument("--num-refs", type=int, default=4)
    p.add_argument("--reference-jsonl", type=Path,
                   help="optional no-GT reranking output with query_index/selected_refs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-matches", type=int, default=6000)
    p.add_argument("--max-intersections", type=int, default=1800)
    p.add_argument("--max-pnp-points", type=int, default=3000)
    p.add_argument("--query-radius", type=float, default=2.0)
    p.add_argument("--seed", type=int, required=True,
                   help="seed for Python/NumPy/Torch/CUDA/OpenCV; required for reproducibility")
    args = p.parse_args()
    if (args.indices is None) == (args.indices_file is None):
        p.error("provide exactly one of --indices or --indices-file")
    if args.indices_file is not None:
        args.indices = [int(x) for x in args.indices_file.read_text().split()]
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = roma_indoor(device=device, coarse_res=560, upsample_res=864,
                        use_custom_corr=False)
    model.eval()
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, "mapping")
    _, query_frames, _, _, _ = read_manifest(args.data_root, args.sequence)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, db_times, base_to_front_rgb_optical(db_root), 0.1)
    K, dist = intrinsics(args.data_root / "mapping/camera_front/color/intrinsics.yaml")
    relative = load_relative(args.results_root / args.sequence, len(query_frames))
    trace = [json.loads(x) for x in args.manager_trace.read_text().splitlines()]
    reranked = None
    if args.reference_jsonl:
        reranked = {int(x["query_index"]): x["selected_refs"]
                     for x in (json.loads(line) for line in args.reference_jsonl.read_text().splitlines())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qi in args.indices:
        frame_start = time.perf_counter()
        row = trace[qi]
        # Retrieval order is frozen in AVL-002; the manager trace only adds
        # evidence, so recomputing these indices cannot use query GT.
        if args.override_refs:
            refs = np.asarray(args.override_refs, dtype=np.int64)
        elif reranked is not None:
            refs = np.asarray(reranked[qi], dtype=np.int64)
        else:
            refs = np.asarray(relative[qi]["indices"][:args.num_refs], dtype=np.int64)
        result = run_one(model, args, db_frames, mapping_pose, K, dist, relative[qi], Path(query_frames[qi]["path"]), refs, device)
        rows.append({"query_index": int(qi), "action": row["action"], "reference_indices": [int(x) for x in refs], **result,
                     "elapsed_s": time.perf_counter() - frame_start,
                     "query_gt_read": False, "selection_uses_gt": False,
                     "seed": int(args.seed), "model": "romatch==0.1.2/roma_indoor"})
        print(json.dumps(rows[-1])[:1000], flush=True)
    args.output.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
