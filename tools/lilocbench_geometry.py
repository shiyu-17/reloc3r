"""RoMa matching and map-view geometry used by the LILocBench v1 backend.

These functions are unchanged from the retired AVL-004 implementation.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


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
    if not len(a[0]) or not len(b[0]):
        empty = np.empty((0, 2), np.float32)
        return empty, empty, empty, np.empty(0, np.float32)
    tree = cKDTree(b[0])
    dist, idx = tree.query(a[0], distance_upper_bound=radius)
    valid = np.isfinite(dist) & (idx < len(b[0]))
    if not np.any(valid):
        empty = np.empty((0, 2), np.float32)
        return empty, empty, empty, np.empty(0, np.float32)
    order = np.argsort(-(a[2][valid] * b[2][idx[valid]]), kind="stable")[:max_points]
    qa = a[0][valid][order]
    rb = b[1][idx[valid][order]]
    ra = a[1][valid][order]
    scores = a[2][valid][order] * b[2][idx[valid][order]]
    return qa, ra, rb, scores


def triangulate(Pa, Pb, xa, xb):
    if len(xa) < 2:
        return np.empty((0, 3), np.float64), np.zeros(len(xa), dtype=bool)
    homog = cv2.triangulatePoints(Pa, Pb, xa.T, xb.T).T
    good = np.abs(homog[:, 3]) > 1e-8
    xyz = homog[good, :3] / homog[good, 3:4]
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
