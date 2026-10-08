#!/usr/bin/env python3
"""Summarize official Reloc3r Chess pose files with the evaluator's math."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


ROOT = Path('_db-q_pair_info')
SCENE = 'chess'
TOPK = 10


def load_pose(pair_id, seq, fid, kind):
    return np.loadtxt(ROOT / f'poses_{SCENE}_pair-id={pair_id}' /
                      f'seq-{seq:02d}_{fid:06d}_pose-{kind}.txt')


def one(sample):
    seq, fid = sample
    poses_db = [load_pose(pid, seq, fid, 'db') for pid in range(TOPK)]
    poses_q2d = [load_pose(pid, seq, fid, 'q2d') for pid in range(TOPK)]
    gt = load_pose(TOPK - 1, seq, fid, 'gt')

    rotations, lines = [], []
    for db, q2d in zip(poses_db, poses_q2d):
        pose_q = db @ q2d
        rotations.append(pose_q[:3, :3])
        lines.append(np.stack((db[:3, 3], pose_q[:3, 3])))
    quat = np.median(Rotation.from_matrix(rotations).as_quat(), axis=0)
    rotation = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()

    lines = np.asarray(lines)
    p, q = lines[:, 0], lines[:, 1]
    direction = q - p
    eye = np.eye(3)[None].repeat(TOPK, axis=0)
    projection = eye - direction[:, :, None] * direction[:, None, :] / np.sum(direction ** 2, axis=1)[:, None, None]
    a = projection.reshape(-1, 3)
    b = (projection @ p[:, :, None]).reshape(-1)
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    center = vt.T @ (np.diag(1 / singular) @ (u.T @ b))

    rel = rotation.T @ gt[:3, :3]
    cos = np.clip((np.trace(rel) - 1) / 2, -1, 1)
    return np.linalg.norm(center - gt[:3, 3]), np.degrees(np.arccos(cos))


if __name__ == '__main__':
    samples = [(seq, fid) for seq in (3, 5) for fid in range(1000)]
    with ProcessPoolExecutor(max_workers=16) as pool:
        result = np.asarray(list(pool.map(one, samples, chunksize=10)))
    np.save('outputs/chess_errors.npy', result)
    print(f'queries={len(result)}')
    print(f'Scene chess median pose error: {np.median(result[:, 0]):.4f} m {np.median(result[:, 1]):.4f} deg')
    print(f'mean pose error: {np.mean(result[:, 0]):.4f} m {np.mean(result[:, 1]):.4f} deg')
