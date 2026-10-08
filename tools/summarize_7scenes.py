#!/usr/bin/env python3
"""Recompute official 7Scenes metrics from Reloc3r evaluator pose files."""
import argparse
from pathlib import Path

import numpy as np

from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='chess')
    parser.add_argument('--seq', type=int, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--poses-root', default='_db-q_pair_info')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.poses_root)
    solver = Reloc3rVisloc()
    terrs, rerrs = [], []
    for fid in range(args.start, args.end):
        databases, relatives = [], []
        for pair_id in range(10):
            folder = root / f'poses_{args.scene}_pair-id={pair_id}'
            databases.append(np.loadtxt(folder / f'seq-{args.seq:02d}_{fid:06d}_pose-db.txt'))
            relatives.append(np.loadtxt(folder / f'seq-{args.seq:02d}_{fid:06d}_pose-q2d.txt'))
        gt = np.loadtxt(root / 'poses_{}_pair-id=9'.format(args.scene)
                       / f'seq-{args.seq:02d}_{fid:06d}_pose-gt.txt')
        pose = solver.motion_averaging(databases, relatives)
        terrs.append(np.linalg.norm(pose[:3, 3] - gt[:3, 3]))
        rerrs.append(get_rot_err(pose[:3, :3], gt[:3, :3]))
    np.savez(args.output, terrs=np.array(terrs), rerrs=np.array(rerrs))
    print(f'wrote {len(terrs)} queries to {args.output}')


if __name__ == '__main__':
    main()
