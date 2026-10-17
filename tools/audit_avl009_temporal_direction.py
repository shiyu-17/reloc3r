#!/usr/bin/env python3
"""Post-hoc evaluator-only audit of the AVL-009 relative-pose direction.

The online baseline fixes the direction from the existing Reloc3r convention
(``mapping_pose @ relative``).  This audit uses query GT only after the run to
compare that convention with its inverse; it never changes the online trace or
thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_avl009_sequence import finite_pose, distance  # noqa: E402
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, read_manifest, tum_camera_poses, pose_error  # noqa: E402


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def load_edges(root: Path, sequence: str):
    output = {}
    for path in sorted((root / sequence).glob("edges_shard_*.npz")):
        data = np.load(path)
        for idx, pose, valid in zip(data["query_indices"], data["relative_poses"], data["valid"]):
            output[int(idx)] = pose if bool(valid) and finite_pose(pose) else None
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gt-delta", type=float, default=0.1)
    args = parser.parse_args()
    results = {}
    for sequence in SEQUENCES:
        _, _, times, _, query_root = read_manifest(args.data_root, sequence)
        gt, delta = tum_camera_poses(args.gt_dir / f"{sequence}_gt_poses.txt", times,
                                     base_to_front_rgb_optical(query_root), args.max_gt_delta)
        edges = load_edges(args.temporal_root, sequence)
        forward = [gt[0]]
        inverse = [gt[0]]
        for index in range(1, len(gt)):
            edge = edges.get(index)
            forward.append(forward[-1] @ edge if edge is not None and finite_pose(forward[-1]) else None)
            inverse.append(inverse[-1] @ np.linalg.inv(edge) if edge is not None and finite_pose(inverse[-1]) else None)
        rows = {}
        for name, predictions in (("forward_convention", forward), ("inverse_convention", inverse)):
            errors = [pose_error(pred, truth) for pred, truth in zip(predictions, gt) if finite_pose(pred)]
            rows[name] = {
                "pose_available": len(errors),
                "median_translation_m": float(np.median([x[0] for x in errors])) if errors else None,
                "median_rotation_deg": float(np.median([x[1] for x in errors])) if errors else None,
                "last_translation_m": float(errors[-1][0]) if errors else None,
                "last_rotation_deg": float(errors[-1][1]) if errors else None,
            }
        results[sequence] = {"frames": len(gt), "gt_match_delta_max_s": float(delta.max()), "directions": rows,
                             "online_direction_used": "forward_convention", "query_gt_read_by_online_trace": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"experiment": "AVL-009", "protocol": "post-hoc evaluator-only direction audit",
                                       "sequences": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
