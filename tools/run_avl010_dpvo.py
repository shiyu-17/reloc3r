#!/usr/bin/env python3
"""Run the official DPVO tracker causally on one LILocBench RGB sequence.

This wrapper deliberately reads only the manifest-ordered RGB frames and the
camera intrinsics.  It records the online pose after each call, without using
``DPVO.terminate()`` (which performs an additional batch optimization over the
completed sequence).  The saved trajectory is therefore suitable for the
AVL-010 causal anchor/state evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def pose_finite(pose: np.ndarray | None) -> bool:
    if pose is None:
        return False
    pose = np.asarray(pose)
    return pose.shape == (4, 4) and np.isfinite(pose).all() and np.linalg.norm(
        pose[3] - np.array([0.0, 0.0, 0.0, 1.0])
    ) < 1e-3


def _online_group_pose(slam, timestamp: int):
    """Resolve the current internal SE3 through keyframe/removal deltas."""
    from dpvo.lietorch import SE3

    if timestamp in slam.pg.delta:
        t0, delta = slam.pg.delta[timestamp]
        parent = _online_group_pose(slam, int(t0))
        return delta * parent if parent is not None else None

    if slam.n > 0:
        matches = np.flatnonzero(slam.pg.tstamps_[: slam.n] == timestamp)
        if len(matches):
            return SE3(slam.pg.poses_[int(matches[-1])])
    return None


def run(args: argparse.Namespace) -> None:
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    manifest = json.loads(args.data_root.joinpath(args.sequence, "manifest.json").read_text())
    frames = manifest["frames"]
    intrinsics = np.loadtxt(args.calib, delimiter=" ").astype(np.float32)
    if intrinsics.shape[0] < 4:
        raise ValueError("calibration must contain fx fy cx cy")
    fx, fy, cx, cy = intrinsics[:4]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["LOOP_CLOSURE", "False", "CLASSIC_LOOP_CLOSURE", "False",
                         "BUFFER_SIZE", str(args.buffer_size)])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("AVL-010 DPVO requires CUDA; run outside the CPU sandbox")

    slam = None
    rows = []
    poses = []
    timings = []
    for frame_idx, item in enumerate(frames):
        image_path = args.data_root.joinpath(args.sequence, item["path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        image = image[: h - h % 16, : w - w % 16]
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(device, non_blocking=True)
        kin = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=device)
        if slam is None:
            slam = DPVO(cfg, str(args.network), ht=image.shape[0], wd=image.shape[1], viz=False)

        start = time.perf_counter()
        with torch.inference_mode():
            slam(int(frame_idx), tensor, kin)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        group_pose = _online_group_pose(slam, int(frame_idx))
        if group_pose is None:
            pose = np.full((4, 4), np.nan, dtype=np.float64)
        else:
            # DPVO's public terminate() exports inverse group poses.  Match
            # that convention while keeping this lookup strictly online.
            pose = group_pose.inv().matrix().detach().cpu().numpy().astype(np.float64)
        poses.append(pose)
        timings.append(float(elapsed))
        rows.append({
            "frame_index": int(frame_idx),
            "timestamp": item["timestamp"],
            "online_pose_available": bool(pose_finite(pose)),
            "elapsed_s": float(elapsed),
            "query_gt_read": False,
            "future_frame_read": False,
        })

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    poses_np = np.stack(poses, axis=0)
    np.savez_compressed(
        out_dir / "trajectory_online.npz",
        frame_indices=np.arange(len(frames), dtype=np.int32),
        poses=poses_np,
        valid=np.asarray([pose_finite(p) for p in poses_np], dtype=bool),
        elapsed_s=np.asarray(timings, dtype=np.float64),
    )
    sampled_count = None
    if args.sample_manifest is not None:
        sample = json.loads(args.sample_manifest.read_text())
        full_ts = np.asarray([float(x["timestamp"]) for x in frames], dtype=np.float64)
        sample_ts = np.asarray([float(x["timestamp"]) for x in sample["frames"]], dtype=np.float64)
        source_indices = np.searchsorted(full_ts, sample_ts, side="left")
        source_indices = np.clip(source_indices, 0, len(full_ts) - 1)
        left = np.maximum(source_indices - 1, 0)
        choose_left = np.abs(full_ts[left] - sample_ts) < np.abs(full_ts[source_indices] - sample_ts)
        source_indices[choose_left] = left[choose_left]
        if np.max(np.abs(full_ts[source_indices] - sample_ts)) > 0.02:
            raise RuntimeError("sample manifest timestamps do not match full-rate stream")
        np.savez_compressed(
            out_dir / "trajectory_query.npz",
            frame_indices=np.arange(len(sample_ts), dtype=np.int32),
            source_full_indices=source_indices.astype(np.int32),
            source_timestamps=full_ts[source_indices],
            poses=poses_np[source_indices],
            valid=np.asarray([pose_finite(p) for p in poses_np[source_indices]], dtype=bool),
        )
        sampled_count = int(len(sample_ts))
    metadata = {
        "experiment": "AVL-010",
        "tracker": "DPVO",
        "tracker_source": "princeton-vl/DPVO",
        "sequence": args.sequence,
        "frame_count": len(frames),
        "valid_online_pose_count": int(sum(pose_finite(p) for p in poses_np)),
        "network": str(Path(args.network).resolve()),
        "config": str(Path(args.config).resolve()),
        "calibration": str(Path(args.calib).resolve()),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mean_tracking_elapsed_s": float(np.mean(timings)),
        "median_tracking_elapsed_s": float(np.median(timings)),
        "total_tracking_elapsed_s": float(np.sum(timings)),
        "final_keyframe_count": int(slam.n),
        "final_delta_count": int(len(slam.pg.delta)),
        "final_keyframe_timestamps": [int(x) for x in slam.pg.tstamps_[: slam.n]],
        "delta_timestamps": sorted(int(x) for x in slam.pg.delta.keys()),
        "query_gt_read": False,
        "future_frame_read": False,
        "terminate_called": False,
        "buffer_size": int(args.buffer_size),
        "sampled_query_count": sampled_count,
        "notes": "Online poses are exported after each call; no final batch optimization.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("third_party/DPVO/config/default.yaml"))
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--sample-manifest", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
