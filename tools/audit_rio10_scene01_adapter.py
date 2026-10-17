#!/usr/bin/env python3
"""Build a read-only RIO10 scene01 adapter manifest and coordinate audit.

The manifest is deliberately *not* a training-label export.  RIO10 supplies
per-frame matrices, but their camera/world direction must be verified by a
backend relative-pose check before they are used as evaluator GT.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLIT = {
    "mapping": ["seq01_01"],
    "validation": ["seq01_02"],
    "test": ["seq01_03", "seq01_04", "seq01_05", "seq01_06"],
}
FRAME = re.compile(r"frame-(\d+)\.color\.jpg$")


def read_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"expected 4x4 pose: {path}")
    if not np.all(np.isfinite(pose)) or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"invalid homogeneous pose: {path}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"non-orthonormal rotation: {path}")
    if np.linalg.det(rotation) < 0.99:
        raise ValueError(f"improper rotation: {path}")
    return pose


def indexed_files(sequence_root: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    images = {}
    for image in sequence_root.glob("frame-*.color.jpg"):
        match = FRAME.match(image.name)
        if match:
            images[int(match.group(1))] = image
    poses = {int(path.name.split(".")[0].split("-")[1]): path
             for path in sequence_root.glob("frame-*.pose.txt")}
    if not images:
        raise ValueError(f"no RGB frames in {sequence_root}")
    return images, poses


def sequence_entry(scene_root: Path, name: str) -> dict[str, Any]:
    root = scene_root / "seq01" / name
    images, pose_paths = indexed_files(root)
    shared = sorted(set(images) & set(pose_paths))
    rgb_only, pose_only = sorted(set(images) - set(pose_paths)), sorted(set(pose_paths) - set(images))
    if rgb_only or pose_only:
        return {
            "sequence": name, "root": str(root), "frame_count": len(images),
            "pose_count": len(pose_paths), "gt_available": False,
            "rgb_without_pose_count": len(rgb_only), "pose_without_rgb_count": len(pose_only),
            "rgb_without_pose_examples": rgb_only[:10],
            "pose_without_rgb_examples": pose_only[:10],
            "image_pose_pairs": [],
        }
    pairs = [(index, images[index], pose_paths[index]) for index in shared]
    poses = [read_pose(pose) for _, _, pose in pairs]
    centers = np.asarray([pose[:3, 3] for pose in poses])
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(0)
    return {
        "sequence": name,
        "root": str(root),
        "frame_count": len(pairs),
        "pose_count": len(pairs), "gt_available": True,
        "first_frame": pairs[0][0], "last_frame": pairs[-1][0],
        "contiguous_indices": [pairs[0][0], pairs[-1][0]] == [0, len(pairs) - 1],
        "image_pose_pairs": [{"frame_index": index, "image_path": str(image),
                                "pose_path": str(pose)} for index, image, pose in pairs],
        "translation_step_median": float(np.median(step)) if len(step) else 0.0,
        "translation_step_max": float(np.max(step)) if len(step) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/liusy/datasets/RIO10/scene01"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    camera = yaml.safe_load((args.root / "seq01" / "seq01_01" / "camera.yaml").read_text())
    model = camera["camera_intrinsics"]["model"]
    if len(model) != 4:
        raise ValueError("expected fx, fy, cx, cy intrinsic model")
    records = {role: [sequence_entry(args.root, sequence) for sequence in sequences]
               for role, sequences in SPLIT.items()}
    payload = {
        "schema_version": 1,
        "dataset": "RIO10", "scene_id": "scene01",
        "purpose": "adapter readiness only; no GT labels are emitted",
        "camera": {"width": int(camera["camera_intrinsics"]["width"]),
                   "height": int(camera["camera_intrinsics"]["height"]),
                   "fx": float(model[0]), "fy": float(model[1]),
                   "cx": float(model[2]), "cy": float(model[3]),
                   "distortion": list(camera["camera_intrinsics"].get("distortion", ()))},
        "split": records,
        "coordinate_contract": {
            "matrix_storage": "per-frame 4x4 rigid transform",
            "pose_direction": "UNRESOLVED",
            "required_next_check": "Compare an estimated relative pose for a held-out pair against both C2W and W2C compositions; choose only the convention that is consistent across sequences.",
            "label_permission": False,
        },
        "local_gt_status": {
            "mapping_validation_pose_complete": all(
                entry["gt_available"] for role in ("mapping", "validation") for entry in records[role]),
            "test_pose_complete": all(entry["gt_available"] for entry in records["test"]),
            "consequence": "No RIO10 test labels or scene-level training/evaluation claim are permitted unless every selected query sequence has poses and the coordinate convention is verified.",
        },
        "scene_disjoint_status": "one physical scene only; adapter smoke allowed, train/test Agent claim prohibited",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({role: [entry["frame_count"] for entry in entries]
                      for role, entries in records.items()}, indent=2))


if __name__ == "__main__":
    main()
