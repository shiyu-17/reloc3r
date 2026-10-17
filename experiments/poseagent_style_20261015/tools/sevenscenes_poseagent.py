"""7Scenes map/query preparation for the causal PoseAgent experiments.

The official split is preserved inside each room: TrainSplit frames form the
map and TestSplit frames form queries.  Query poses are written to a separate
evaluator sidecar and are never loaded by :func:`make_backend`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend

SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
# Kinect intrinsics supplied with the original 7Scenes benchmark.
K = np.array([[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _split_sequences(scene_root: Path, split: str) -> list[str]:
    name = "TrainSplit.txt" if split == "train" else "TestSplit.txt"
    values = scene_root.joinpath(name).read_text(encoding="utf-8").split()
    return [f"seq-{int(value.removeprefix('sequence')):02d}" for value in values]


def _frames(scene_root: Path, split: str, stride: int) -> list[dict[str, Any]]:
    if stride < 1:
        raise ValueError("stride must be positive")
    records: list[dict[str, Any]] = []
    for sequence in _split_sequences(scene_root, split):
        paths = sorted(scene_root.joinpath(sequence).glob("frame-*.color.png"))
        for position, image in enumerate(paths):
            if position % stride:
                continue
            pose_path = image.with_name(image.name.replace(".color.png", ".pose.txt"))
            depth_path = image.with_name(image.name.replace(".color.png", ".depth.png"))
            pose = np.loadtxt(pose_path, dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue
            records.append({"path": str(image.resolve()), "depth_path": str(depth_path.resolve()),
                            "timestamp": float(len(records)), "sequence": sequence, "pose": pose})
    return records


def descriptor(path: str) -> np.ndarray:
    """Deterministic cheap retrieval descriptor; no pose or test-frame mixing."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    small = cv2.resize(image, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = small - small.mean()
    return small.reshape(-1) / max(float(np.linalg.norm(small)), 1e-6)


def prepare_scene(data_root: Path, output_root: Path, scene: str, *, map_stride: int = 10,
                  query_stride: int = 10, topk: int = 12) -> Path:
    if scene not in SCENES:
        raise ValueError(f"unknown 7Scenes scene: {scene}")
    maps = _frames(Path(data_root) / scene, "train", map_stride)
    queries = _frames(Path(data_root) / scene, "test", query_stride)
    if len(maps) < 2 or not queries:
        raise ValueError(f"insufficient frames for {scene}")
    map_descriptors = np.stack([descriptor(row["path"]) for row in maps])
    query_descriptors = np.stack([descriptor(row["path"]) for row in queries])
    scores = query_descriptors @ map_descriptors.T
    k = min(int(topk), len(maps))
    indices = np.argsort(-scores, axis=1)[:, :k].astype(np.int64)
    selected = np.take_along_axis(scores, indices, axis=1).astype(np.float32)
    destination = Path(output_root) / scene
    destination.mkdir(parents=True, exist_ok=True)
    # The online manifest intentionally has no query poses.
    manifest = {"schema_version": 1, "dataset": "7scenes", "scene_id": scene,
                "map_split": "official_train", "query_split": "official_test",
                "intrinsics": K.tolist(),
                "mapping_frames": [{key: value for key, value in row.items() if key != "pose"} for row in maps],
                "query_frames": [{key: value for key, value in row.items() if key != "pose"} for row in queries],
                "retrieval": "normalized_32x24_grayscale_cosine", "query_gt_in_online_manifest": False}
    destination.joinpath("online_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    np.save(destination / "mapping_poses.npy", np.stack([row["pose"] for row in maps]))
    # Explicitly evaluator-only sidecar, excluded from all online APIs.
    np.save(destination / "query_gt.npy", np.stack([row["pose"] for row in queries]))
    np.savez_compressed(destination / "retrieval.npz", mapping_indices=indices, mapping_scores=selected)
    return destination


def make_backend(prepared_root: Path, scene: str, *, nfeatures: int = 1200) -> CpuOrbLocalizationBackend:
    root = Path(prepared_root) / scene
    manifest = json.loads(root.joinpath("online_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "7scenes" or manifest.get("query_gt_in_online_manifest") is not False:
        raise ValueError("invalid or unsafe 7Scenes online manifest")
    retrieval = np.load(root / "retrieval.npz")
    return CpuOrbLocalizationBackend.from_frames(
        sequence=scene, mapping_frames=manifest["mapping_frames"], query_frames=manifest["query_frames"],
        mapping_poses=np.load(root / "mapping_poses.npy"), retrieval_indices=retrieval["mapping_indices"],
        retrieval_scores=retrieval["mapping_scores"], camera_matrix=np.asarray(manifest["intrinsics"]),
        nfeatures=nfeatures)
