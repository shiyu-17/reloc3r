#!/usr/bin/env python3
"""Fixed Reloc3r diagnostic baseline for the public LILocBench sequences.

The mapping run is the only image database.  Query GT is loaded exclusively by
the ``analyse`` stage, after pose candidates have been written.  This makes the
database/query boundary explicit and leaves an auditable artifact for each
reference pose prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

# Make the repository importable when this file is executed as
# ``python tools/eval_lilocbench_reloc3r.py`` from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from reloc3r.image_retrieval.netvlad import NetVLAD
from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.image import ImgNorm


CAMERA_FRAME = "camera_front_color_optical_frame"


def read_manifest(root: Path, sequence: str):
    sequence_root = root / sequence
    manifest = json.loads((sequence_root / "manifest.json").read_text(encoding="utf-8"))
    frames = []
    for item in manifest["frames"]:
        frames.append({
            "timestamp": float(item["timestamp"]),
            "path": sequence_root / item["path"],
        })
    timestamps = np.array([frame["timestamp"] for frame in frames], dtype=np.float64)
    intrinsics = np.asarray(
        yaml.safe_load((sequence_root / "camera_front/color/intrinsics.yaml").read_text())["K"],
        dtype=np.float32,
    ).reshape(3, 3)
    return manifest, frames, timestamps, intrinsics, sequence_root


def transform_from_yaml(entry):
    quaternion = [entry["rotation"][key] for key in ("x", "y", "z", "w")]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = [entry["translation"][key] for key in ("x", "y", "z")]
    return matrix


def base_to_front_rgb_optical(sequence_root: Path):
    edges = yaml.safe_load((sequence_root / "transformations.yaml").read_text())
    graph = {}
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        parent_from_child = transform_from_yaml(edge)
        graph.setdefault(parent, []).append((child, parent_from_child))
        graph.setdefault(child, []).append((parent, np.linalg.inv(parent_from_child)))

    queue = [("base_link", np.eye(4, dtype=np.float64))]
    visited = set()
    while queue:
        frame, base_from_frame = queue.pop(0)
        if frame == CAMERA_FRAME:
            return base_from_frame
        if frame in visited:
            continue
        visited.add(frame)
        for child, frame_from_child in graph.get(frame, []):
            if child not in visited:
                queue.append((child, base_from_frame @ frame_from_child))
    raise RuntimeError(f"Cannot compose base_link -> {CAMERA_FRAME}")


def tum_camera_poses(gt_path: Path, rgb_timestamps: np.ndarray, base_from_camera: np.ndarray, max_delta_s: float):
    raw = np.loadtxt(gt_path, dtype=np.float64)
    gt_timestamps = raw[:, 0]
    positions = raw[:, 1:4]
    quaternions = raw[:, 4:8]
    poses = []
    deltas = []
    for timestamp in rgb_timestamps:
        right = np.searchsorted(gt_timestamps, timestamp)
        choices = [index for index in (right - 1, right) if 0 <= index < len(gt_timestamps)]
        index = min(choices, key=lambda candidate: abs(gt_timestamps[candidate] - timestamp))
        delta = abs(gt_timestamps[index] - timestamp)
        if delta > max_delta_s:
            raise RuntimeError(
                f"No GT within {max_delta_s:.3f}s for RGB {timestamp:.9f}; nearest delta={delta:.6f}s"
            )
        world_from_base = np.eye(4, dtype=np.float64)
        world_from_base[:3, :3] = Rotation.from_quat(quaternions[index]).as_matrix()
        world_from_base[:3, 3] = positions[index]
        poses.append(world_from_base @ base_from_camera)
        deltas.append(delta)
    return np.asarray(poses), np.asarray(deltas)


def retrieval_tensor(path: Path, device: torch.device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read {path}")
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)
    minimum, maximum = tensor.amin(dim=(1, 2, 3), keepdim=True), tensor.amax(dim=(1, 2, 3), keepdim=True)
    return (tensor - minimum) / (maximum - minimum).clamp_min(1.0)


@torch.no_grad()
def descriptors(extractor, frames, device):
    output = []
    for frame in tqdm(frames, desc="NetVLAD descriptors"):
        output.append(extractor({"image": retrieval_tensor(frame["path"], device)})["global_descriptor"].cpu().numpy()[0])
    return np.asarray(output, dtype=np.float32)


def stage_retrieve(args):
    device = torch.device(args.device)
    _, db_frames, _, _, _ = read_manifest(args.data_root, args.mapping_sequence)
    _, query_frames, _, _, _ = read_manifest(args.data_root, args.query_sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = NetVLAD(NetVLAD.default_conf).eval().to(device)
    db_desc = descriptors(extractor, db_frames, device)
    query_desc = descriptors(extractor, query_frames, device)
    scores = query_desc @ db_desc.T
    topk_indices = np.argsort(-scores, axis=1)[:, :args.topk]
    topk_scores = np.take_along_axis(scores, topk_indices, axis=1)
    np.savez_compressed(
        args.output_dir / "retrieval.npz",
        mapping_indices=topk_indices.astype(np.int32),
        mapping_scores=topk_scores.astype(np.float32),
    )
    (args.output_dir / "retrieval_manifest.json").write_text(json.dumps({
        "mapping_sequence": args.mapping_sequence,
        "query_sequence": args.query_sequence,
        "topk": args.topk,
        "mapping_frames": len(db_frames),
        "query_frames": len(query_frames),
        "database_only": True,
    }, indent=2) + "\n", encoding="utf-8")


def resize_for_reloc3r(path: Path, intrinsics: np.ndarray, resolution=(512, 384)):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    margin_x, margin_y = min(cx, width - cx), min(cy, height - cy)
    if margin_x <= width / 5 or margin_y <= height / 5:
        raise RuntimeError(f"Principal point is unsuitable for crop: {path}")
    image = image.crop((cx - margin_x, cy - margin_y, cx + margin_x, cy + margin_y))
    cropped = intrinsics.copy()
    cropped[0, 2] -= cx - margin_x
    cropped[1, 2] -= cy - margin_y
    scale = max(resolution[0] / image.size[0], resolution[1] / image.size[1]) + 1e-8
    resized_size = tuple(np.floor(np.asarray(image.size) * scale).astype(int))
    image = image.resize(resized_size, Image.Resampling.LANCZOS)
    cropped[:2, :] *= scale
    left = int(round(cropped[0, 2] - resolution[0] / 2))
    top = int(round(cropped[1, 2] - resolution[1] / 2))
    image = image.crop((left, top, left + resolution[0], top + resolution[1]))
    cropped[0, 2] -= left
    cropped[1, 2] -= top
    return ImgNorm(image), cropped.astype(np.float32)


def make_batch(pairs, intrinsics, device):
    db_images, q_images, db_intrinsics, q_intrinsics = [], [], [], []
    for db_path, q_path in pairs:
        db_image, db_k = resize_for_reloc3r(db_path, intrinsics)
        q_image, q_k = resize_for_reloc3r(q_path, intrinsics)
        db_images.append(db_image)
        q_images.append(q_image)
        db_intrinsics.append(db_k)
        q_intrinsics.append(q_k)
    height, width = db_images[0].shape[-2:]
    def view(images, matrices):
        return {
            "img": torch.stack(images).to(device, non_blocking=True),
            "true_shape": torch.tensor([[height, width]] * len(images), dtype=torch.int32, device=device),
            "camera_intrinsics": torch.from_numpy(np.stack(matrices)).to(device, non_blocking=True),
        }
    return view(db_images, db_intrinsics), view(q_images, q_intrinsics)


def stage_pose(args):
    device = torch.device(args.device)
    _, db_frames, _, intrinsics, _ = read_manifest(args.data_root, args.mapping_sequence)
    _, query_frames, _, _, _ = read_manifest(args.data_root, args.query_sequence)
    retrieved = np.load(args.output_dir / "retrieval.npz")
    indices = retrieved["mapping_indices"]
    selected_queries = np.arange(len(query_frames))[np.arange(len(query_frames)) % args.num_shards == args.shard_index]
    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    candidates = np.full((len(selected_queries), args.topk, 4, 4), np.nan, dtype=np.float64)
    for out_index, query_index in enumerate(tqdm(selected_queries, desc=f"Reloc3r shard {args.shard_index}")):
        pairs = [(db_frames[db_index]["path"], query_frames[query_index]["path"])
                 for db_index in indices[query_index]]
        estimates = []
        for start in range(0, len(pairs), args.batch_size):
            batch = make_batch(pairs[start:start + args.batch_size], intrinsics, device)
            pose = inference_relpose(batch, model, device, use_amp=args.amp).detach().cpu().numpy()
            estimates.append(pose)
        relative_poses = np.concatenate(estimates)
        # Query GT is not loaded in this stage.  Mapping poses are attached during analysis.
        candidates[out_index] = relative_poses
    np.savez_compressed(
        args.output_dir / f"relative_poses_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz",
        query_indices=selected_queries.astype(np.int32),
        mapping_indices=indices[selected_queries].astype(np.int32),
        relative_poses=candidates,
    )


def rotation_error(first, second):
    return float(np.degrees(Rotation.from_matrix(first.T @ second).magnitude()))


def pose_error(predicted, ground_truth):
    return float(np.linalg.norm(predicted[:3, 3] - ground_truth[:3, 3])), rotation_error(predicted[:3, :3], ground_truth[:3, :3])


def aggregate(absolute_candidates):
    if len(absolute_candidates) == 1:
        return absolute_candidates[0]
    return Reloc3rVisloc().motion_averaging(
        [pose for pose in absolute_candidates], [np.linalg.inv(db) @ pose for db, pose in zip(absolute_candidates, absolute_candidates)]
    )


def average_from_relative(mapping_poses, relative_poses):
    candidates = np.asarray([db @ relative for db, relative in zip(mapping_poses, relative_poses)])
    if len(candidates) == 1:
        return candidates[0], candidates
    # Reconstruct the relative poses expected by the existing motion averager.
    return Reloc3rVisloc().motion_averaging(list(mapping_poses), list(relative_poses)), candidates


def candidate_dispersion(candidates):
    if len(candidates) < 2:
        return 0.0, 0.0
    translations, rotations = [], []
    for first, second in combinations(candidates, 2):
        translations.append(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        rotations.append(rotation_error(first[:3, :3], second[:3, :3]))
    return float(np.median(translations)), float(np.median(rotations))


def failure_runs(successes, timestamps):
    runs, begin = [], None
    for index, success in enumerate(successes):
        if not success and begin is None:
            begin = index
        if (success or index == len(successes) - 1) and begin is not None:
            end = index - 1 if success else index
            runs.append({"start_index": begin, "end_index": end, "frames": end - begin + 1,
                         "seconds": float(timestamps[end] - timestamps[begin])})
            begin = None
    return runs


def stage_analyse(args):
    _, db_frames, db_times, _, db_root = read_manifest(args.data_root, args.mapping_sequence)
    _, query_frames, query_times, _, query_root = read_manifest(args.data_root, args.query_sequence)
    db_pose, db_delta = tum_camera_poses(args.mapping_gt, db_times, base_to_front_rgb_optical(db_root), args.max_gt_delta)
    query_pose, query_delta = tum_camera_poses(args.query_gt, query_times, base_to_front_rgb_optical(query_root), args.max_gt_delta)
    retrieved = np.load(args.output_dir / "retrieval.npz")
    ref_indices, scores = retrieved["mapping_indices"], retrieved["mapping_scores"]
    all_relative = {}
    for shard in range(args.num_shards):
        artifact = np.load(args.output_dir / f"relative_poses_shard_{shard:02d}_of_{args.num_shards:02d}.npz")
        for local, query_index in enumerate(artifact["query_indices"]):
            all_relative[int(query_index)] = artifact["relative_poses"][local]
    if len(all_relative) != len(query_frames):
        raise RuntimeError(f"Expected {len(query_frames)} query outputs, found {len(all_relative)}")

    rows, predicted_10, successes = [], [], []
    for query_index, frame in enumerate(query_frames):
        rel = all_relative[query_index]
        mapping = db_pose[ref_indices[query_index]]
        final_poses, candidates = average_from_relative(mapping, rel)
        k_predictions = {}
        for k in (1, 3, 5, 10):
            if k <= len(candidates):
                k_predictions[k] = candidates[0] if k == 1 else Reloc3rVisloc().motion_averaging(
                    list(mapping[:k]), list(rel[:k])
                )
        translation_error, rotation_error_deg = pose_error(final_poses, query_pose[query_index])
        conflict_translation, conflict_rotation = candidate_dispersion(candidates)
        success = translation_error <= 0.25 and rotation_error_deg <= 5.0
        successes.append(success)
        predicted_10.append(final_poses)
        rows.append({
            "query_index": query_index,
            "timestamp": frame["timestamp"],
            "gt_delta_s": query_delta[query_index],
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error_deg,
            "success_0.25m_5deg": success,
            "candidate_translation_dispersion_m": conflict_translation,
            "candidate_rotation_dispersion_deg": conflict_rotation,
            "reference_indices": ref_indices[query_index].tolist(),
            "reference_timestamps": db_times[ref_indices[query_index]].tolist(),
            "reference_scores": scores[query_index].tolist(),
            "topk_pose_delta_from_k10": {
                str(k): {"translation_m": pose_error(k_predictions[k], final_poses)[0],
                         "rotation_deg": pose_error(k_predictions[k], final_poses)[1]}
                for k in k_predictions
            },
            "topk_success": {str(k): bool(pose_error(k_predictions[k], query_pose[query_index])[0] <= 0.25 and
                                           pose_error(k_predictions[k], query_pose[query_index])[1] <= 5.0)
                             for k in k_predictions},
        })

    predicted_10, successes = np.asarray(predicted_10), np.asarray(successes, dtype=bool)
    jumps = []
    for index in range(1, len(predicted_10)):
        predicted_motion = np.linalg.inv(predicted_10[index - 1]) @ predicted_10[index]
        gt_motion = np.linalg.inv(query_pose[index - 1]) @ query_pose[index]
        jump_translation, jump_rotation = pose_error(predicted_motion, gt_motion)
        jumps.append((jump_translation, jump_rotation))
        rows[index]["trajectory_jump_translation_m"] = jump_translation
        rows[index]["trajectory_jump_rotation_deg"] = jump_rotation
    jump_values = np.asarray(jumps) if jumps else np.empty((0, 2))
    def rate(threshold_t, threshold_r):
        return float(np.mean((np.array([row["translation_error_m"] for row in rows]) <= threshold_t) &
                             (np.array([row["rotation_error_deg"] for row in rows]) <= threshold_r)))
    runs = failure_runs(successes, query_times)
    summary = {
        "mapping_sequence": args.mapping_sequence,
        "query_sequence": args.query_sequence,
        "frames": len(rows),
        "gt_match_delta_s": {"mapping_max": float(db_delta.max()), "query_max": float(query_delta.max())},
        "translation_error_m": {"median": float(np.median([row["translation_error_m"] for row in rows]))},
        "rotation_error_deg": {"median": float(np.median([row["rotation_error_deg"] for row in rows]))},
        "success_rate": {"0.25m_5deg": rate(.25, 5), "0.5m_10deg": rate(.5, 10), "1m_15deg": rate(1, 15)},
        "failure_runs": {"count": len(runs), "median_frames": float(np.median([run["frames"] for run in runs])) if runs else 0.0,
                         "p95_frames": float(np.percentile([run["frames"] for run in runs], 95)) if runs else 0.0,
                         "max_frames": max((run["frames"] for run in runs), default=0),
                         "max_seconds": max((run["seconds"] for run in runs), default=0.0)},
        "trajectory_jump": {"median_translation_m": float(np.median(jump_values[:, 0])) if len(jump_values) else 0.0,
                            "median_rotation_deg": float(np.median(jump_values[:, 1])) if len(jump_values) else 0.0,
                            "rate_gt_1m_or_20deg": float(np.mean((jump_values[:, 0] > 1) | (jump_values[:, 1] > 20))) if len(jump_values) else 0.0},
        "candidate_conflict": {"median_translation_m": float(np.median([row["candidate_translation_dispersion_m"] for row in rows])),
                               "median_rotation_deg": float(np.median([row["candidate_rotation_dispersion_deg"] for row in rows]))},
    }
    with (args.output_dir / "frame_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "failure_runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_index", "timestamp", "translation_error_m", "rotation_error_deg",
                                                    "success_0.25m_5deg", "candidate_translation_dispersion_m",
                                                    "candidate_rotation_dispersion_deg", "trajectory_jump_translation_m",
                                                    "trajectory_jump_rotation_deg"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(json.dumps(summary, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("retrieve", "pose", "analyse"), required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--mapping-sequence", default="mapping")
    result.add_argument("--query-sequence", default="static_0")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mapping-gt", type=Path)
    result.add_argument("--query-gt", type=Path)
    result.add_argument("--max-gt-delta", type=float, default=.03)
    result.add_argument("--topk", type=int, default=10)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--amp", action="store_true")
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--num-shards", type=int, default=1)
    return result


def main():
    args = parser().parse_args()
    if args.stage == "retrieve":
        stage_retrieve(args)
    elif args.stage == "pose":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        stage_pose(args)
    else:
        if args.mapping_gt is None or args.query_gt is None:
            raise ValueError("analyse requires --mapping-gt and --query-gt")
        stage_analyse(args)


if __name__ == "__main__":
    main()
