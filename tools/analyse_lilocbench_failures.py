#!/usr/bin/env python3
"""Turn AVL-002 frame metrics into auditable temporal/spatial failure evidence.

All GT-derived quantities in this script are post-hoc labels.  They are never
read by retrieval or Reloc3r inference.  Semantic labels are deliberately
operational hypotheses and must be checked against the emitted contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

# Allow direct execution as ``python tools/analyse_lilocbench_failures.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical,
    read_manifest,
    tum_camera_poses,
)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else float(default)


def image_features(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"laplacian_var": math.nan, "edge_density": math.nan,
                "intensity_mean": math.nan, "intensity_std": math.nan}
    small = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(small, cv2.CV_64F).var()
    edges = cv2.Canny(small, 50, 150)
    return {
        "laplacian_var": float(lap),
        "edge_density": float(np.mean(edges > 0)),
        "intensity_mean": float(np.mean(small)),
        "intensity_std": float(np.std(small)),
    }


def image_difference(first: Path, second: Path):
    a = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return math.nan
    a = cv2.resize(a, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Per-image centering reduces the effect of exposure changes.
    a -= a.mean()
    b -= b.mean()
    return float(np.mean(np.abs(a - b)) / 255.0)


def relative_frame_path(path: Path, root: Path) -> str:
    """Serialize a manifest path relative to its sequence root."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def pose_yaw(pose):
    # World z is the floor-normal in the LILocBench coordinate convention.
    return float(Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX")[0])


def angle_delta(first, second):
    return float(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())


def load_poses(data_root: Path, sequence: str, gt_path: Path, max_delta: float):
    _, frames, times, _, root = read_manifest(data_root, sequence)
    pose, delta = tum_camera_poses(gt_path, times, base_to_front_rgb_optical(root), max_delta)
    return frames, times, root, pose, delta


def thresholds(rows):
    return {
        "candidate_translation_p75": percentile([r["candidate_translation_dispersion_m"] for r in rows], 75),
        "candidate_rotation_p75": percentile([r["candidate_rotation_dispersion_deg"] for r in rows], 75),
        "ref_distance_p75": percentile([r["top1_reference_distance_m"] for r in rows], 75),
        "appearance_p75": percentile([r["appearance_change"] for r in rows], 75),
        "laplacian_p25": percentile([r["laplacian_var"] for r in rows], 25),
    }


def label_row(row, threshold):
    tags = []
    if row["top1_reference_distance_m"] > max(2.0, threshold["ref_distance_p75"]):
        tags.append("reference_mismatch")
    if (row["candidate_translation_dispersion_m"] > max(2.0, threshold["candidate_translation_p75"]) or
            row["candidate_rotation_dispersion_deg"] > max(10.0, threshold["candidate_rotation_p75"])):
        tags.append("candidate_conflict")
    if (row.get("trajectory_jump_translation_m", 0.0) > 1.0 or
            row.get("trajectory_jump_rotation_deg", 0.0) > 20.0):
        tags.append("trajectory_jump")
    if row["turn_motion_deg"] > 10.0 or row["step_distance_m"] > 1.0:
        tags.append("turn_motion")
    if row["laplacian_var"] < threshold["laplacian_p25"]:
        tags.append("low_texture_or_blur")
    if row["appearance_change"] > threshold["appearance_p75"]:
        tags.append("appearance_change")
    if not tags:
        tags.append("unclassified")
    priority = ("reference_mismatch", "candidate_conflict", "trajectory_jump",
                "turn_motion", "appearance_change", "low_texture_or_blur", "unclassified")
    primary = next(item for item in priority if item in tags)
    return primary, tags


def spatial_and_temporal(rows, grid_size=2.0):
    for row in rows:
        row["grid_x"] = int(math.floor(row["gt_x_m"] / grid_size))
        row["grid_y"] = int(math.floor(row["gt_y_m"] / grid_size))
    cells = {}
    for row in rows:
        key = f"{row['grid_x']},{row['grid_y']}"
        cells.setdefault(key, []).append(row)
    spatial = []
    for key, values in sorted(cells.items()):
        spatial.append({
            "cell": key,
            "frames": len(values),
            "failure_frames": int(sum(not x["success"] for x in values)),
            "failure_rate": float(np.mean([not x["success"] for x in values])),
            "median_translation_error_m": float(np.median([x["translation_error_m"] for x in values])),
            "median_candidate_dispersion_m": float(np.median([x["candidate_translation_dispersion_m"] for x in values])),
            "median_reference_distance_m": float(np.median([x["top1_reference_distance_m"] for x in values])),
            "dominant_failure_type": max((x["primary_failure_type"] for x in values if not x["success"]),
                                          key=lambda t: sum(y["primary_failure_type"] == t and not y["success"] for y in values),
                                          default="none"),
        })
    temporal = []
    for start in range(0, len(rows), 20):
        values = rows[start:start + 20]
        temporal.append({
            "start_index": start,
            "end_index": start + len(values) - 1,
            "start_timestamp": values[0]["timestamp"],
            "end_timestamp": values[-1]["timestamp"],
            "frames": len(values),
            "failure_rate": float(np.mean([not x["success"] for x in values])),
            "median_translation_error_m": float(np.median([x["translation_error_m"] for x in values])),
            "median_candidate_dispersion_m": float(np.median([x["candidate_translation_dispersion_m"] for x in values])),
            "median_reference_distance_m": float(np.median([x["top1_reference_distance_m"] for x in values])),
            "failure_types": {t: sum((not x["success"]) and t in x["failure_types"] for x in values)
                              for t in ("reference_mismatch", "candidate_conflict", "trajectory_jump",
                                        "turn_motion", "appearance_change", "low_texture_or_blur")},
        })
    return spatial, temporal


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(sequence, rows, output_dir, segment, query_root):
    chosen = rows[segment["start_index"]:segment["end_index"] + 1]
    if len(chosen) > 12:
        indices = np.linspace(0, len(chosen) - 1, 12).astype(int)
        chosen = [chosen[i] for i in indices]
    tiles = []
    for row in chosen:
        image = Image.open(query_root / row["image_path"]).convert("RGB").resize((320, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 320, 42), fill=(0, 0, 0))
        draw.text((4, 3), f"i={row['query_index']} fail={not row['success']}", fill=(255, 255, 0))
        draw.text((4, 21), f"disp={row['candidate_translation_dispersion_m']:.1f} ref={row['top1_reference_distance_m']:.1f}", fill=(255, 255, 0))
        tiles.append(image)
    sheet = Image.new("RGB", (4 * 320, math.ceil(len(tiles) / 4) * 240), (30, 30, 30))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 4) * 320, (index // 4) * 240))
    name = f"{sequence}_failure_{segment['start_index']:04d}_{segment['end_index']:04d}.jpg"
    sheet.save(output_dir / name, quality=92)
    return name


def analyse_sequence(args, sequence):
    result_root = args.results_root / sequence
    rows = read_jsonl(result_root / "frame_diagnostics.jsonl")
    frames, times, query_root, query_pose, query_delta = load_poses(
        args.data_root, sequence, args.gt_dir / f"{sequence}_gt_poses.txt", args.max_gt_delta)
    db_frames, db_times, db_root, db_pose, _ = load_poses(
        args.data_root, "mapping", args.mapping_gt, args.max_gt_delta)
    thresholds_placeholder = None
    for index, (row, frame, pose) in enumerate(zip(rows, frames, query_pose)):
        ref_indices = np.asarray(row["reference_indices"], dtype=np.int64)
        ref_distances = np.linalg.norm(db_pose[ref_indices, :3, 3] - pose[None, :3, 3], axis=1)
        # read_manifest already returns paths rooted at each sequence root.
        # Joining them with the root again silently produced doubled paths.
        ref_image = Path(db_frames[int(ref_indices[0])]["path"])
        image_path = Path(frame["path"])
        image_stats = image_features(image_path)
        previous_diff = image_difference(Path(frames[index - 1]["path"]), image_path) if index else 0.0
        step = float(np.linalg.norm(pose[:3, 3] - query_pose[index - 1, :3, 3])) if index else 0.0
        turn = math.degrees(angle_delta(query_pose[index - 1], pose)) if index else 0.0
        row.update({
            "image_path": relative_frame_path(image_path, query_root),
            "gt_x_m": float(pose[0, 3]), "gt_y_m": float(pose[1, 3]), "gt_z_m": float(pose[2, 3]),
            "gt_delta_s": float(query_delta[index]),
            "top1_reference_distance_m": float(ref_distances[0]),
            "topk_reference_distance_median": float(np.median(ref_distances)),
            "topk_reference_distance_min": float(np.min(ref_distances)),
            "laplacian_var": image_stats["laplacian_var"],
            "edge_density": image_stats["edge_density"],
            "intensity_mean": image_stats["intensity_mean"],
            "intensity_std": image_stats["intensity_std"],
            "appearance_change": float(image_difference(ref_image, image_path)),
            "frame_change": float(previous_diff),
            "step_distance_m": step,
            "turn_motion_deg": turn,
            "success": bool(row["success_0.25m_5deg"]),
        })
    threshold = thresholds(rows)
    for row in rows:
        primary, tags = label_row(row, threshold)
        row["primary_failure_type"] = primary
        row["failure_types"] = tags
    spatial, temporal = spatial_and_temporal(rows)
    failure_runs = json.loads((result_root / "failure_runs.json").read_text())
    # Keep temporal ordering in the labelled file, but use an explicit length
    # ordering for contact sheets and the summary's representative segments.
    longest_runs = sorted(failure_runs, key=lambda x: (-x["frames"], x["start_index"]))
    for run in failure_runs:
        values = rows[run["start_index"]:run["end_index"] + 1]
        run["dominant_failure_type"] = max((x["primary_failure_type"] for x in values),
                                            key=lambda t: sum(x["primary_failure_type"] == t for x in values),
                                            default="none")
        run["mean_candidate_dispersion_m"] = float(np.mean([x["candidate_translation_dispersion_m"] for x in values]))
        run["mean_reference_distance_m"] = float(np.mean([x["top1_reference_distance_m"] for x in values]))
        run["mean_turn_motion_deg"] = float(np.mean([x["turn_motion_deg"] for x in values]))
        run["failure_types"] = {t: sum(t in x["failure_types"] for x in values)
                                 for t in ("reference_mismatch", "candidate_conflict", "trajectory_jump",
                                           "turn_motion", "appearance_change", "low_texture_or_blur")}
    seq_out = args.output_dir / sequence
    seq_out.mkdir(parents=True, exist_ok=True)
    (seq_out / "frame_diagnostics_labelled.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (seq_out / "failure_runs_labelled.json").write_text(json.dumps(failure_runs, indent=2) + "\n", encoding="utf-8")
    (seq_out / "spatial_grid.json").write_text(json.dumps(spatial, indent=2) + "\n", encoding="utf-8")
    (seq_out / "temporal_windows.json").write_text(json.dumps(temporal, indent=2) + "\n", encoding="utf-8")
    (seq_out / "thresholds.json").write_text(json.dumps(threshold, indent=2) + "\n", encoding="utf-8")
    write_csv(seq_out / "spatial_grid.csv", spatial)
    write_csv(seq_out / "temporal_windows.csv", temporal)
    sheet_dir = seq_out / "contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    for segment in longest_runs[:5]:
        segment["contact_sheet"] = contact_sheet(sequence, rows, sheet_dir, segment, query_root)
    summary = {
        "sequence": sequence,
        "frames": len(rows),
        "failure_frames": int(sum(not r["success"] for r in rows)),
        "failure_rate": float(np.mean([not r["success"] for r in rows])),
        "thresholds": threshold,
        "failure_type_counts": {t: int(sum((not r["success"]) and t in r["failure_types"] for r in rows))
                                 for t in ("reference_mismatch", "candidate_conflict", "trajectory_jump",
                                           "turn_motion", "appearance_change", "low_texture_or_blur")},
        "top_failure_segments": longest_runs[:5],
        "spatial_hotspots": sorted(spatial, key=lambda x: (-x["failure_rate"], -x["frames"]))[:10],
        "temporal_windows": temporal,
        "gt_is_posthoc_only": True,
    }
    (seq_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mapping-gt", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gt-delta", type=float, default=0.1)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analyse_sequence(args, sequence) for sequence in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
