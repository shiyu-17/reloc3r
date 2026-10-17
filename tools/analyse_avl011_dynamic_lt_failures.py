#!/usr/bin/env python3
"""Post-hoc diagnosis of Dynamic+LT failure modes for AVL-011.

This script is deliberately an evaluator/diagnostic.  It may read GT to
measure reference coverage and success labels, but none of these fields are
used by an online route.  Labels ending in ``_proxy`` are operational
hypotheses, not semantic ground truth for pedestrians or furniture changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

# Allow direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical,
    read_manifest,
    tum_camera_poses,
)


def read_jsonl(path: Path):
    return {int(row["query_index"]): row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def ref_distances(ref_indices, mapping_pose, query_pose):
    refs = mapping_pose[np.asarray(ref_indices, dtype=np.int64)]
    trans = np.linalg.norm(refs[:, :3, 3] - query_pose[None, :3, 3], axis=1)
    rot = np.asarray([rotation_error(ref, query_pose) for ref in refs], dtype=np.float64)
    return trans, rot


def make_contact(rows, query_root: Path, path: Path, title: str, max_tiles=24):
    if not rows:
        return
    rows = rows[:max_tiles]
    tw, th = 300, 220
    tiles = []
    for row in rows:
        try:
            image = Image.open(query_root / row["image_path"]).convert("RGB").resize((tw, th - 45))
        except Exception:
            continue
        tile = Image.new("RGB", (tw, th), "white")
        tile.paste(image, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, th - 42), f"i={row['query_index']} {row['outcome']}", fill="black")
        draw.text((4, th - 27), f"ref={row['ref_label']} mem={int(row['memory_success'])} roma={int(row['roma_success'])}", fill="black")
        draw.text((4, th - 12), f"in={row['roma_inliers']} rep={row['roma_reprojection_px']}", fill="black")
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    sheet = Image.new("RGB", (cols * tw, math.ceil(len(tiles) / cols) * th), "#dddddd")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path, quality=92)


def analyse_sequence(args, sequence: str):
    data_root = args.data_root
    _, query_frames, query_times, _, query_root = read_manifest(data_root, sequence)
    _, mapping_frames, mapping_times, _, mapping_root = read_manifest(data_root, "mapping")
    query_pose, _ = tum_camera_poses(args.gt_dir / f"{sequence}_gt_poses.txt", query_times,
                                      base_to_front_rgb_optical(query_root), args.max_gt_delta)
    mapping_pose, _ = tum_camera_poses(args.mapping_gt, mapping_times,
                                       base_to_front_rgb_optical(mapping_root), args.max_gt_delta)

    diag = read_jsonl(args.diagnostics_root / sequence / "frame_diagnostics_labelled.jsonl")
    memory = read_jsonl(args.memory_root / sequence / "frame_results.jsonl")
    roma = read_jsonl(args.roma_root / f"{sequence}_top100_merged.jsonl")
    roma_eval = {int(row["query_index"]): row for row in
                 json.loads((args.roma_eval_root / f"roma_{sequence}.json").read_text())["frames"]}
    retrieval = np.load(args.retrieval_root / f"{sequence}_top100.npz")
    top100 = retrieval["mapping_indices"]

    rows = []
    for qi in range(len(query_frames)):
        d = diag[qi]
        m = memory[qi]["methods"]
        r = roma[qi]
        re = roma_eval[qi]
        mem_success = bool(m["memory3"]["success_0.25m_5deg"])
        roma_success = bool(re.get("metric") and re["metric"].get("success_0.25m_5deg", False))
        mem_internal = bool(m["memory3"].get("internal_reliable", False))
        ref10_t, ref10_r = ref_distances(d["reference_indices"], mapping_pose, query_pose[qi])
        ref100_t, ref100_r = ref_distances(top100[qi], mapping_pose, query_pose[qi])
        top10_min_t = float(np.min(ref10_t))
        top100_min_t = float(np.min(ref100_t))
        top100_min_r = float(np.min(ref100_r))

        # Spatial retrieval diagnostics.  These thresholds are intentionally
        # broad proxies: metric proximity does not prove visual visibility.
        if top100_min_t > args.no_reference_m or (top100_min_t > args.reference_m and top100_min_r > args.reference_deg):
            ref_label = "reference_unavailable_proxy"
        elif top10_min_t > args.reference_m and top100_min_t <= args.reference_m:
            ref_label = "reference_wrong_selection_proxy"
        else:
            ref_label = "reference_spatially_covered_proxy"

        labels = set(d.get("failure_types", []))
        if d.get("candidate_translation_dispersion_m", 0.0) > args.candidate_translation_m or d.get("candidate_rotation_dispersion_deg", 0.0) > args.candidate_rotation_deg:
            labels.add("candidate_conflict")
        if d.get("frame_change", 0.0) > args.frame_change and d.get("laplacian_var", 1e9) < args.blur_laplacian:
            labels.add("occlusion_or_motion_degradation_proxy")
        if d.get("appearance_change", 0.0) > args.scene_appearance and d.get("frame_change", 0.0) <= args.frame_change:
            labels.add("persistent_scene_change_proxy")
        if r.get("pose") is None:
            labels.add("roma_no_pose")
        elif not roma_success:
            labels.add("roma_pose_available_but_wrong")
        if mem_internal and not mem_success:
            labels.add("memory_available_but_wrong")
        if mem_success:
            labels.add("memory_usable")

        if mem_success and roma_success:
            outcome = "both_success"
        elif mem_success:
            outcome = "memory_only"
        elif roma_success:
            outcome = "roma_only"
        else:
            outcome = "both_fail"

        rows.append({
            "sequence": sequence,
            "query_index": qi,
            "timestamp": float(query_times[qi]),
            "image_path": d["image_path"],
            "outcome": outcome,
            "memory_success": mem_success,
            "memory_internal_reliable": mem_internal,
            "roma_success": roma_success,
            "roma_pose_available": r.get("pose") is not None,
            "roma_inliers": int(r.get("inliers", 0)),
            "roma_reprojection_px": r.get("median_reprojection_px"),
            "ref_label": ref_label,
            "top10_min_reference_distance_m": top10_min_t,
            "top100_min_reference_distance_m": top100_min_t,
            "top100_min_reference_rotation_deg": top100_min_r,
            "candidate_translation_dispersion_m": d.get("candidate_translation_dispersion_m"),
            "candidate_rotation_dispersion_deg": d.get("candidate_rotation_dispersion_deg"),
            "appearance_change": d.get("appearance_change"),
            "frame_change": d.get("frame_change"),
            "laplacian_var": d.get("laplacian_var"),
            "operational_labels": sorted(labels),
            "gt_is_posthoc_only": True,
        })

    out = args.output_dir / sequence
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

    outcomes = Counter(row["outcome"] for row in rows)
    labels = Counter(label for row in rows for label in row["operational_labels"])
    ref_labels = Counter(row["ref_label"] for row in rows)
    cross_ref = defaultdict(Counter)
    cross_operational = defaultdict(Counter)
    for row in rows:
        cross_ref[row["ref_label"]][row["outcome"]] += 1
        for label in row["operational_labels"]:
            cross_operational[label][row["outcome"]] += 1

    representatives = {}
    for outcome in ("both_fail", "memory_only", "roma_only", "both_success"):
        candidates = [row for row in rows if row["outcome"] == outcome]
        representatives[outcome] = [row["query_index"] for row in candidates[:20]]
        make_contact(candidates, query_root, out / f"{outcome}.jpg", outcome)
    both_fail = [row for row in rows if row["outcome"] == "both_fail"]
    for label in ("reference_unavailable_proxy", "reference_wrong_selection_proxy", "occlusion_or_motion_degradation_proxy", "persistent_scene_change_proxy", "roma_pose_available_but_wrong", "memory_available_but_wrong"):
        selected = [row for row in both_fail if label in row["operational_labels"] or row["ref_label"] == label]
        make_contact(selected, query_root, out / f"both_fail_{label}.jpg", label)

    report = {
        "protocol": "AVL-011 post-hoc GT diagnostic; labels are operational proxies, not semantic ground truth",
        "sequence": sequence,
        "frames": len(rows),
        "outcome_counts": dict(outcomes),
        "operational_label_counts": dict(labels),
        "reference_label_counts": dict(ref_labels),
        "outcome_by_reference_label": {key: dict(value) for key, value in sorted(cross_ref.items())},
        "outcome_by_operational_label": {key: dict(value) for key, value in sorted(cross_operational.items())},
        "both_fail_rows": [row["query_index"] for row in both_fail],
        "representatives": representatives,
        "thresholds": {
            "reference_m": args.reference_m,
            "reference_deg": args.reference_deg,
            "no_reference_m": args.no_reference_m,
            "candidate_translation_m": args.candidate_translation_m,
            "candidate_rotation_deg": args.candidate_rotation_deg,
            "frame_change": args.frame_change,
            "blur_laplacian": args.blur_laplacian,
            "scene_appearance": args.scene_appearance,
        },
        "gt_is_posthoc_only": True,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--mapping-gt", type=Path, required=True)
    p.add_argument("--diagnostics-root", type=Path, required=True)
    p.add_argument("--memory-root", type=Path, required=True)
    p.add_argument("--roma-root", type=Path, required=True)
    p.add_argument("--roma-eval-root", type=Path, required=True)
    p.add_argument("--retrieval-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", default=["lt_changes_dynamics_0"])
    p.add_argument("--max-gt-delta", type=float, default=0.1)
    p.add_argument("--reference-m", type=float, default=2.0)
    p.add_argument("--reference-deg", type=float, default=45.0)
    p.add_argument("--no-reference-m", type=float, default=4.0)
    p.add_argument("--candidate-translation-m", type=float, default=1.5)
    p.add_argument("--candidate-rotation-deg", type=float, default=15.0)
    p.add_argument("--frame-change", type=float, default=0.15)
    p.add_argument("--blur-laplacian", type=float, default=450.0)
    p.add_argument("--scene-appearance", type=float, default=0.20)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [analyse_sequence(args, sequence) for sequence in args.sequences]
    (args.output_dir / "summary_all.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
