#!/usr/bin/env python3
"""Audit the 97 A.5 HLoc/RoMa common failures.

The labels in this artifact are operational evidence, not semantic ground
truth.  Query GT is used only to identify the common-failure set and to attach
post-hoc spatial/error fields.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def jsonl(path):
    return {int(x["query_index"]): x for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def median(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else None


def contact_sheet(rows, out, data_root, cols=5, thumb=(240, 180)):
    tiles = []
    for row in rows:
        path = data_root / row["sequence"] / row["image_path"]
        try:
            im = Image.open(path).convert("RGB").resize(thumb)
        except Exception:
            continue
        canvas = Image.new("RGB", (thumb[0], thumb[1] + 34), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, thumb[1] + 3), f'{row["sequence"]}:{row["query_index"]}', fill="black")
        tiles.append(canvas)
    if not tiles:
        return
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb[0], rows_n * (thumb[1] + 34)), "#dddddd")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * thumb[0], (i // cols) * (thumb[1] + 34)))
    sheet.save(out, quality=92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a5-route-dir", type=Path, required=True)
    p.add_argument("--hloc-dir", type=Path, required=True)
    p.add_argument("--roma-dir", type=Path, required=True)
    p.add_argument("--diagnostics-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = {}
    for seq in SEQUENCES:
        hroute = {int(x["query_index"]): x for x in json.loads((args.a5_route_dir / f"hloc_{seq}.json").read_text())["frames"]}
        rroute = {int(x["query_index"]): x for x in json.loads((args.a5_route_dir / f"roma_{seq}.json").read_text())["frames"]}
        hrec, rrec = jsonl(args.hloc_dir / f"{seq}_top100_merged.jsonl"), jsonl(args.roma_dir / f"{seq}_top100_merged.jsonl")
        diag = jsonl(args.diagnostics_root / seq / "frame_diagnostics_labelled.jsonl")
        rows = []
        for qi in sorted(set(hroute) & set(rroute)):
            if hroute[qi].get("action") != "call_geometry" or rroute[qi].get("action") != "call_geometry":
                continue
            hs = bool(hroute[qi].get("metric") and hroute[qi]["metric"]["success_0.25m_5deg"])
            rs = bool(rroute[qi].get("metric") and rroute[qi]["metric"]["success_0.25m_5deg"])
            if hs or rs:
                continue
            d = diag[qi]
            labels = list(d.get("failure_types", []))
            operational = set(labels)
            if d.get("topk_reference_distance_median", 0) > 5.0:
                operational.add("large_reference_distance_proxy")
            if d.get("appearance_change", 0) > 0.15 or d.get("frame_change", 0) > 0.15:
                operational.add("appearance_change_proxy")
            if d.get("laplacian_var", 1e9) < 450:
                operational.add("low_texture_or_blur_proxy")
            hpose, rpose = hrec[qi].get("pose"), rrec[qi].get("pose")
            if hpose is not None and rpose is not None:
                operational.add("both_pose_available_but_wrong")
            row = {"sequence": seq, "query_index": qi, "image_path": d["image_path"], "operational_labels": sorted(operational), "primary_failure_type": d.get("primary_failure_type"), "candidate_translation_dispersion_m": d.get("candidate_translation_dispersion_m"), "candidate_rotation_dispersion_deg": d.get("candidate_rotation_dispersion_deg"), "appearance_change": d.get("appearance_change"), "frame_change": d.get("frame_change"), "laplacian_var": d.get("laplacian_var"), "topk_reference_distance_median": d.get("topk_reference_distance_median"), "top1_reference_distance_m": d.get("top1_reference_distance_m"), "turn_motion_deg": d.get("turn_motion_deg"), "hloc_pose_available": hpose is not None, "roma_pose_available": rpose is not None, "hloc_inliers": hrec[qi].get("inliers"), "roma_inliers": rrec[qi].get("inliers"), "hloc_reprojection_px": hrec[qi].get("median_reprojection_px"), "roma_reprojection_px": rrec[qi].get("median_reprojection_px"), "gt_x_m": d.get("gt_x_m"), "gt_y_m": d.get("gt_y_m"), "gt_z_m": d.get("gt_z_m")}
            rows.append(row); all_rows.append(row)
        counts = Counter(label for row in rows for label in row["operational_labels"])
        summaries[seq] = {"frames": len(rows), "operational_label_counts": dict(counts), "median_features": {k: median(rows, k) for k in ("candidate_translation_dispersion_m", "candidate_rotation_dispersion_deg", "appearance_change", "frame_change", "laplacian_var", "topk_reference_distance_median", "turn_motion_deg", "hloc_inliers", "roma_inliers", "hloc_reprojection_px", "roma_reprojection_px")}}
        contact_sheet(rows, out / f"{seq}_common_failures.jpg", args.data_root)
    (out / "common_failure_rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_rows))
    report = {"protocol": "A.8 post-hoc audit; operational labels are not semantic ground truth", "total_frames": len(all_rows), "sequences": summaries, "global_operational_label_counts": dict(Counter(label for row in all_rows for label in row["operational_labels"]))}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
