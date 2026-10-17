#!/usr/bin/env python3
"""Pair-level RoMa indoor smoke test for AVL-004.

The script writes only dense 2D correspondence evidence. It never reads GT or
produces a localization score; scoring is a later, separate evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from romatch import roma_indoor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-matches", type=int, default=10000)
    args = parser.parse_args()
    device = torch.device(args.device)
    # The optional local_corr CUDA extension is absent in the reloc3r env.
    # Native PyTorch correlation keeps the official weights/model definition
    # unchanged and is slower but auditable.
    model = roma_indoor(device=device, coarse_res=560, upsample_res=864,
                        use_custom_corr=False)
    model.eval()
    warp, certainty = model.match(str(args.query), str(args.reference), device=device)
    matches, match_certainty = model.sample(warp, certainty, num=args.num_matches)
    query_image = cv2.imread(str(args.query), cv2.IMREAD_GRAYSCALE)
    ref_image = cv2.imread(str(args.reference), cv2.IMREAD_GRAYSCALE)
    if query_image is None or ref_image is None:
        raise FileNotFoundError("Unable to read smoke-test image")
    hq, wq = query_image.shape[:2]
    hr, wr = ref_image.shape[:2]
    q_points, r_points = model.to_pixel_coordinates(matches, hq, wq, hr, wr)
    q_points = q_points.detach().cpu().numpy().astype(np.float32)
    r_points = r_points.detach().cpu().numpy().astype(np.float32)
    confidence = match_certainty.detach().cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), query_points=q_points,
                        reference_points=r_points, confidence=confidence)
    summary = {
        "query": str(args.query), "reference": str(args.reference),
        "matches": int(len(q_points)),
        "confidence_median": float(np.median(confidence)) if len(confidence) else 0.0,
        "confidence_p90": float(np.percentile(confidence, 90)) if len(confidence) else 0.0,
        "device": str(device), "gt_read": False,
        "model": "romatch==0.1.2 / roma_indoor",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
