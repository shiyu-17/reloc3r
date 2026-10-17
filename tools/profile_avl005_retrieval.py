#!/usr/bin/env python3
"""Measure mapping-only NetVLAD retrieval stages for the four LILocBench queries."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from reloc3r.image_retrieval.netvlad import NetVLAD
from tools.eval_lilocbench_reloc3r import read_manifest, retrieval_tensor


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)
    ext = NetVLAD(NetVLAD.default_conf).eval().to(device)
    _, mapping, _, _, _ = read_manifest(args.data_root, "mapping")
    t0 = time.perf_counter()
    dbd = [ext({"image": retrieval_tensor(f["path"], device)})["global_descriptor"].cpu().numpy()[0]
           for f in mapping]
    mapping_descriptor_s = time.perf_counter() - t0
    dbd = np.asarray(dbd, np.float32)
    report = {"protocol": "mapping-only NetVLAD; query GT not loaded", "mapping_frames": len(mapping),
              "mapping_descriptor_s": mapping_descriptor_s, "sequences": {}}
    for seq in SEQUENCES:
        _, query, _, _, _ = read_manifest(args.data_root, seq)
        t0 = time.perf_counter()
        qd = [ext({"image": retrieval_tensor(f["path"], device)})["global_descriptor"].cpu().numpy()[0]
              for f in query]
        query_descriptor_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = np.asarray(qd, np.float32) @ dbd.T
        order = np.argsort(-scores, axis=1)[:, :args.topk]
        ranking_s = time.perf_counter() - t0
        report["sequences"][seq] = {
            "frames": len(query), "query_descriptor_s": query_descriptor_s,
            "ranking_s": ranking_s, "total_s": query_descriptor_s + ranking_s,
            "query_gt_read": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
