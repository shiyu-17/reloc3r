#!/usr/bin/env python3
"""Prepare official 7Scenes train-map/test-query manifests and retrieval."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.sevenscenes_poseagent import SCENES, prepare_scene

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("data/7scenes"))
parser.add_argument("--output-root", type=Path, default=Path("data/7scenes_poseagent"))
parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
parser.add_argument("--map-stride", type=int, default=10)
parser.add_argument("--query-stride", type=int, default=10)
parser.add_argument("--topk", type=int, default=12)
args = parser.parse_args()
outputs = [prepare_scene(args.data_root, args.output_root, scene, map_stride=args.map_stride,
                         query_stride=args.query_stride, topk=args.topk) for scene in args.scenes]
print(json.dumps({"prepared": [str(path) for path in outputs], "query_gt_boundary": "evaluator sidecar only"}, indent=2))
