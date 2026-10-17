#!/usr/bin/env python3
"""Create evaluator-only labels for structured PoseAgent verifier training.

The input is an Oracle action-tree trace.  Online traces remain GT-free; this
script joins each executed candidate with query GT *offline* and emits a
separate labelled JSONL dataset.  The output must never be passed wholesale to
an online policy, only its trained artifact may be loaded by the verifier.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_active_evidence_policies import load_query_gt, success_pose
from tools.eval_lilocbench_reloc3r import pose_error
from tools.poseagent_style import StructuredVerifierFeatures


def expand(patterns: Iterable[str]) -> list[Path]:
    paths = sorted({Path(item) for pattern in patterns for item in glob.glob(pattern)})
    if not paths:
        raise ValueError("no trace paths matched")
    return paths


def scene_for(sequence: str, catalog: Mapping[str, Any]) -> str:
    sequences = catalog.get("sequences", catalog)
    metadata = sequences.get(sequence, {})
    return str(metadata.get("scene_id", sequence))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="Oracle JSONL paths or globs")
    parser.add_argument("--scene-catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--seven-scenes-prepared-root", type=Path,
                        help="7Scenes prepared root; reads evaluator-only <scene>/query_gt.npy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.scene_catalog.read_text(encoding="utf-8"))
    labels: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for path in expand(args.traces):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace = json.loads(line)
            # The legacy oracle stores many candidates under one frame trace;
            # the 7Scenes generator writes one online-safe candidate per line.
            nodes = trace.get("oracle_action_values", [trace])
            for node in nodes:
                sequence = str(node.get("sequence", trace.get("sequence")))
                if sequence not in labels:
                    if args.seven_scenes_prepared_root:
                        labels[sequence] = np.load(args.seven_scenes_prepared_root / sequence / "query_gt.npy")
                    else:
                        labels[sequence] = load_query_gt(args.data_root, args.gt_root, sequence)
                candidate = node.get("candidate") or {}
                pose = node.get("candidate_pose")
                observation = node.get("state") or {}
                if not observation:
                    continue
                frame = int(node["state"]["frame_index"])
                if not 0 <= frame < len(labels[sequence]):
                    raise ValueError(f"GT frame index outside sequence: {sequence}/{frame}")
                # A tool failure is useful negative supervision for the
                # ranker, but has no pose/evidence vector for the verifier.
                if pose is None or not candidate:
                    if node.get("ranker_profile"):
                        rows.append({"schema_version": 1, "source_trace": str(path),
                                     "scene_id": scene_for(sequence, catalog), "sequence": sequence,
                                     "frame_index": frame, "node_id": node.get("node_id"),
                                     "observation": observation, "candidate": None,
                                     "ranker_profile": node.get("ranker_profile"), "action": node.get("action"),
                                     "label": {"success": False, "translation_error_m": float("inf"),
                                               "rotation_error_deg": float("inf")}, "offline_only": True})
                    continue
                translation, rotation = pose_error(np.asarray(pose, dtype=np.float64), labels[sequence][frame])
                # This validates the fixed online schema before serializing
                # labels; feature values are not duplicated to avoid a second
                # source of truth for online preprocessing.
                StructuredVerifierFeatures.vector(observation, candidate)
                rows.append({
                    "schema_version": 1,
                    "source_trace": str(path),
                    "scene_id": scene_for(sequence, catalog),
                    "sequence": sequence,
                    "frame_index": frame,
                    "node_id": node.get("node_id"),
                    "observation": observation,
                    "candidate": candidate,
                    "ranker_profile": node.get("ranker_profile"),
                    "action": node.get("action"),
                    "label": {
                        "success": success_pose(pose, labels[sequence][frame]),
                        "translation_error_m": float(translation),
                        "rotation_error_deg": float(rotation),
                    },
                    "offline_only": True,
                })
    if not rows:
        raise ValueError("no pose-producing Oracle candidates found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "scenes": sorted({row["scene_id"] for row in rows}),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
