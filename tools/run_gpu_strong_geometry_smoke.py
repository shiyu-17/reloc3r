#!/usr/bin/env python3
"""One-frame, online RoMa strong-geometry smoke with a causal receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--sequence", default="static_0")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = LILocBenchReloc3rBackend(
        args.data_root, args.mapping_gt, args.sequence,
        args.retrieval_root / args.sequence / "retrieval.npz", device=args.device)
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=4)
    observation = env.reset(args.sequence, start_frame=args.frame_index)
    refs = observation.frame["reference_groups"][0]["reference_indices"]
    observation, _, _, info = env.step(LocalizationAction.geometry(*refs))
    verification = None
    if observation.candidate and observation.candidate["pose_available"]:
        observation, _, _, verification = env.step(LocalizationAction.verify())
    result = {"sequence": args.sequence, "frame_index": args.frame_index,
              "references": refs, "geometry_event": info["event"],
              "candidate": observation.candidate,
              "verification_event": None if verification is None else verification["event"],
              "audit": env.audit_trace()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
