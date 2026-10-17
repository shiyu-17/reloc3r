#!/usr/bin/env python3
"""Generate GT-free 7Scenes tool/reference candidates for offline labels.

This executes every cheap-profiled reference-group/tool option independently.
It writes online-safe traces only; run build_poseagent_verifier_dataset.py
separately to join the evaluator-only query_gt sidecar.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.poseagent_style import DEFAULT_TOOLS, PairProfiler
from tools.sevenscenes_poseagent import SCENES, make_backend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prepared-root", type=Path, default=Path("data/7scenes_poseagent"))
parser.add_argument("--scenes", nargs="+", choices=SCENES, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-queries-per-scene", type=int, default=200)
parser.add_argument("--query-start", type=int, default=0)
parser.add_argument("--nfeatures", type=int, default=1200)
parser.add_argument("--max-groups", type=int, default=4)
args = parser.parse_args()
if args.max_queries_per_scene < 1 or args.max_groups < 1:
    parser.error("query and group limits must be positive")
profiler = PairProfiler()
rows = []
for scene in args.scenes:
    backend = make_backend(args.prepared_root, scene, nfeatures=args.nfeatures)
    stop = min(backend.num_frames(scene), args.query_start + args.max_queries_per_scene)
    for frame in range(args.query_start, stop):
        # Fresh environment makes each action value a counterfactual from the
        # same online state.  No query pose is available to this process.
        base = AgenticLocalizationEnv(backend, max_actions_per_frame=3, budget_s=90.0)
        initial = base.reset(scene, start_frame=frame)
        for group_index, group in enumerate(profiler.groups(initial)[:args.max_groups]):
            refs = tuple(int(value) for value in group["reference_indices"])
            for tool in DEFAULT_TOOLS:
                if len(refs) < tool.min_references:
                    continue
                env = AgenticLocalizationEnv(backend, max_actions_per_frame=3, budget_s=90.0)
                observation = env.reset(scene, start_frame=frame)
                profile = profiler.profile(observation, tool, group_index, group)
                observation, _, _, _ = env.step(tool.action(refs))
                if observation.candidate is None or not observation.candidate.get("pose_available", False):
                    rows.append({"schema_version": 1, "dataset": "7scenes", "sequence": scene,
                                 "state": observation.as_dict(), "candidate": observation.candidate,
                                 "candidate_pose": None, "ranker_profile": profile,
                                 "action": {"tool_id": tool.tool_id, "group_index": group_index,
                                            "reference_indices": list(refs)},
                                 "online_audit": env.audit_trace()})
                    continue
                observation, _, _, _ = env.step(LocalizationAction.verify())
                value = observation.as_dict()
                candidate = value.get("candidate")
                pose = None if env.state is None or env.state.current_candidate is None else env.state.current_candidate.pose
                if candidate and pose is not None:
                    rows.append({"schema_version": 1, "dataset": "7scenes", "sequence": scene,
                                 "state": value, "candidate": candidate, "candidate_pose": pose.tolist(),
                                 "ranker_profile": profile,
                                 "action": {"tool_id": tool.tool_id, "group_index": group_index,
                                            "reference_indices": list(refs)},
                                 "online_audit": env.audit_trace()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
print(json.dumps({"candidates": len(rows), "scenes": args.scenes, "output": str(args.output),
                  "boundary": "query GT was not loaded"}, indent=2))
