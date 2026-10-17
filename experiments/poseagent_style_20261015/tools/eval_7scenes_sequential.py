#!/usr/bin/env python3
"""Run the existing causal sequential policies and finite-depth Oracle on 7Scenes.

The two test rooms are never used for model selection.  Query poses are loaded
only by this offline evaluator after the online backend/policy trace exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.eval_active_evidence_sequential import (
    DEFAULT_UTILITY, run_oracle, run_policy, summarize,
)
from tools.sevenscenes_poseagent import SCENES


def args_for(root: Path, output: Path, *, max_acquisitions: int, budget_ms: float,
             oracle_depth: int, nfeatures: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        backend="7scenes_cpu_orb", prepared_root=root, nfeatures=nfeatures,
        max_acquisitions=max_acquisitions, budget_ms=budget_ms, oracle_depth=oracle_depth,
        seed=seed, geometry_schedule_rate=0.5, uncertainty_threshold=0.8,
        router_threshold=0.45, utility=dict(DEFAULT_UTILITY), output_dir=output,
        retrieval_root=Path("unused"), data_root=Path("unused"), mapping_gt=Path("unused"),
        device="cpu", verifier_config=None, stage3d_preaction_features=False,
    )


def episode_summary(records, gt, chunk: int) -> dict:
    episodes = []
    for start in range(0, len(records), chunk):
        part = records[start:start + chunk]
        successes = [success_pose_safe(row, gt) for row in part]
        accepted = [x for x in part if x.get("accepted")]
        episodes.append({"start": start, "end": start + len(part) - 1,
                         "episode_success": bool(any(successes)),
                         "frames": len(part), "accepted": len(accepted),
                         "false_accepts": int(sum(bool(x.get("accepted")) and not ok
                                                  for x, ok in zip(part, successes)))})
    return {"episodes": len(episodes), "episode_successes": sum(x["episode_success"] for x in episodes),
            "episode_success_rate": (sum(x["episode_success"] for x in episodes) / len(episodes)
                                     if episodes else 0.0),
            "false_accepts": sum(x["false_accepts"] for x in episodes),
            "episodes_detail": episodes}


def success_pose_safe(row, gt):
    pose = row.get("pose")
    if pose is None:
        return False
    from tools.eval_active_evidence_policies import success_pose
    return bool(success_pose(pose, gt[int(row["frame_index"])]))


def dependency(oracle_rows) -> dict:
    states = dependent = 0
    for row in oracle_rows:
        roots = row.get("oracle_action_values", [])
        values = {int(node.get("action_value", {}).get("best_terminal_success", 0)) for node in roots}
        if roots:
            states += 1
            dependent += int(len(values) > 1)
    return {"states": states, "action_dependent_states": dependent,
            "action_dependent_fraction": dependent / states if states else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/7scenes_poseagent"))
    parser.add_argument("--scenes", nargs="+", choices=("redkitchen", "stairs"),
                        default=["redkitchen", "stairs"])
    parser.add_argument("--max-frames", type=int, default=25,
                        help="per-room query frames; use 0 for every prepared query")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--oracle-depth", type=int, default=1,
                        help="depth 1 is the exhaustive all-action baseline for CPU smoke; depth 2 is paper gate")
    parser.add_argument("--budget-ms", type=float, default=30000.0)
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_frames < 0 or args.oracle_depth < 1:
        parser.error("max-frames must be nonnegative and oracle-depth positive")
    policies = ("fixed", "fixed_geometry", "geometry_greedy", "feedback_rule")
    output = {"schema_version": 1, "dataset": "7scenes", "frozen_test_scenes": args.scenes,
              "policy_summaries": {}, "oracle_summaries": {}, "gate": {}}
    for scene in args.scenes:
        gt = np.load(args.prepared_root / scene / "query_gt.npy")
        backend_args = args_for(args.prepared_root, args.output.parent,
                                max_acquisitions=args.max_acquisitions,
                                budget_ms=args.budget_ms, oracle_depth=args.oracle_depth,
                                nfeatures=args.nfeatures, seed=777)
        total = len(gt)
        end = total if args.max_frames == 0 else min(total, args.start + args.max_frames)
        scene_rows = {}
        for policy in policies:
            result = run_policy(backend_args, scene, policy, args.start, end)
            scene_rows[policy] = {"summary": summarize(result["records"], gt),
                                  "records": result["records"], "audit": result["audit"]}
        oracle = run_oracle(backend_args, scene, args.start, end, gt)
        oracle_summary = summarize(oracle["records"], gt)
        scene_rows["oracle"] = {"summary": oracle_summary, "records": oracle["records"],
                                "audit": oracle["audit"], "dependency": dependency(oracle["records"])}
        best_policy = max(policies, key=lambda name: scene_rows[name]["summary"]["full_frame_success"])
        base = episode_summary(scene_rows[best_policy]["records"], gt, max(1, min(10, end - args.start)))
        oracle_ep = episode_summary(oracle["records"], gt, max(1, min(10, end - args.start)))
        scene_rows["gate"] = {
            "best_fixed_policy": best_policy,
            "best_fixed_episode_success_rate": base["episode_success_rate"],
            "oracle_episode_success_rate": oracle_ep["episode_success_rate"],
            "oracle_gap_pp": 100.0 * (oracle_ep["episode_success_rate"] - base["episode_success_rate"]),
            "action_dependency": scene_rows["oracle"]["dependency"],
            "status": "candidate_gate_only",
        }
        output["policy_summaries"][scene] = {name: value["summary"] for name, value in scene_rows.items()
                                              if name in policies}
        output["oracle_summaries"][scene] = {"summary": oracle_summary,
                                              "dependency": scene_rows["oracle"]["dependency"],
                                              "gate": scene_rows["gate"],
                                              "audit": oracle["audit"]}
        output.setdefault("traces", {})[scene] = scene_rows
    gaps = [output["oracle_summaries"][scene]["gate"]["oracle_gap_pp"] for scene in args.scenes]
    deps = [output["oracle_summaries"][scene]["dependency"]["action_dependent_fraction"] for scene in args.scenes]
    output["gate"] = {
        "minimum_oracle_success_gain_pp": 5.0,
        "minimum_action_dependent_state_fraction": 0.10,
        "oracle_gap_pp_by_scene": dict(zip(args.scenes, gaps)),
        "action_dependency_by_scene": dict(zip(args.scenes, deps)),
        "oracle_gap_pass": bool(gaps and min(gaps) >= 5.0),
        "action_dependency_pass": bool(deps and min(deps) >= 0.10),
        "learned_agent_go": False,
        "reason": "candidate_gate_only; frozen test run does not authorize learned-agent claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["gate"], indent=2))


if __name__ == "__main__":
    main()
