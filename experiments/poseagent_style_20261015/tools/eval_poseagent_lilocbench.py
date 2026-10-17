#!/usr/bin/env python3
"""Paired LILocBench comparison: frozen feedback rule vs PoseAgent-style baseline.

This is intentionally an engineering/module ablation only.  The PoseAgent
controller receives no query GT and uses the frozen backend verifier; query GT
is joined only after both online traces have been completed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.eval_active_evidence_policies import SEQUENCES, load_query_gt
from tools.eval_active_evidence_sequential import run_policy, summarize
from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend
from tools.poseagent_style import PairProfiler, PoseAgentFallbackController, RuleRanker, RuleVerifier


def backend(args: argparse.Namespace, sequence: str):
    return LILocBenchReloc3rBackend(
        args.data_root, args.mapping_gt, sequence, args.retrieval_root / sequence / "retrieval.npz",
        device=args.device, verifier_config=args.verifier_config, preaction_features=args.preaction_features)


def sequential_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend="reloc3r", data_root=args.data_root, gt_root=args.gt_root, mapping_gt=args.mapping_gt,
        retrieval_root=args.retrieval_root, device=args.device, verifier_config=args.verifier_config,
        stage3d_preaction_features=args.preaction_features, nfeatures=500,
        max_acquisitions=args.max_estimator_calls, budget_ms=args.budget_ms, seed=args.seed,
        geometry_schedule_rate=args.geometry_schedule_rate, uncertainty_threshold=args.uncertainty_threshold,
        router_threshold=args.router_threshold,
    )


def run_poseagent(args: argparse.Namespace, sequence: str, start: int, end: int) -> dict[str, Any]:
    env = AgenticLocalizationEnv(backend(args, sequence),
                                 max_actions_per_frame=2 * args.max_estimator_calls + 2,
                                 budget_s=args.budget_ms / 1000.0)
    observation = env.reset(sequence, start_frame=start)
    controller = PoseAgentFallbackController(profiler=PairProfiler(), ranker=RuleRanker(),
                                             verifier=RuleVerifier(),
                                             max_estimator_calls=args.max_estimator_calls)
    if args.geometry_after_verifier_reject:
        controller.enable_geometry_after_verifier_reject()
    records = []
    while not observation.done and observation.frame_index < end:
        frame = observation.frame_index
        actions, cost, accepted_pose = [], 0.0, None
        frame_started = time.perf_counter()
        while observation.frame_index == frame and not observation.done:
            decision = controller.choose(observation)
            if decision.action.kind == "accept_candidate":
                assert env.state is not None and env.state.current_candidate is not None
                accepted_pose = env.state.current_candidate.pose.copy()
            observation, _, _, info = env.step(decision.action)
            event = info["event"]
            elapsed = event.get("execution", {}).get("elapsed_s", event.get("evidence", {}).get("elapsed_s", 0.0))
            cost += max(0.0, float(elapsed or 0.0))
            actions.append({"kind": decision.action.kind,
                            "references": list(decision.action.reference_indices),
                            "controller": dict(decision.metadata), "event": event})
            if decision.action.kind in {"accept_candidate", "defer_frame"}:
                break
        records.append({"sequence": sequence, "frame_index": frame, "policy": "poseagent_rule",
                        "accepted": accepted_pose is not None,
                        "pose": None if accepted_pose is None else accepted_pose.tolist(),
                        "acquisitions": [item for item in actions if item["kind"] in {
                            "localize_with_references", "run_strong_geometry", "retrieve_memory"}],
                        "tool_actions": sum(item["kind"] in {"localize_with_references", "run_strong_geometry", "retrieve_memory"}
                                            for item in actions),
                        "terminal_action": actions[-1]["kind"] if actions else "defer_frame",
                        "decisions": actions, "measured_tool_cost_s": cost,
                        "budget_s": args.budget_ms / 1000.0,
                        "budget_overshoot": cost > args.budget_ms / 1000.0,
                        "wall_time_s": time.perf_counter() - frame_started})
    audit = env.audit_trace()
    if not audit["passed"]:
        raise RuntimeError(f"causal audit failed for {sequence}: {audit}")
    return {"records": records, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--preaction-features", action="store_true")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-estimator-calls", type=int, default=2)
    parser.add_argument("--geometry-after-verifier-reject", action="store_true",
                        help="registered no-training parity ablation; do Geometry after a rejected cheap pose")
    parser.add_argument("--budget-ms", type=float, default=30000.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--geometry-schedule-rate", type=float, default=0.35)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.8)
    parser.add_argument("--router-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output: dict[str, Any] = {"schema_version": 1, "purpose": "paired LILocBench PoseAgent-style ablation",
                               "config": {key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()}, "runs": {}}
    legacy_args = sequential_args(args)
    for sequence in args.sequences:
        gt = load_query_gt(args.data_root, args.gt_root, sequence)
        end = min(len(gt), args.start + args.max_frames)
        poseagent = run_poseagent(args, sequence, args.start, end)
        # Fresh backend/environment under the same frozen policy contract.
        feedback = run_policy(legacy_args, sequence, "feedback_rule", args.start, end)
        output["runs"][sequence] = {
            "poseagent_rule": {"summary": summarize(poseagent["records"], gt), "audit": poseagent["audit"]},
            "feedback_rule": {"summary": summarize(feedback["records"], gt), "audit": feedback["audit"]},
        }
        args.output_dir.joinpath("traces").mkdir(parents=True, exist_ok=True)
        for name, run in (("poseagent_rule", poseagent), ("feedback_rule", feedback)):
            args.output_dir.joinpath("traces", f"{sequence}_{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in run["records"]), encoding="utf-8")
    def total(name: str) -> dict[str, float]:
        rows = [value[name]["summary"] for value in output["runs"].values()]
        frames = sum(int(row["frames"]) for row in rows)
        correct = sum(int(row["accepted_correct"]) for row in rows)
        accepted = sum(int(row["accepted"]) for row in rows)
        false = sum(int(row["false_accepts"]) for row in rows)
        return {"frames": frames, "accepted_correct": correct, "success": correct / frames if frames else 0.0,
                "far": false / accepted if accepted else 0.0,
                "mean_cost_s": float(np.mean([row["mean_measured_tool_cost_s"] for row in rows])) if rows else 0.0}
    output["pooled"] = {"poseagent_rule": total("poseagent_rule"), "feedback_rule": total("feedback_rule")}
    output["comparison"] = {"success_delta_pp": 100.0 * (output["pooled"]["poseagent_rule"]["success"] - output["pooled"]["feedback_rule"]["success"]),
                            "far_delta": output["pooled"]["poseagent_rule"]["far"] - output["pooled"]["feedback_rule"]["far"],
                            "cost_delta_s": output["pooled"]["poseagent_rule"]["mean_cost_s"] - output["pooled"]["feedback_rule"]["mean_cost_s"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": output["pooled"], "comparison": output["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
