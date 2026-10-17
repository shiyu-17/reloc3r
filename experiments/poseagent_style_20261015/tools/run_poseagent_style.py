#!/usr/bin/env python3
"""Run the lightweight PoseAgent-style skeleton on a causal backend.

This is an engineering runner, not a scientific evaluator: it never loads
query GT.  Use the independent evaluator and scene-disjoint protocol for
metrics.  It is intentionally separate from the legacy sequential runner so
existing Stage 3 results remain immutable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv
from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend
from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend
from tools.sevenscenes_poseagent import make_backend as make_7scenes_backend
from tools.poseagent_style import (PairProfiler, PoseAgentFallbackController,
                                   RuleRanker, RuleVerifier, StructuredVerifier,
                                   StructuredRanker, XGBRankerAdapter)


def make_backend(args: argparse.Namespace):
    if args.backend == "7scenes_cpu_orb":
        return make_7scenes_backend(args.prepared_root, args.sequence, nfeatures=args.nfeatures)
    retrieval = args.retrieval_root / args.sequence / "retrieval.npz"
    if args.backend == "cpu_orb":
        return CpuOrbLocalizationBackend(args.data_root, args.mapping_gt, args.sequence,
                                         retrieval, nfeatures=args.nfeatures)
    return LILocBenchReloc3rBackend(
        args.data_root, args.mapping_gt, args.sequence, retrieval,
        device=args.device, verifier_config=args.verifier_config,
        preaction_features=args.preaction_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--mapping-gt", type=Path, default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--backend", choices=("cpu_orb", "reloc3r", "7scenes_cpu_orb"), default="cpu_orb")
    parser.add_argument("--prepared-root", type=Path, default=Path("data/7scenes_poseagent"),
                        help="used only by --backend 7scenes_cpu_orb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfeatures", type=int, default=500)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--preaction-features", action="store_true")
    parser.add_argument("--ranker-model", type=Path,
                        help="optional XGBRanker model; omitted uses the transparent rule ranker")
    parser.add_argument("--structured-verifier", type=Path,
                        help="optional structured verifier directory; omitted uses frozen RuleVerifier")
    parser.add_argument("--max-estimator-calls", type=int, default=2)
    parser.add_argument("--budget-ms", type=float, default=30000.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_frames < 1 or args.max_estimator_calls < 1:
        parser.error("--max-frames and --max-estimator-calls must be positive")

    profiler = PairProfiler()
    if args.ranker_model:
        metadata = json.loads(args.ranker_model.joinpath("metadata.json").read_text(encoding="utf-8"))
        ranker = (StructuredRanker.load(args.ranker_model, profiler)
                  if metadata.get("family") == "poseagent_structured_ranker"
                  else XGBRankerAdapter.load(args.ranker_model, profiler))
    else:
        ranker = RuleRanker()
    verifier = StructuredVerifier.load(args.structured_verifier) if args.structured_verifier else RuleVerifier()
    backend = make_backend(args)
    env = AgenticLocalizationEnv(
        backend, max_actions_per_frame=2 * args.max_estimator_calls + 2,
        budget_s=args.budget_ms / 1000.0,
    )
    observation = env.reset(args.sequence, start_frame=args.start)
    controller = PoseAgentFallbackController(profiler=profiler, ranker=ranker, verifier=verifier,
                                             max_estimator_calls=args.max_estimator_calls)
    records: list[dict[str, Any]] = []
    end = min(args.start + args.max_frames, env.state.frame_count)
    while not observation.done and observation.frame_index < end:
        decision = controller.choose(observation)
        before = observation.as_dict()
        observation, _, _, info = env.step(decision.action)
        records.append({
            "sequence": before["sequence"], "frame_index": before["frame_index"],
            "state": before, "action": {"kind": decision.action.kind,
                                             "reference_indices": list(decision.action.reference_indices)},
            "controller": dict(decision.metadata), "event": info["event"],
            "causal": info["causal"],
        })
    audit = env.audit_trace()
    if not audit["passed"]:
        raise RuntimeError(f"causal audit failed: {audit}")
    output = {
        "schema_version": 1,
        "purpose": "PoseAgent-style skeleton engineering trace; no query GT loaded",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "records": records,
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
