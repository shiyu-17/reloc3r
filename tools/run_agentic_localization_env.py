#!/usr/bin/env python3
"""Run the first active-evidence baseline on the causal environment.

This is deliberately a small rule policy, not an Agent claim.  It proves that
choosing a different reference batch causes new Reloc3r inference and that
only verifier-approved poses are carried into the next frame's Memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction


def run(args):
    if args.backend == "reloc3r":
        from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend
        backend = LILocBenchReloc3rBackend(
            data_root=args.data_root, mapping_gt=args.mapping_gt,
            query_sequence=args.sequence, retrieval_npz=args.retrieval,
            device=args.device, min_references=args.min_references,
        )
    else:
        from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend
        backend = CpuOrbLocalizationBackend(
            data_root=args.data_root, mapping_gt=args.mapping_gt,
            query_sequence=args.sequence, retrieval_npz=args.retrieval,
            nfeatures=args.nfeatures,
        )
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=args.max_actions_per_frame)
    observation = env.reset(args.sequence)
    trace = []
    while (not observation.done and
           (args.max_frames is None or observation.frame_index < args.max_frames)):
        refs = observation.frame.get("retrieval_references", [])
        # The policy may choose a different reference subset on the same
        # frame.  Each subset is passed to the live backend and rerun.
        batches = [refs[start:start + args.references_per_action]
                   for start in range(0, len(refs), args.references_per_action)]
        batches = [batch for batch in batches if batch]
        finished = False
        for batch in batches[:args.max_reference_attempts]:
            action = LocalizationAction.localize(*batch)
            observation, reward, done, info = env.step(action)
            trace.append({"action": action.kind, "references": batch,
                          "frame_index": observation.frame_index, "reward": reward,
                          "info": info})
            if done:
                finished = True
                break
            if observation.candidate and observation.candidate.get("pose_available"):
                observation, reward, done, info = env.step(LocalizationAction.verify())
                trace.append({"action": "verify_candidate", "frame_index": observation.frame_index,
                              "reward": reward, "info": info})
                if observation.candidate and observation.candidate.get("verified"):
                    observation, reward, done, info = env.step(LocalizationAction.accept())
                    trace.append({"action": "accept_candidate", "frame_index": observation.frame_index,
                                  "reward": reward, "info": info})
                    finished = True
                    break
        if finished:
            continue
        # Memory retrieval is causal and can only see accepted records from
        # earlier frames.  If it cannot be verified, defer without writing it.
        observation, reward, done, info = env.step(LocalizationAction.retrieve())
        trace.append({"action": "retrieve_memory", "frame_index": observation.frame_index,
                      "reward": reward, "info": info})
        if observation.candidate and observation.candidate.get("pose_available"):
            observation, reward, done, info = env.step(LocalizationAction.verify())
            trace.append({"action": "verify_memory", "frame_index": observation.frame_index,
                          "reward": reward, "info": info})
            if observation.candidate and observation.candidate.get("verified"):
                observation, reward, done, info = env.step(LocalizationAction.accept())
                trace.append({"action": "accept_memory", "frame_index": observation.frame_index,
                              "reward": reward, "info": info})
                continue
        observation, reward, done, info = env.step(LocalizationAction.defer())
        trace.append({"action": "defer_frame", "frame_index": observation.frame_index,
                      "reward": reward, "info": info})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace))
    (args.output.parent / "environment_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in env.trace))
    environment_audit = env.audit_trace()
    if not environment_audit["passed"]:
        raise RuntimeError("causal environment audit failed: " + repr(environment_audit))
    summary = {
        "sequence": args.sequence,
        "transitions": len(trace),
        "accepted_memory_records": len(env.state.memory),
        "final_frame_index": env.state.frame_index,
        "done": env.state.done,
        "query_gt_read": False,
        "uses_future": False,
        "selection_uses_gt": False,
        "policy": "reference_batch_active_baseline_with_causal_memory",
        "backend": args.backend,
        "partial_run": not observation.done,
        "environment_audit": environment_audit,
    }
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping-gt", type=Path, required=True,
                        help="map asset poses; query GT is never read")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("reloc3r", "cpu_orb"), default="reloc3r")
    parser.add_argument("--nfeatures", type=int, default=800,
                        help="CPU ORB feature count; ignored by Reloc3r backend")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="optional partial-run limit for smoke tests")
    parser.add_argument("--references-per-action", type=int, default=3)
    parser.add_argument("--max-reference-attempts", type=int, default=3)
    parser.add_argument("--min-references", type=int, default=2)
    # Three reference attempts use six transitions; memory query/verify and a
    # final defer can use three more.
    parser.add_argument("--max-actions-per-frame", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
