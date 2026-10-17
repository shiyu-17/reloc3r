#!/usr/bin/env python3
"""Persist a one-frame real-image reference-switch causal smoke trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.agentic_localization_env import AgenticLocalizationEnv, LocalizationAction
from tools.cpu_orb_agent_backend import CpuOrbLocalizationBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping-gt", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="static_0")
    parser.add_argument("--nfeatures", type=int, default=800)
    args = parser.parse_args()

    backend = CpuOrbLocalizationBackend(
        args.data_root, args.mapping_gt, args.sequence, args.retrieval,
        nfeatures=args.nfeatures)
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=6)
    observation = env.reset(args.sequence)
    refs = observation.frame["retrieval_references"]
    first = tuple(refs[:3])
    second = tuple(refs[3:6])
    if len(first) < 2 or len(second) < 2:
        raise RuntimeError("retrieval file does not provide two reference groups")
    env.step(LocalizationAction.localize(*first))
    env.step(LocalizationAction.localize(*second))
    observation, _, _, _ = env.step(LocalizationAction.verify())
    verified = bool(observation.candidate and observation.candidate.get("verified"))
    if verified:
        env.step(LocalizationAction.accept())
    audit = env.audit_trace()
    if not audit["passed"]:
        raise RuntimeError(repr(audit))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in env.trace))
    summary = {
        "sequence": args.sequence,
        "frame_index": 0,
        "first_reference_group": list(first),
        "second_reference_group": list(second),
        "second_candidate_verified": verified,
        "memory_records": len(env.state.memory),
        "audit": audit,
        "query_gt_read": False,
        "uses_future": False,
        "selection_uses_gt": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
