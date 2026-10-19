#!/usr/bin/env python3
"""Run the LILocBench v1 precomputed-reference baseline.

Map poses and precomputed retrieval rankings are inputs. Query ground truth is
not loaded here; the retrieval stage is explicitly marked precomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tools.agentic_localization_env import AgenticLocalizationEnv
from tools.lilocbench_agent_backend import LILocBenchReloc3rBackend
from tools.unified_localization_policy import run_fixed_order


ROOT = Path(__file__).resolve().parents[1]


def run(sequence: str, data_root: Path, mapping_gt: Path, retrieval_npz: Path,
        start: int, count: int, device: str, max_actions: int, budget_s: float,
        *, geometry_on_failure: bool = False, verifier_config: Path | None = None,
        backend=None) -> dict:
    if max_actions < 3 or budget_s <= 0.:
        raise ValueError("max_actions must be at least 3 and budget_s positive")
    setup_started = time.perf_counter()
    if backend is None:
        backend = LILocBenchReloc3rBackend(
            data_root, mapping_gt, sequence, retrieval_npz, device=device,
            verifier_config=verifier_config)
    if backend.live_capabilities.reference_retrieval != "precomputed":
        raise ValueError("v1 requires precomputed NetVLAD reference rankings")
    setup_s = time.perf_counter() - setup_started
    env = AgenticLocalizationEnv(backend, max_actions_per_frame=max_actions,
                                 require_online_execution=True, budget_s=budget_s,
                                 enforce_tool_budget=True, allow_memory_query=False)
    result = run_fixed_order(env, sequence, start, count, backend.live_capabilities,
                             geometry_on_failure=geometry_on_failure)
    source_files = [Path(__file__), ROOT / "tools/unified_localization_policy.py",
                    ROOT / "tools/active_evidence_protocol.py",
                    ROOT / "tools/agentic_localization_env.py",
                    ROOT / "tools/lilocbench_agent_backend.py",
                    ROOT / "tools/lilocbench_geometry.py",
                    ROOT / "reloc3r/reloc3r_relpose.py",
                    ROOT / "reloc3r/reloc3r_visloc.py"]
    input_files = [mapping_gt, retrieval_npz]
    if verifier_config is not None:
        input_files.append(verifier_config)
    weights = sorted(Path("/home/liusy/.cache/huggingface/hub/models--siyan824--reloc3r-512/snapshots").glob(
        "*/model.safetensors"))
    if len(weights) == 1:
        input_files.append(weights[0])
    roma_weights = Path("/home/liusy/.cache/torch/hub/checkpoints/roma_indoor.pth")
    if geometry_on_failure and roma_weights.exists():
        input_files.append(roma_weights)
    result.update({
        "dataset": "LILocBench", "sequence": sequence,
        "policy": "Reloc3r groups in order; optional RoMa geometry after group failures; verify each available pose; accept first verified candidate or defer",
        "budget_s_per_frame": float(budget_s),
        "model_setup_wall_s": float(setup_s),
        "cost_complete_online_inference": False,
        "cost_limitation": "NetVLAD rankings were precomputed; their production cost is not included",
        "mapping_gt_role": "reference map poses only; query GT is not loaded",
        "source_fingerprints_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files},
        "input_fingerprints_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_files if path.exists()},
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--mapping-gt", type=Path,
                        default=Path("data/lilocbench/gt/mapping_gt_poses.txt"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("outputs/lilocbench/AVL-002"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-actions-per-frame", type=int, default=12)
    parser.add_argument("--budget-s", type=float, default=20.)
    parser.add_argument("--geometry-on-failure", action="store_true")
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.sequence, args.data_root, args.mapping_gt,
                 args.retrieval_root / args.sequence / "retrieval.npz",
                 args.start, args.count, args.device, args.max_actions_per_frame,
                 args.budget_s, geometry_on_failure=args.geometry_on_failure,
                 verifier_config=args.verifier_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"sequence": args.sequence,
                      "frames": result["query_count_executed"],
                      "environment_audit": result["environment_audit"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
