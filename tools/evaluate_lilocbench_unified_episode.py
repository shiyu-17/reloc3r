#!/usr/bin/env python3
"""Score a saved unified LILocBench trace after online execution has ended."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical, read_manifest, tum_camera_poses,
)


def load_query_gt(data_root: Path, gt_root: Path, sequence: str):
    _, _, query_times, _, query_root = read_manifest(data_root, sequence)
    poses, _ = tum_camera_poses(
        gt_root / f"{sequence}_gt_poses.txt", query_times,
        base_to_front_rgb_optical(query_root), 0.1)
    return poses


def error(estimate, truth):
    estimate, truth = np.asarray(estimate, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    relative = estimate[:3, :3].T @ truth[:3, :3]
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))
    return {"translation_m": translation, "rotation_deg": rotation,
            "correct_0_5m_10deg": translation <= .5 and rotation <= 10.,
            "correct_0_25m_5deg": translation <= .25 and rotation <= 5.}


def evaluate(trace_path: Path, data_root: Path, gt_root: Path) -> dict:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    retrieval_mode = trace.get("reference_retrieval_provenance")
    if trace.get("dataset") != "LILocBench" or retrieval_mode != "precomputed":
        raise ValueError("expected a LILocBench v1 precomputed-reference trace")
    if not trace.get("environment_audit", {}).get("passed") or trace.get("evaluator_gt_loaded") is not False:
        raise ValueError("online trace did not pass the GT isolation audit")
    sequence = trace["sequence"]
    gt = load_query_gt(data_root, gt_root, sequence)
    by_frame = defaultdict(list)
    for record in trace["trace"]:
        if record.get("type") == "step":
            by_frame[int(record["frame_index_before"])].append(record)
    indices = [int(x["query_index"]) for x in trace["decisions"]]
    if len(indices) != len(set(indices)) or set(indices) != set(by_frame):
        raise ValueError("summary frames disagree with executed frames")
    rows = []
    total_cost_by_action = defaultdict(float)
    for decision in trace["decisions"]:
        index = int(decision["query_index"])
        if not 0 <= index < len(gt) or int(decision["query_frame_id"]) != index:
            raise ValueError("query frame identity mismatch")
        candidates = []
        active = None
        terminal = None
        accepted_index = None
        for record in by_frame[index]:
            kind = record["action"]["kind"]
            event = record["event"]
            if terminal is not None:
                raise ValueError("action after terminal")
            elapsed = event.get("execution", {}).get(
                "elapsed_s", event.get("evidence", {}).get("elapsed_s", 0.))
            total_cost_by_action[kind] += max(0., float(elapsed))
            if kind in ("localize_with_references", "run_strong_geometry"):
                expected = "localize" if kind == "localize_with_references" else "strong_geometry"
                if event.get("type") != expected or list(event.get("references", ())) != list(
                        record["action"]["reference_indices"]):
                    raise ValueError("pose action and receipt disagree")
                active = len(candidates)
                candidates.append({"pose": event.get("pose"), "verified": False,
                                   "verification_status": "unavailable"})
            elif kind == "verify_candidate":
                if active is None or candidates[active]["pose"] is None or event.get("type") != "verify":
                    raise ValueError("verification has no active candidate")
                candidates[active]["verified"] = bool(event["accepted"])
                candidates[active]["verification_status"] = event["evidence"].get(
                    "status", "supported" if event["accepted"] else "rejected")
            elif kind == "select_candidate":
                active = int(record["action"]["candidate_index"])
                if active < 0 or active >= len(candidates) or candidates[active]["pose"] is None:
                    raise ValueError("selected candidate is unavailable")
            elif kind == "accept_candidate":
                if active is None or not candidates[active]["verified"] or event.get("type") != "accept":
                    raise ValueError("accepted candidate was not verified")
                terminal, accepted_index = "accept", active
            elif kind == "defer_frame":
                if event.get("type") != "defer_frame":
                    raise ValueError("defer action and receipt disagree")
                terminal = "defer"
            else:
                raise ValueError(f"unsupported action {kind}")
        if (decision["terminal"] != terminal or
                decision["accepted_candidate_index"] != accepted_index):
            raise ValueError("decision summary disagrees with executed trace")
        attempts = decision["groups_attempted"]
        if len(attempts) != len(candidates):
            raise ValueError("candidate count disagrees with summary")
        for number, (attempt, candidate) in enumerate(zip(attempts, candidates)):
            if (int(attempt["candidate_index"]) != number or
                    bool(attempt["accepted_by_verifier"]) != candidate["verified"] or
                    attempt["verification_status"] != candidate["verification_status"]):
                raise ValueError("verification summary disagrees with executed trace")
        scored = [error(candidate["pose"], gt[index]) if candidate["pose"] is not None else None
                  for candidate in candidates]
        accepted = scored[accepted_index] if accepted_index is not None else None
        rows.append({"query_index": index, "terminal": terminal,
                     "accepted_candidate_index": accepted_index,
                     "accepted_error": accepted,
                     "candidate_errors": scored,
                     "correct_candidate_available_0_5m_10deg": any(
                         x is not None and x["correct_0_5m_10deg"] for x in scored)})
    accepted_rows = [x for x in rows if x["accepted_error"] is not None]
    correct_broad = sum(x["accepted_error"]["correct_0_5m_10deg"] for x in accepted_rows)
    correct_strict = sum(x["accepted_error"]["correct_0_25m_5deg"] for x in accepted_rows)
    return {"schema_version": "lilocbench_v1_posthoc",
            "source_trace": str(trace_path),
            "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sequence": sequence, "query_count": len(rows),
            "reference_retrieval_provenance": retrieval_mode,
            "accepted_count": len(accepted_rows),
            "correct_accept_count_0_5m_10deg": correct_broad,
            "false_accept_count_0_5m_10deg": len(accepted_rows) - correct_broad,
            "correct_accept_count_0_25m_5deg": correct_strict,
            "false_accept_count_0_25m_5deg": len(accepted_rows) - correct_strict,
            "queries_with_correct_candidate_but_deferred": sum(
                row["accepted_error"] is None and row["correct_candidate_available_0_5m_10deg"]
                for row in rows),
            "online_cost_seconds_by_action": dict(total_cost_by_action),
            "online_cost_seconds_total": float(sum(total_cost_by_action.values())),
            "cost_complete_online_inference": bool(trace["cost_complete_online_inference"]),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.trace, args.data_root, args.gt_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "sequence", "query_count", "accepted_count",
        "correct_accept_count_0_5m_10deg", "false_accept_count_0_5m_10deg",
        "online_cost_seconds_total")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
