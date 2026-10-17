#!/usr/bin/env python3
"""Evaluate episode-level active-evidence value and apply the frozen Go gate."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FrameKey = tuple[str, int]


def load_named_traces(specs: Sequence[str]) -> dict[str, dict[FrameKey, dict[str, Any]]]:
    result: dict[str, dict[FrameKey, dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("trace must be NAME=PATH_OR_GLOB")
        name, pattern = spec.split("=", 1)
        target = result.setdefault(name, {})
        for match in sorted(glob.glob(pattern)):
            for line in Path(match).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["sequence"]), int(row["frame_index"]))
                if key in target:
                    raise ValueError(f"duplicate {name} trace row {key}")
                target[key] = row
    return result


def episode_result(episode: Mapping[str, Any], rows: Mapping[FrameKey, Mapping[str, Any]],
                   is_correct: Callable[[str, Mapping[str, Any]], bool]) -> dict[str, Any]:
    sequence = str(episode["sequence"])
    start, end = int(episode["start_frame"]), int(episode["end_frame"])
    expected_frames = list(range(start, end + 1))
    selected = [rows[(sequence, frame)] for frame in expected_frames if (sequence, frame) in rows]
    missing = [frame for frame in expected_frames if (sequence, frame) not in rows]
    def online(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return row.get("online", row)
    def evaluator(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return row.get("evaluator", row)
    accepted = [row for row in selected if bool(evaluator(row).get("accepted", False))]
    correct = [row for row in accepted if is_correct(sequence, row)]
    false_accepts = len(accepted) - len(correct)
    first_correct = None if not correct else min(int(row["frame_index"]) for row in correct) - start
    return {
        "episode_id": episode["episode_id"], "scene_id": episode["scene_id"],
        "sequence": sequence, "split": episode["split"],
        "frames_expected": len(expected_frames), "frames_available": len(selected),
        "missing_frames": missing,
        "right_censored": bool(episode.get("right_censored", False) or missing),
        "episode_success": bool(correct), "accepted": len(accepted),
        "false_accepts": false_accepts,
        "episode_has_false_accept": bool(false_accepts),
        "time_to_first_correct_frames": first_correct,
        "memory_writes": int(sum(bool(evaluator(row).get("memory_write", False)) for row in selected)),
        "memory_pollution_events": int(sum(
            bool(evaluator(row).get("memory_write", False)) and
            bool(evaluator(row).get("accepted", False)) and not is_correct(sequence, row)
            for row in selected)),
        "tool_calls": int(sum(int(online(row).get("tool_actions", len(online(row).get("acquisitions", ()))))
                              for row in selected)),
        "measured_tool_cost_s": float(sum(float(online(row).get("measured_tool_cost_s", 0.0))
                                          for row in selected)),
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(results)
    accepted = sum(int(row["accepted"]) for row in results)
    false_accepts = sum(int(row["false_accepts"]) for row in results)
    recovered = [row for row in results if row["episode_success"]]
    censored = [row for row in results if row["right_censored"] and not row["episode_success"]]
    unrecovered = [row for row in results if not row["episode_success"] and not row["right_censored"]]
    return {
        "episodes": count,
        "episode_successes": len(recovered),
        "episode_success_rate": len(recovered) / count if count else 0.0,
        "unrecovered_rate": len(unrecovered) / count if count else 0.0,
        "right_censored_episodes": len(censored),
        "right_censored_rate": len(censored) / count if count else 0.0,
        "missing_trace_frames": int(sum(len(row["missing_frames"]) for row in results)),
        "false_accepts": false_accepts,
        "episodes_with_false_accept": sum(int(bool(row["episode_has_false_accept"])) for row in results),
        "false_accept_rate": false_accepts / accepted if accepted else 0.0,
        "memory_writes": sum(int(row["memory_writes"]) for row in results),
        "memory_pollution_events": sum(int(row["memory_pollution_events"]) for row in results),
        "mean_time_to_first_correct_frames": (float(np.mean([
            row["time_to_first_correct_frames"] for row in recovered])) if recovered else None),
        "mean_tool_calls": float(np.mean([row["tool_calls"] for row in results])) if results else 0.0,
        "mean_measured_tool_cost_s": float(np.mean([
            row["measured_tool_cost_s"] for row in results])) if results else 0.0,
    }


def action_dependency(action_rows: Sequence[Mapping[str, Any]],
                      episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    membership = {(str(ep["sequence"]), frame)
                  for ep in episodes for frame in range(int(ep["start_frame"]), int(ep["end_frame"]) + 1)}
    states = terminal_dependent = utility_dependent = 0
    seen = set()
    for row in action_rows:
        payload = row.get("evaluator", row)
        # Oracle rollouts store values directly.  Policy-induced audits store
        # the same evaluator-only payload per online decision.  Treat both
        # forms identically while retaining the decision index in the key.
        labels = payload.get("oracle_decisions", ())
        if labels:
            candidates = ((int(item.get("decision_index", 0)), item.get("oracle_label", {}))
                          for item in labels)
        else:
            candidates = ((int(row.get("decision_index", 0)), payload),)
        for decision_index, label in candidates:
            key = (str(row["sequence"]), int(row["frame_index"]), decision_index)
            if key[:2] not in membership or key in seen:
                continue
            seen.add(key)
            roots = [node for node in label.get("oracle_action_values", ())
                     if not node.get("parent_path_node_ids")]
            outcomes = {bool(node.get("action_value", {}).get("best_terminal_success", 0)) for node in roots}
            q_values = {round(float(node.get("action_value", {}).get("q_star", 0.0)), 9) for node in roots}
            if roots:
                states += 1
                terminal_dependent += int(len(outcomes) > 1)
                utility_dependent += int(len(q_values) > 1)
    return {"states": states,
            "action_dependent_states": terminal_dependent,
            "action_dependent_fraction": terminal_dependent / states if states else 0.0,
            "utility_dependent_states": utility_dependent,
            "utility_dependent_fraction": utility_dependent / states if states else 0.0}


def evaluate(manifest: Mapping[str, Any], traces: Mapping[str, Mapping[FrameKey, Mapping[str, Any]]],
             is_correct: Callable[[str, Mapping[str, Any]], bool], *, oracle_name: str,
             fixed_candidates: Sequence[str], action_rows: Sequence[Mapping[str, Any]],
             gate: Mapping[str, Any], predictive_signal_passed: bool = False) -> dict[str, Any]:
    episodes = list(manifest["episodes"])
    results = {name: [episode_result(ep, rows, is_correct) for ep in episodes]
               for name, rows in traces.items()}
    available_splits = {str(ep["split"]) for ep in episodes}
    selection_split = "validation" if "validation" in available_splits else "audit_only"
    evaluation_split = "test" if "test" in available_splits else "audit_only"
    summaries = {name: {
        split: aggregate([row for row in policy_rows if row["split"] == split])
        for split in sorted(available_splits)} for name, policy_rows in results.items()}
    candidates = [name for name in fixed_candidates if name in summaries]
    if not candidates:
        raise ValueError("no fixed candidate policy traces were supplied")
    best_fixed = max(candidates, key=lambda name: (
        summaries[name][selection_split]["episode_success_rate"],
        -summaries[name][selection_split]["mean_measured_tool_cost_s"]))
    if oracle_name not in summaries:
        raise ValueError(f"missing oracle trace: {oracle_name}")
    base = summaries[best_fixed][evaluation_split]
    oracle = summaries[oracle_name][evaluation_split]
    success_gain_pp = 100.0 * (oracle["episode_success_rate"] - base["episode_success_rate"])
    cost_reduction = ((base["mean_measured_tool_cost_s"] - oracle["mean_measured_tool_cost_s"]) /
                      base["mean_measured_tool_cost_s"] if base["mean_measured_tool_cost_s"] else 0.0)
    dependency = action_dependency(action_rows, [ep for ep in episodes
                                                 if ep["split"] == evaluation_split])
    success_or_cost = (
        success_gain_pp >= float(gate["minimum_oracle_success_gain_pp"]) or
        (oracle["episode_success_rate"] >= max(
             base["episode_success_rate"],
             float(gate.get("minimum_matched_episode_success_rate", 0.1))) and
         cost_reduction >= float(gate["or_minimum_cost_reduction_fraction_at_matched_success"])))
    safety = (oracle["false_accept_rate"] <= base["false_accept_rate"] and
              oracle["memory_pollution_events"] <= base["memory_pollution_events"])
    scene_deltas = {}
    eval_scenes = sorted({str(ep["scene_id"]) for ep in episodes if ep["split"] == evaluation_split})
    for scene in eval_scenes:
        base_scene = aggregate([row for row in results[best_fixed]
                                if row["split"] == evaluation_split and row["scene_id"] == scene])
        oracle_scene = aggregate([row for row in results[oracle_name]
                                  if row["split"] == evaluation_split and row["scene_id"] == scene])
        scene_deltas[scene] = oracle_scene["episode_success_rate"] - base_scene["episode_success_rate"]
    scene_consistent = len(scene_deltas) >= 1 and all(value >= 0.0 for value in scene_deltas.values())
    realizable = bool(manifest.get("scene_disjoint_ready") and success_or_cost and safety and
                      scene_consistent and dependency["action_dependent_fraction"] >=
                      float(gate["minimum_action_dependent_state_fraction"]))
    return {
        "schema_version": 1, "claim_scope": manifest.get("claim_scope"),
        "selection_split": selection_split, "evaluation_split": evaluation_split,
        "best_fixed_policy": best_fixed, "policy_summaries": summaries,
        "comparison": {"oracle": oracle_name, "oracle_success_gain_pp": success_gain_pp,
                       "oracle_cost_reduction_fraction": cost_reduction,
                       "scene_success_deltas": scene_deltas, **dependency},
        "gate": {"success_or_cost": success_or_cost, "safety": safety,
                 "scene_consistent": scene_consistent,
                 "scene_disjoint_ready": bool(manifest.get("scene_disjoint_ready")),
                 "realizable_value_pass": realizable,
                 "predictive_preaction_signal_passed": predictive_signal_passed,
                 "learned_agent_go": bool(realizable and predictive_signal_passed)},
        "episode_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-ids", nargs="*", default=(),
                        help="optional evaluated episode subset, matching runner shards")
    parser.add_argument("--trace", action="append", required=True, help="NAME=PATH_OR_GLOB")
    parser.add_argument("--oracle-name", default="finite_depth_oracle")
    parser.add_argument("--fixed-candidates", nargs="+", default=["fixed", "fixed_geometry"])
    parser.add_argument("--oracle-labels", nargs="*", default=[])
    parser.add_argument("--protocol-config", type=Path,
                        default=Path("configs/active_evidence_realizable_value_v1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--predictive-signal-passed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.episode_ids:
        wanted = set(args.episode_ids)
        manifest["episodes"] = [episode for episode in manifest["episodes"]
                                if str(episode.get("episode_id")) in wanted]
        found = {str(episode["episode_id"]) for episode in manifest["episodes"]}
        if wanted - found:
            parser.error("unknown --episode-ids: " + ", ".join(sorted(wanted - found)))
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    traces = load_named_traces(args.trace)
    from tools.eval_lilocbench_reloc3r import (
        base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses,
    )
    gt = {}
    for sequence in {str(ep["sequence"]) for ep in manifest["episodes"]}:
        _, _, query_times, _, query_root = read_manifest(args.data_root, sequence)
        gt[sequence], _ = tum_camera_poses(
            args.gt_root / f"{sequence}_gt_poses.txt", query_times,
            base_to_front_rgb_optical(query_root), 0.1)

    def is_correct(sequence: str, row: Mapping[str, Any]) -> bool:
        pose = row.get("evaluator", row).get("pose")
        if pose is None:
            return False
        translation, rotation = pose_error(
            np.asarray(pose, dtype=np.float64), gt[sequence][int(row["frame_index"])])
        return bool(translation <= 0.25 and rotation <= 5.0)

    action_rows = []
    for pattern in args.oracle_labels:
        for match in glob.glob(pattern):
            action_rows.extend(json.loads(line) for line in Path(match).read_text().splitlines() if line.strip())
    report = evaluate(manifest, traces, is_correct, oracle_name=args.oracle_name,
                      fixed_candidates=args.fixed_candidates, action_rows=action_rows,
                      gate=protocol["go_gate"],
                      predictive_signal_passed=args.predictive_signal_passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_fixed_policy": report["best_fixed_policy"],
                      "comparison": report["comparison"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
