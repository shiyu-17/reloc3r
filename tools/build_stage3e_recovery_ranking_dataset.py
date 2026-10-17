#!/usr/bin/env python3
"""Build targeted Stage 3E recovery/cost ranking data from Oracle traces.

The builder accepts the frozen Stage 3D utility corpus and/or policy-induced
``dagger_oracle_labels.jsonl`` files.  Oracle outcomes are used only as
offline labels; the retained observation and action descriptors are available
before the selected action is executed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ACQUISITION = {"localize", "geometry", "memory"}
CHEAP = {"localize", "memory"}


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def state_key(observation: Mapping[str, Any]) -> str:
    state = observation.get("state") or {}
    candidate = observation.get("candidate") or {}
    return "|".join((
        str(observation.get("sequence", "")), str(observation.get("frame_index", "")),
        str(state.get("frame_action_count", "")), str(state.get("action_count", "")),
        str(candidate.get("source", "none")), str(candidate.get("frame_index", "")),
    ))


def make_tags(observation: Mapping[str, Any], actions: list[Mapping[str, Any]]) -> set[str]:
    candidate = observation.get("candidate") or {}
    state = observation.get("state") or {}
    tags = {"recovery" if candidate else "root"}
    if candidate:
        tags.add("candidate:" + str(candidate.get("source", "unknown")))
    if int(state.get("reject_streak", 0) or 0) >= 2:
        tags.add("reject_streak")
    if int(state.get("memory_size", 0) or 0) > 0:
        tags.add("memory_available")
    if any(str(action["kind"]) == "geometry" for action in actions):
        tags.add("geometry_available")
    return tags


def geometry_margin(actions: list[Mapping[str, Any]]) -> float | None:
    geometry = [number(action["oracle_value"]) for action in actions if action["kind"] == "geometry"]
    cheap = [number(action["oracle_value"]) for action in actions if action["kind"] in CHEAP]
    return None if not geometry or not cheap else max(geometry) - max(cheap)


def targeting_weight(tags: set[str], margin: float, boundary: float) -> tuple[str, float]:
    if margin > 0.0 and "recovery" in tags:
        return "geometry_positive_recovery", 4.0
    if margin > 0.0:
        return "geometry_positive_root", 3.0
    if abs(margin) <= boundary:
        return "geometry_boundary", 2.0
    if "recovery" in tags:
        return "recovery_negative", 1.5
    return "root_negative", 1.0


def _standard_action(action: Mapping[str, Any], *, value: float, metrics: Mapping[str, Any], prior_cost: float) -> dict[str, Any]:
    return {
        "kind": str(action.get("kind")),
        "references": list(action.get("references") or []),
        "is_terminal_action": False,
        "recovery_target": float(bool(metrics.get("correct_accept", False))),
        "incremental_cost_s": max(0.0, number(metrics.get("measured_tool_cost_s"), prior_cost) - prior_cost),
        "oracle_value": value,
    }


def from_utility_row(row: Mapping[str, Any], boundary_margin: float) -> dict[str, Any] | None:
    observation = row.get("observation") or {}
    prior = number((observation.get("state") or {}).get("frame_tool_cost_s"))
    actions = []
    for action in row.get("actions") or []:
        if str(action.get("kind")) not in ACQUISITION:
            continue
        actions.append(_standard_action(
            action, value=number(action.get("oracle_incremental_utility", action.get("q_star"))),
            metrics=action.get("best_terminal_metrics") or {}, prior_cost=prior))
    return make_record(observation, actions, str(row.get("sequence", "")), int(row.get("frame_index", 0)),
                       str(row.get("state_id", state_key(observation))), "frozen_oracle", boundary_margin)


def from_dagger_row(row: Mapping[str, Any], source: str, boundary_margin: float) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for node in row.get("oracle_action_values") or []:
        if int(node.get("depth", 1)) != 1:
            continue
        action = node.get("action") or {}
        if str(action.get("kind")) not in ACQUISITION:
            continue
        observation = node.get("state") or {}
        key = state_key(observation)
        group = grouped.setdefault(key, {"observation": observation, "actions": []})
        prior = number((observation.get("state") or {}).get("frame_tool_cost_s"))
        value = node.get("action_value") or {}
        group["actions"].append(_standard_action(action, value=number(value.get("q_star")),
                                                    metrics=value.get("best_terminal_metrics") or {}, prior_cost=prior))
    records = []
    for key, group in grouped.items():
        record = make_record(group["observation"], group["actions"], str(row.get("sequence", "")),
                             int(row.get("frame_index", 0)), key, source, boundary_margin)
        if record is not None:
            records.append(record)
    return records


def make_record(observation: Mapping[str, Any], actions: list[dict[str, Any]], sequence: str,
                frame_index: int, identifier: str, source: str, boundary_margin: float) -> dict[str, Any] | None:
    kinds = {action["kind"] for action in actions}
    if "geometry" not in kinds or not (kinds & CHEAP):
        return None
    margin = geometry_margin(actions)
    assert margin is not None
    tags = make_tags(observation, actions)
    reason, weight = targeting_weight(tags, margin, boundary=boundary_margin)
    return {
        "state_id": identifier, "sequence": sequence, "frame_index": frame_index,
        "observation": observation, "actions": actions, "source_type": source,
        "target_tags": sorted(tags), "sampling_reason": reason,
        "geometry_margin": margin, "training_weight": weight,
    }


def iter_source(path: Path, boundary_margin: float) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "actions" in row and "observation" in row:
            record = from_utility_row(row, boundary_margin)
            if record is not None:
                yield record
        elif "oracle_action_values" in row:
            yield from from_dagger_row(row, "dagger", boundary_margin)
        else:
            raise ValueError(f"unrecognized input row in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-keep-probability", type=float, default=0.35)
    parser.add_argument("--boundary-margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1777)
    args = parser.parse_args()
    if not 0.0 <= args.negative_keep_probability <= 1.0 or args.boundary_margin < 0.0:
        parser.error("invalid negative keep probability or boundary margin")
    rng = random.Random(args.seed)
    records, counts = [], Counter()
    seen = set()
    for path in args.input:
        for record in iter_source(path, args.boundary_margin):
            # Natural root negatives are downsampled only in training data;
            # all positives, boundary states and recovery states are retained.
            if record["sampling_reason"] == "root_negative" and rng.random() > args.negative_keep_probability:
                continue
            dedupe = (record["source_type"], record["state_id"], tuple((a["kind"], tuple(a["references"])) for a in record["actions"]))
            if dedupe in seen:
                continue
            seen.add(dedupe); records.append(record); counts[record["sampling_reason"]] += 1
    records.sort(key=lambda row: (row["source_type"], row["sequence"], row["frame_index"], row["state_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "recovery_cost_pairwise.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema_version": 1, "purpose": "Stage 3E targeted recovery/cost pairwise ranking corpus",
        "inputs": [str(path) for path in args.input], "states": len(records),
        "sampling_reason_counts": dict(sorted(counts.items())), "negative_keep_probability": args.negative_keep_probability,
        "boundary_margin": args.boundary_margin,
        "online_boundary": "only pre-action observations and legal action descriptors are retained; Oracle labels are offline training targets",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
