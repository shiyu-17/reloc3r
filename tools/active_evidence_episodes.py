#!/usr/bin/env python3
"""Mine causal failure-triggered recovery episodes from frozen online traces.

The miner only uses policy-visible receipts.  Ground truth is deliberately
absent: correctness is joined later by the value evaluator.  Scene splits are
made by ``scene_id`` rather than sequence name; a one-scene dataset is marked
``audit_only`` instead of being presented as scene-disjoint evidence.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TriggerConfig:
    conflict_risk: float = 1.0
    max_translation_dispersion_m: float = 1.5
    max_rotation_dispersion_deg: float = 15.0
    stale_anchor_frames: int = 10
    low_match_count: int = 20
    low_coverage_ratio: float = 0.15
    recovery_patience: int = 1
    max_episode_frames: int = 12


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trigger_reasons(row: Mapping[str, Any], config: TriggerConfig) -> list[str]:
    """Return online-observable trigger labels for one frame record."""
    online = row.get("online", row)
    reasons: set[str] = set()
    terminal = str(online.get("terminal_action", ""))
    # The miner intentionally reads no evaluator success/pose fields.  An
    # explicit accepted terminal action is the only online clean-accept cue.
    if not terminal.startswith("accept") or terminal.startswith(("defer", "reject")):
        reasons.add("verifier_failure")
    for decision in online.get("decisions", ()):
        state = decision.get("state", {}).get("state", {})
        if int(state.get("reject_streak", 0)) > 0:
            reasons.add("reject_streak")
        if int(state.get("anchor_age", 0)) >= config.stale_anchor_frames:
            reasons.add("stale_anchor")
    for action in online.get("acquisitions", online.get("program", ())):
        candidate = action.get("verified_candidate") or action.get("candidate") or {}
        evidence = candidate.get("evidence", {})
        verification = candidate.get("verification", action.get("verification", {}))
        if (_number(verification.get("total_risk")) >= config.conflict_risk or
                _number(evidence.get("translation_dispersion_m")) > config.max_translation_dispersion_m or
                _number(evidence.get("rotation_dispersion_deg")) > config.max_rotation_dispersion_deg):
            reasons.add("candidate_conflict")
        if int(verification.get("reject_streak", 0)) > 0:
            reasons.add("reject_streak")
        if int(verification.get("anchor_age", 0)) >= config.stale_anchor_frames:
            reasons.add("stale_anchor")
        cheap = evidence.get("cheap_correspondence", {})
        if cheap.get("available") and (
                int(cheap.get("valid_matches", config.low_match_count)) < config.low_match_count or
                _number(cheap.get("coverage_ratio"), 1.0) < config.low_coverage_ratio):
            reasons.add("occlusion_or_low_texture_proxy")
    if online.get("memory_corrupted") is True:
        reasons.add("corrupted_memory_injection")
    return sorted(reasons)


def mine_sequence(sequence: str, rows: Sequence[Mapping[str, Any]],
                  config: TriggerConfig) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_frame: int | None = None
    accepted_streak = 0
    for row in ordered:
        frame = int(row["frame_index"])
        reasons = trigger_reasons(row, config)
        if current is not None and previous_frame is not None and frame != previous_frame + 1:
            current["end_frame"] = previous_frame
            current["right_censored"] = True
            current["end_reason"] = "trace_gap"
            episodes.append(current)
            current = None
        if current is None and reasons:
            current = {
                "sequence": sequence,
                "start_frame": frame,
                "end_frame": frame,
                "initial_triggers": reasons,
                "trigger_types": list(reasons),
                "observed_accept": False,
                "right_censored": False,
                "end_reason": None,
            }
            accepted_streak = 0
        if current is not None:
            current["end_frame"] = frame
            current["trigger_types"] = sorted(set(current["trigger_types"]) | set(reasons))
            clean_accept = str(row.get("online", row).get("terminal_action", "")).startswith("accept") and not reasons
            accepted_streak = accepted_streak + 1 if clean_accept else 0
            if accepted_streak >= config.recovery_patience:
                current["observed_accept"] = True
                current["end_reason"] = "observed_accept"
                episodes.append(current)
                current = None
            elif frame - int(current["start_frame"]) + 1 >= config.max_episode_frames:
                current["right_censored"] = True
                current["end_reason"] = "max_episode_frames"
                episodes.append(current)
                current = None
        previous_frame = frame
    if current is not None:
        current["right_censored"] = True
        current["end_reason"] = "trace_end"
        episodes.append(current)
    for index, episode in enumerate(episodes):
        episode["episode_id"] = f"{sequence}:{episode['start_frame']:06d}:{index:03d}"
        episode["frame_count"] = episode["end_frame"] - episode["start_frame"] + 1
    return episodes


def assign_scene_splits(scene_ids: Iterable[str], seed: int = 20261014) -> dict[str, str]:
    scenes = sorted(set(str(value) for value in scene_ids))
    if len(scenes) < 3:
        return {scene: "audit_only" for scene in scenes}
    ranked = sorted(scenes, key=lambda scene: hashlib.sha256(
        f"{seed}:{scene}".encode("utf-8")).hexdigest())
    result = {}
    for index, scene in enumerate(ranked):
        fraction = index / len(ranked)
        result[scene] = "train" if fraction < 0.6 else "validation" if fraction < 0.8 else "test"
    # Small collections can miss validation under the fractional rule.
    result[ranked[-2]] = "validation"
    result[ranked[-1]] = "test"
    return result


def load_episode_spans(path: Path, warmup_frames: int = 0,
                       episode_ids: Sequence[str] = ()) -> dict[str, tuple[int, int]]:
    """Return inclusive/exclusive continuous spans covering registered episodes."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    spans: dict[str, tuple[int, int]] = {}
    wanted = set(episode_ids)
    for episode in payload.get("episodes", ()):
        if wanted and str(episode.get("episode_id")) not in wanted:
            continue
        sequence = str(episode["sequence"])
        start = max(0, int(episode["start_frame"]) - int(warmup_frames))
        end = int(episode["end_frame"]) + 1
        if sequence in spans:
            start = min(start, spans[sequence][0])
            end = max(end, spans[sequence][1])
        spans[sequence] = (start, end)
    if not spans:
        raise ValueError("episode manifest contains no episodes")
    found = {str(episode.get("episode_id")) for episode in payload.get("episodes", ())
             if not wanted or str(episode.get("episode_id")) in wanted}
    if wanted - found:
        raise ValueError("unknown episode ids: " + ", ".join(sorted(wanted - found)))
    return spans


def load_jsonl(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sequence"]), int(row["frame_index"]))
            if key in seen:
                raise ValueError(f"duplicate sequence/frame {key} from {path}")
            seen.add(key)
            rows.setdefault(key[0], []).append(row)
    return rows


def build_manifest(rows_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
                   scene_catalog: Mapping[str, Any], config: TriggerConfig,
                   *, seed: int, trace_paths: Sequence[Path]) -> dict[str, Any]:
    sequence_meta = scene_catalog.get("sequences", scene_catalog)
    missing = sorted(set(rows_by_sequence) - set(sequence_meta))
    if missing:
        raise ValueError("scene catalog missing sequences: " + ", ".join(missing))
    scene_split = assign_scene_splits(
        (sequence_meta[sequence]["scene_id"] for sequence in rows_by_sequence), seed)
    episodes = []
    for sequence, rows in sorted(rows_by_sequence.items()):
        metadata = sequence_meta[sequence]
        for episode in mine_sequence(sequence, rows, config):
            episode["scene_id"] = str(metadata["scene_id"])
            episode["condition"] = str(metadata.get("condition", sequence))
            episode["split"] = scene_split[episode["scene_id"]]
            episodes.append(episode)
    scenes = sorted(scene_split)
    return {
        "schema_version": 1,
        "protocol": "failure-triggered recovery episodes from policy-visible receipts",
        "trigger_config": asdict(config),
        "split_seed": seed,
        "scene_split": scene_split,
        "scene_disjoint_ready": len(scenes) >= 3,
        "claim_scope": "scene_disjoint" if len(scenes) >= 3 else "audit_only",
        "scene_count": len(scenes),
        "sequence_count": len(rows_by_sequence),
        "episode_count": len(episodes),
        "source_traces": [str(path) for path in trace_paths],
        "causal": {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False},
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True, help="JSONL paths or glob patterns")
    parser.add_argument("--scene-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-episode-frames", type=int, default=12)
    args = parser.parse_args()
    paths = sorted({Path(match) for pattern in args.traces for match in glob.glob(pattern)})
    if not paths:
        parser.error("--traces matched no files")
    catalog = json.loads(args.scene_catalog.read_text(encoding="utf-8"))
    config = TriggerConfig(max_episode_frames=args.max_episode_frames)
    manifest = build_manifest(load_jsonl(paths), catalog, config,
                              seed=args.seed, trace_paths=paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "scene_count", "sequence_count", "episode_count", "scene_disjoint_ready", "claim_scope")},
        indent=2))


if __name__ == "__main__":
    main()
