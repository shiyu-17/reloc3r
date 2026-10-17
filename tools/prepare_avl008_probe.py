#!/usr/bin/env python3
"""Freeze AVL-008 stratified samples and redacted VLM multi-image requests.

Outcome cohorts are used only to balance the sample manifest.  The generated
request file intentionally contains no cohort, pose, GT, or action outcome.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
SEED = 20261010
TARGET_PER_COHORT = 54
DATA_ROOT = Path("data/lilocbench/prepared_1hz")
MEMORY_ROOT = Path("outputs/lilocbench/AVL-003")
ROMA_ROOT = Path("outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0")
RETRIEVAL_ROOT = Path("outputs/lilocbench/AVL-002")
OUT_ROOT = Path("outputs/lilocbench/AVL-008")

PROMPT = """You are assessing observation context for visual localization. You receive images in this exact order:
1) CURRENT query image; 2-4) mapping REFERENCES retrieved for the current image; remaining images) HISTORY, strictly earlier query frames.
Do not infer a camera pose and do not choose a localization algorithm. Classify only the visual context by returning exactly one JSON object, with no markdown:
{"label":"one allowed label","confidence":0.0,"evidence":"at most 20 English words"}
Allowed labels: temporary_dynamic_occlusion, motion_blur_or_weak_observation, persistent_scene_change, reference_semantic_mismatch, normal_or_compatible, uncertain.
Use uncertain when the supplied images do not support a confident distinction. Evidence must describe visible cues only, not hidden reasoning.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cohort(memory: bool, roma: bool) -> str:
    if memory and not roma:
        return "memory_only"
    if roma and not memory:
        return "roma_only"
    if not memory and not roma:
        return "both_fail"
    return "both_success"


def proportional_sample(items, n, rng):
    """Sample by sequence in proportion to cohort size, with deterministic ties."""
    by_seq = defaultdict(list)
    for item in items:
        by_seq[item["sequence"]].append(item)
    total = len(items)
    if n >= total:
        return sorted(items, key=lambda x: (x["sequence"], x["query_index"]))
    raw = {s: n * len(by_seq[s]) / total for s in SEQUENCES}
    take = {s: min(len(by_seq[s]), int(raw[s])) for s in SEQUENCES}
    left = n - sum(take.values())
    order = sorted(SEQUENCES, key=lambda s: (raw[s] - int(raw[s]), len(by_seq[s])), reverse=True)
    for s in order:
        if not left:
            break
        if take[s] < len(by_seq[s]):
            take[s] += 1
            left -= 1
    selected = []
    for s in SEQUENCES:
        part = sorted(by_seq[s], key=lambda x: x["query_index"])
        if take[s]:
            ids = rng.choice(len(part), size=take[s], replace=False)
            selected.extend(part[i] for i in sorted(ids.tolist()))
    return sorted(selected, key=lambda x: (x["sequence"], x["query_index"]))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mapping_manifest = json.loads((DATA_ROOT / "mapping" / "manifest.json").read_text())
    mapping_paths = [DATA_ROOT / "mapping" / x["path"] for x in mapping_manifest["frames"]]
    cohort_items = defaultdict(list)
    seq_manifests = {}
    for seq in SEQUENCES:
        manifest = json.loads((DATA_ROOT / seq / "manifest.json").read_text())
        seq_manifests[seq] = manifest
        memory = [bool(json.loads(x)["methods"]["memory3"].get("success_0.25m_5deg", False))
                  for x in (MEMORY_ROOT / seq / "frame_results.jsonl").read_text().splitlines()]
        roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False))
                for x in json.loads((ROMA_ROOT / f"{seq}.json").read_text())["frames"]]
        assert len(memory) == len(roma) == len(manifest["frames"])
        for i, (m, r) in enumerate(zip(memory, roma)):
            path = DATA_ROOT / seq / manifest["frames"][i]["path"]
            cohort_items[cohort(m, r)].append({
                "sequence": seq, "query_index": i, "timestamp": manifest["frames"][i]["timestamp"],
                "current_path": str(path), "current_sha256": sha256(path),
            })
    rng = np.random.default_rng(SEED)
    selected = []
    for name in ("memory_only", "roma_only", "both_fail", "both_success"):
        chosen = proportional_sample(cohort_items[name], min(TARGET_PER_COHORT, len(cohort_items[name])), rng)
        for row in chosen:
            row["offline_cohort"] = name
        selected.extend(chosen)

    requests = []
    for sample_id, row in enumerate(sorted(selected, key=lambda x: (x["offline_cohort"], x["sequence"], x["query_index"]))):
        seq, i = row["sequence"], row["query_index"]
        retrieval = np.load(RETRIEVAL_ROOT / seq / "retrieval.npz")
        refs = [mapping_paths[int(x)] for x in retrieval["mapping_indices"][i, :3]]
        history = [DATA_ROOT / seq / seq_manifests[seq]["frames"][j]["path"] for j in range(max(0, i - 3), i)]
        request = {
            "sample_id": sample_id, "sequence": seq, "query_index": i, "timestamp": row["timestamp"],
            "current_path": row["current_path"], "reference_paths": [str(x) for x in refs],
            "history_paths": [str(x) for x in history], "prompt": PROMPT,
            "query_gt_read": False, "future_frame_read": False, "outcome_or_action_read": False,
        }
        requests.append(request)
        row["sample_id"] = sample_id
        row["reference_paths"] = [str(x) for x in refs]
        row["history_paths"] = [str(x) for x in history]
    manifest = {
        "experiment": "AVL-008", "sampling_seed": SEED, "target_per_cohort": TARGET_PER_COHORT,
        "cohort_counts_all": {k: len(v) for k, v in cohort_items.items()},
        "cohort_counts_selected": {k: sum(x["offline_cohort"] == k for x in selected) for k in cohort_items},
        "selection_note": "offline cohort is for stratified sampling/evaluation only and is absent from request JSONL",
        "samples": sorted(selected, key=lambda x: x["sample_id"]),
    }
    (OUT_ROOT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT_ROOT / "requests.jsonl").open("w") as f:
        for row in requests:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (OUT_ROOT / "prompt.txt").write_text(PROMPT)
    print(json.dumps({"selected": len(selected), "counts": manifest["cohort_counts_selected"]}, indent=2))


if __name__ == "__main__":
    main()
