#!/usr/bin/env python3
"""GT/outcome-only evaluator for the frozen AVL-008 semantic probe."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path("outputs/lilocbench/AVL-008")
SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
LABELS = ["temporary_dynamic_occlusion", "motion_blur_or_weak_observation", "persistent_scene_change", "reference_semantic_mismatch", "normal_or_compatible", "uncertain"]
COHORTS = ["memory_only", "roma_only", "both_fail", "both_success"]
MEMORY_LABELS = {"temporary_dynamic_occlusion", "motion_blur_or_weak_observation", "normal_or_compatible"}
ROMA_LABELS = {"persistent_scene_change", "reference_semantic_mismatch"}
RNG = np.random.default_rng(20261010)


def cramers_v(rows):
    if not rows:
        return float("nan")
    table = np.zeros((len(COHORTS), len(LABELS)), dtype=np.float64)
    for r in rows:
        table[COHORTS.index(r["offline_cohort"]), LABELS.index(r["label"])] += 1
    n = table.sum()
    if not n:
        return float("nan")
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    good = expected > 0
    chi2 = ((table - expected) ** 2 / np.where(good, expected, 1)).sum()
    return float(np.sqrt((chi2 / n) / min(table.shape[0] - 1, table.shape[1] - 1)))


def action(label):
    if label in MEMORY_LABELS:
        return "memory"
    if label in ROMA_LABELS:
        return "roma"
    return "abstain"


def ba(rows, use_dino=False, selective=False):
    special = [r for r in rows if r["offline_cohort"] in ("memory_only", "roma_only")]
    if selective:
        special = [r for r in special if action(r["label"]) != "abstain"]
    recalls = []
    for target in ("memory", "roma"):
        members = [r for r in special if ("memory" if r["offline_cohort"] == "memory_only" else "roma") == target]
        if not members:
            return float("nan")
        pred = [("roma" if r["dino_action"] == "roma" else "memory") if use_dino else action(r["label"]) for r in members]
        recalls.append(float(np.mean([p == target for p in pred])))
    return float(np.mean(recalls))


def stratified_bootstrap(rows, fn, count=10000):
    per_seq = {s: [r for r in rows if r["sequence"] == s] for s in SEQUENCES}
    values = []
    for _ in range(count):
        sample = []
        for s, part in per_seq.items():
            if part:
                sample.extend(part[i] for i in RNG.integers(0, len(part), len(part)))
        value = fn(sample)
        if np.isfinite(value):
            values.append(value)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main():
    manifest = json.loads((OUT / "sample_manifest.json").read_text())
    responses = {r["sample_id"]: r for r in (json.loads(x) for x in (OUT / "responses.jsonl").read_text().splitlines())}
    dino = {}
    root = Path("outputs/lilocbench/AVL-007/traces/dino_similarity/tree_depth4")
    for seq in SEQUENCES:
        dino[seq] = {r["query_index"]: r["action"] for r in (json.loads(x) for x in (root / f"{seq}.jsonl").read_text().splitlines())}
    rows, invalid = [], []
    for sample in manifest["samples"]:
        response = responses.get(sample["sample_id"])
        if not response or not response.get("valid"):
            invalid.append({"sample_id": sample["sample_id"], "offline_cohort": sample["offline_cohort"],
                            "sequence": sample["sequence"], "reason": None if not response else response.get("parse_errors")})
            continue
        rows.append({**sample, "label": response["label"], "confidence": response["confidence"],
                     "dino_action": dino[sample["sequence"]][sample["query_index"]]})
    # Cohort-label contingency and a label-permutation null.
    observed_v = cramers_v(rows)
    labels = np.asarray([r["label"] for r in rows], dtype=object)
    perm_ge = 0
    for _ in range(10000):
        shuffled = RNG.permutation(labels)
        perm_rows = [{**r, "label": label} for r, label in zip(rows, shuffled)]
        perm_ge += cramers_v(perm_rows) >= observed_v - 1e-12
    special = [r for r in rows if r["offline_cohort"] in ("memory_only", "roma_only")]
    vlm_ba = ba(rows)
    dino_ba = ba(rows, use_dino=True)
    selective = [r for r in special if action(r["label"]) != "abstain"]
    summary = {
        "protocol": {"vlm_outcome_input": False, "future_frame_input": False, "query_gt_input": False,
                     "primary_action_metric": "uncertain counts as an incorrect action", "dino_baseline": "AVL-007 dino_similarity/tree_depth4 LOSO on identical sampled frames"},
        "frames_requested": len(manifest["samples"]), "frames_valid": len(rows), "invalid_responses": invalid,
        "invalid_by_cohort": dict(Counter(x["offline_cohort"] for x in invalid)),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "cohort_label_counts": {c: dict(Counter(r["label"] for r in rows if r["offline_cohort"] == c)) for c in COHORTS},
        "cohort_label_association": {"cramers_v": observed_v, "permutation_p": float((perm_ge + 1) / 10001),
                                      "bootstrap95": stratified_bootstrap(rows, cramers_v)},
        "special_action": {
            "frames": len(special), "vlm_balanced_accuracy": vlm_ba, "dino_balanced_accuracy": dino_ba,
            "difference_vlm_minus_dino": vlm_ba - dino_ba,
            "difference_bootstrap95": stratified_bootstrap(rows, lambda x: ba(x) - ba(x, use_dino=True)),
            "vlm_memory_only_recall": float(np.mean([action(r["label"]) == "memory" for r in special if r["offline_cohort"] == "memory_only"])),
            "vlm_roma_only_recall": float(np.mean([action(r["label"]) == "roma" for r in special if r["offline_cohort"] == "roma_only"])),
            "abstain_rate": float(np.mean([action(r["label"]) == "abstain" for r in special])),
            "selective_coverage": len(selective) / len(special) if special else 0.0,
            "selective_balanced_accuracy": ba(rows, selective=True),
        },
    }
    summary["decision"] = {
        "semantic_relation_pass": bool(observed_v >= 0.30 and summary["cohort_label_association"]["permutation_p"] < 0.05 and summary["cohort_label_association"]["bootstrap95"][0] > 0.10),
        "extra_action_information_pass": bool(summary["special_action"]["difference_vlm_minus_dino"] >= 0.10 and summary["special_action"]["difference_bootstrap95"][0] > 0),
        "agent_go": False,
        "note": "Even if both probe criteria pass, AVL-008 only motivates a future held-out VLM Context Manager; it is not an Agent result.",
    }
    (OUT / "evaluator_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
