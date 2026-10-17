#!/usr/bin/env python3
"""Aggregate seeded RoMa recovery routes without using GT for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def load_route(path):
    return {int(row["query_index"]): row for row in json.loads(path.read_text())["frames"]}


def bootstrap_ci(values, rng, n=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [None, None]
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def success(row):
    return bool(row.get("metric") and row["metric"].get("success_0.25m_5deg"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="AVL-005 root containing seed{0,1,2}/route_eval")
    p.add_argument("--manager-route", type=Path, required=True,
                   help="A.5 route_eval directory for HLoc and action gating")
    p.add_argument("--avl003-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(20261010)
    report = {
        "protocol": "3 fixed RoMa seeds on identical A.5 952-frame geometry pool; GT evaluator-only",
        "seeds": [0, 1, 2],
        "sequences": {},
    }
    for seq in SEQUENCES:
        manager = load_route(args.manager_route / f"roma_{seq}.json")
        hloc = load_route(args.manager_route / f"hloc_{seq}.json")
        call_indices = sorted(qi for qi, row in manager.items() if row.get("action") == "call_geometry")
        seed_routes = [load_route(args.root / f"seed{s}" / "route_eval" / f"roma_{seq}.json") for s in report["seeds"]]
        seed_rates = [sum(success(route[qi]) for qi in call_indices) / len(call_indices) for route in seed_routes]
        seed_full = [sum(success(route[qi]) for qi in call_indices) for route in seed_routes]
        common_sets = []
        per_seed_confusion = []
        oracle_rates = []
        avl003 = {int(row["query_index"]): row for row in
                  (json.loads(line) for line in (args.avl003_root / seq / "frame_results.jsonl").read_text().splitlines())}
        for route in seed_routes:
            common = {qi for qi in call_indices if not success(hloc[qi]) and not success(route[qi])}
            common_sets.append(common)
            both = hloc_only = roma_only = neither = 0
            oracle_success = 0
            for qi in call_indices:
                hs, rs = success(hloc[qi]), success(route[qi])
                if hs and rs: both += 1
                elif hs: hloc_only += 1
                elif rs: roma_only += 1
                else: neither += 1
                base = avl003[qi]["methods"]
                any_success = (bool(base["consensus3"]["success_0.25m_5deg"]) or
                               bool(base["memory3"]["success_0.25m_5deg"]) or hs or rs)
                oracle_success += int(any_success)
            per_seed_confusion.append({"both": both, "hloc_only": hloc_only,
                                       "roma_only": roma_only, "neither": neither})
            oracle_rates.append(oracle_success / len(call_indices))
        jaccard = []
        for i in range(len(common_sets)):
            for j in range(i + 1, len(common_sets)):
                union = common_sets[i] | common_sets[j]
                jaccard.append(len(common_sets[i] & common_sets[j]) / len(union) if union else 1.0)
        report["sequences"][seq] = {
            "geometry_call_frames": len(call_indices),
            "success_counts": seed_full,
            "success_rates": seed_rates,
            "success_mean": float(np.mean(seed_rates)),
            "success_sample_std": float(np.std(seed_rates, ddof=1)),
            "success_bootstrap_95ci": bootstrap_ci(seed_rates, rng),
            "confusion_by_seed": per_seed_confusion,
            "common_failure_counts": [len(x) for x in common_sets],
            "common_failure_pairwise_jaccard": jaccard,
            "oracle_call_pool_rates": oracle_rates,
            "oracle_call_pool_mean": float(np.mean(oracle_rates)),
            "oracle_call_pool_sample_std": float(np.std(oracle_rates, ddof=1)),
        }
    all_rates = [x["success_rates"] for x in report["sequences"].values()]
    report["weighted_by_frames"] = {
        "per_seed_success": [float(sum(report["sequences"][s]["success_counts"][i] for s in report["sequences"]) /
                                   sum(report["sequences"][s]["geometry_call_frames"] for s in report["sequences"]))
                             for i in range(3)]
    }
    report["weighted_by_frames"]["mean"] = float(np.mean(report["weighted_by_frames"]["per_seed_success"]))
    report["weighted_by_frames"]["sample_std"] = float(np.std(report["weighted_by_frames"]["per_seed_success"], ddof=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
