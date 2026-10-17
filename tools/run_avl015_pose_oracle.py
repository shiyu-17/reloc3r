#!/usr/bin/env python3
"""AVL-015 Q1: pose-action complementarity and cost-aware oracle.

This is a post-hoc evaluator only. It reads frozen action outcomes and timing
ledgers, never drives an online route. Historical-keyframe outcomes are joined
from the independently saved causal Q3 trace and scored here with query GT.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import (base_to_front_rgb_optical,
                                           pose_error, read_manifest,
                                           tum_camera_poses)


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
ACTIONS = ("consensus", "memory", "hloc", "roma", "historical")
GEOMETRY_ACTIONS = {"hloc", "roma", "historical"}


def read_json(path: Path):
    return json.loads(path.read_text())


def load_costs(cost_path: Path, break_even_path: Path):
    """Return per-sequence action costs in seconds.

    Reloc3r is the registered AVL-006 estimate.  Memory adds the registered
    AVL-005 state/memory proxy.  HLoc/RoMa costs are their full route elapsed
    times, including their reference reranking path.  Shared retrieval/context
    is charged once in the aggregate, not once per action.
    """
    cost = read_json(cost_path)
    be = read_json(break_even_path)
    out = {}
    for seq in SEQUENCES:
        info = cost["sequences"][seq]
        n = int(info["frames"])
        reloc = float(be["per_sequence_reloc3r_s_per_frame"][seq])
        memory = float(info["manager"]["memory"]["sum_s"]) / n
        hloc = float(info["hloc"]["elapsed_s"]["sum_s"]) / n
        # RoMa's matching/triangulation/PnP row does not include the shared
        # top-100 query extraction and LightGlue reranking stages.  Those
        # stages are required by the registered ``rerank + RoMa`` action and
        # are charged here once per frame.
        roma = (float(info["hloc"]["query_extract_s"]["sum_s"])
                + float(info["hloc"]["rerank_s"]["sum_s"])
                + float(info["roma"]["elapsed_s"]["sum_s"])) / n
        out[seq] = {
            "consensus": reloc,
            "memory": reloc + memory,
            "hloc": hloc,
            "roma": roma,
        }
    # Q1 compares pose-producing actions, so only retrieval is shared by all
    # actions.  Context extraction and state-update overhead belong to a
    # concrete Manager implementation and are reported separately in the
    # registered AVL-005/006 ledgers; charging them here would mix two cost
    # boundaries and make Always-RoMa incomparable.
    shared = float(be["shared_retrieval_s"])
    return out, shared


def load_historical(seq, trace_root=Path("outputs/lilocbench/AVL-015/historical_keyframe_memory")):
    """Return causal historical success bits and measured per-frame costs."""
    _, _, times, _, seq_root = read_manifest(Path("data/lilocbench/prepared_1hz"), seq)
    gt, _ = tum_camera_poses(Path("data/lilocbench/gt") / f"{seq}_gt_poses.txt", times,
                              base_to_front_rgb_optical(seq_root), 0.1)
    rows = {}
    for line in (trace_root / seq / "trace.jsonl").read_text().splitlines():
        raw = json.loads(line)
        qi = int(raw["query_index"])
        pose = raw.get("pose")
        ok = False
        if pose is not None:
            t, r = pose_error(np.asarray(pose, dtype=np.float64), gt[qi])
            ok = bool(t <= 0.25 and r <= 5.0)
        rows[qi] = {"success": ok, "elapsed_s": float(raw.get("elapsed_s", 0.0)),
                    "available": pose is not None,
                    "causal": raw.get("causal", {})}
    return rows


def pair_stats(rows, a, b):
    both = a_only = b_only = neither = 0
    for row in rows:
        x, y = bool(row["success"][a]), bool(row["success"][b])
        if x and y:
            both += 1
        elif x:
            a_only += 1
        elif y:
            b_only += 1
        else:
            neither += 1
    return {"both": both, f"{a}_only": a_only, f"{b}_only": b_only, "neither": neither}


def summarize_rows(rows):
    n = len(rows)
    counts = {a: sum(bool(r["success"][a]) for r in rows) for a in ACTIONS}
    counts["any"] = sum(any(bool(r["success"][a]) for a in ACTIONS) for r in rows)
    unique = {
        a: sum(bool(r["success"][a]) and not any(bool(r["success"][b]) for b in ACTIONS if b != a)
               for r in rows)
        for a in ACTIONS
    }
    return {
        "frames": n,
        "success_counts": counts,
        "success_rates": {k: v / n if n else 0.0 for k, v in counts.items()},
        "unique_action_success": unique,
        "pairwise": {f"{a}__{b}": pair_stats(rows, a, b)
                     for i, a in enumerate(ACTIONS) for b in ACTIONS[i + 1:]},
    }


def fixed_frontier(summary, costs):
    """Return fixed-action Accuracy–Geometry-calls–Latency points and Pareto set."""
    n = int(summary["frames"])
    points = []
    for action in ACTIONS:
        calls = n if action in GEOMETRY_ACTIONS else 0
        points.append({"action": action, "success_rate": summary["success_rates"][action],
                      "geometry_calls": calls, "geometry_call_rate": calls / n if n else 0.0,
                      "latency_s": float(costs[action])})
    pareto = []
    for p in points:
        dominated = any(q["success_rate"] >= p["success_rate"]
                        and q["geometry_calls"] <= p["geometry_calls"]
                        and q["latency_s"] <= p["latency_s"]
                        and (q["success_rate"] > p["success_rate"]
                             or q["geometry_calls"] < p["geometry_calls"]
                             or q["latency_s"] < p["latency_s"])
                        for q in points)
        if not dominated:
            pareto.append(p)
    return {"points": points, "pareto": pareto}


def fixed_cost(rows, seq, action_costs, shared_for_sequence=0.0):
    return float(shared_for_sequence + len(rows) * action_costs[seq][rows[0]["_action_cost_key"]])


def min_cost_curve(rows, seq, action_costs, shared_total=0.0):
    """Minimum total cost for exactly k successful frames.

    Each frame can choose one of the frozen actions.  The result is an
    oracle lower bound, not an implementable route: the success bit comes
    from the post-hoc evaluator and is never visible to an online policy.
    """
    base = 0.0
    deltas = []
    forced_success = 0
    for row in rows:
        costs = action_costs[seq]
        failing = [costs[a] for a in ACTIONS if not row["success"][a]]
        succeeding = [costs[a] for a in ACTIONS if row["success"][a]]
        if not failing:
            # Every action succeeds. Pick the cheapest one; a success is
            # unavoidable for this frame.
            base += min(succeeding)
            forced_success += 1
            continue
        fail_cost = min(failing)
        base += fail_cost
        if succeeding:
            deltas.append(min(succeeding) - fail_cost)
    deltas.sort()
    # Exact-k costs can decrease when a successful action is cheaper than all
    # failing actions.  The reported frontier is therefore the minimum cost
    # for *at least* k successes, obtained by a suffix minimum.
    exact = [None] * (len(rows) + 1)
    current = base + shared_total
    for i in range(len(deltas) + 1):
        exact[forced_success + i] = current
        if i < len(deltas):
            current += deltas[i]
    best = [None] * len(exact)
    running = None
    for i in range(len(exact) - 1, -1, -1):
        if exact[i] is not None:
            running = exact[i] if running is None else min(running, exact[i])
        best[i] = running
    return [{"success_count": k, "success_rate": k / len(rows), "min_cost_s": best[k]}
            for k in range(len(best)) if best[k] is not None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", type=Path,
                   default=Path("outputs/lilocbench/AVL-004/action_oracle/summary_always.json"))
    p.add_argument("--cost", type=Path,
                   default=Path("outputs/lilocbench/AVL-005/unified_cost/summary.json"))
    p.add_argument("--break-even", type=Path,
                   default=Path("outputs/lilocbench/AVL-006/break_even.json"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/lilocbench/AVL-015/pose_oracle/summary.json"))
    args = p.parse_args()

    oracle = read_json(args.oracle)
    action_costs, shared = load_costs(args.cost, args.break_even)
    report = {
        "experiment": "AVL-015",
        "component": "Q1_pose_oracle",
        "protocol": {
            "type": "posthoc_gt_only_evaluator",
            "online_route_selection": False,
            "query_gt_read_by_online_routes": False,
                   "historical_keyframe_relocalization": "joined_from_causal_Q3_trace; GT only in this evaluator",
            "pose_actions": list(ACTIONS),
            "rerank_is_pose_action": False,
            "reject_is_pose_action": False,
            "split": "per-sequence first ceil(40%) validation, remainder test",
            "cost_shared_s": shared,
            "cost_note": "Reloc3r is AVL-006 registered estimate; Memory adds AVL-005 memory proxy; HLoc includes its recorded rerank+geometry route; RoMa includes top-100 query extraction+rerank+RoMa elapsed. Only shared retrieval is charged once; Manager context/state overhead remains a separate ledger.",
        },
        "cost_model_s_per_frame": action_costs,
        "sequences": {},
        "aggregate": {},
    }

    all_rows = {"all": [], "validation": [], "test": []}
    for seq in SEQUENCES:
        rows = [dict(r) for r in oracle["sequences"][seq]["rows"]]
        rows.sort(key=lambda x: int(x["query_index"]))
        historical = load_historical(seq)
        hist_cost = sum(x["elapsed_s"] for x in historical.values()) / len(rows)
        action_costs[seq]["historical"] = hist_cost
        for r in rows:
            h = historical[int(r["query_index"])]
            r["success"]["historical"] = bool(h["success"])
            r["historical_available"] = bool(h["available"])
            r["historical_causal"] = h["causal"]
        cut = int(math.ceil(len(rows) * 0.4))
        for r in rows:
            r["_action_cost_key"] = seq
        val, test = rows[:cut], rows[cut:]
        seq_report = {
            "frames": len(rows),
            "validation_frames": len(val),
            "test_frames": len(test),
            "all": summarize_rows(rows),
            "validation": summarize_rows(val),
            "test": summarize_rows(test),
            "cost_oracle": {
                "all": min_cost_curve(rows, seq, action_costs, shared_total=shared * len(rows) / 1696.0),
                "test": min_cost_curve(test, seq, action_costs, shared_total=shared * len(test) / 1696.0),
            },
            "fixed_action_cost_s_all": {
                a: float(shared * len(rows) / 1696.0 + len(rows) * action_costs[seq][a]) for a in ACTIONS
            },
        }
        seq_report["fixed_frontier_all"] = fixed_frontier(
            seq_report["all"], seq_report["fixed_action_cost_s_all"])
        report["sequences"][seq] = seq_report
        all_rows["all"].extend(rows)
        all_rows["validation"].extend(val)
        all_rows["test"].extend(test)

    for split, rows in all_rows.items():
        summary = summarize_rows(rows)
        summary["fixed_action_cost_s"] = {
            a: float(shared * len(rows) / 1696.0 + sum(action_costs[seq][a] for seq in SEQUENCES for _ in []))
            for a in ACTIONS
        }
        # Correct aggregate fixed costs by sequence, preserving frame weights.
        summary["fixed_action_cost_s"] = {
            a: float(shared * len(rows) / 1696.0 + sum(
                len([r for r in rows if r.get("_action_cost_key") == seq]) * action_costs[seq][a]
                for seq in SEQUENCES
            )) for a in ACTIONS
        }
        summary["fixed_frontier"] = fixed_frontier(summary, summary["fixed_action_cost_s"])
        summary["cost_oracle"] = None
        report["aggregate"][split] = summary

    # Aggregate cost-aware oracle with sequence-specific costs.
    for split, rows in all_rows.items():
        base = 0.0
        deltas = []
        forced = 0
        for row in rows:
            seq = row["_action_cost_key"]
            costs = action_costs[seq]
            failing = [costs[a] for a in ACTIONS if not row["success"][a]]
            succeeding = [costs[a] for a in ACTIONS if row["success"][a]]
            if not failing:
                base += min(succeeding); forced += 1
            else:
                fail = min(failing); base += fail
                if succeeding: deltas.append(min(succeeding) - fail)
        deltas.sort()
        exact = [None] * (len(rows) + 1)
        current = base + shared * len(rows) / 1696.0
        for i in range(len(deltas) + 1):
            exact[forced + i] = current
            if i < len(deltas):
                current += deltas[i]
        curve = [None] * len(exact)
        running = None
        for i in range(len(exact) - 1, -1, -1):
            if exact[i] is not None:
                running = exact[i] if running is None else min(running, exact[i])
            curve[i] = running
        curve = [{"success_count": k, "success_rate": k / len(rows), "min_cost_s": curve[k]}
                 for k in range(len(curve)) if curve[k] is not None]
        report["aggregate"][split]["cost_oracle"] = {
            "maximum_success_count": forced + len(deltas),
            "maximum_success_rate": (forced + len(deltas)) / len(rows),
            "min_cost_curve": curve,
        }

    # Remove private helper fields from persisted data.
    for seq in SEQUENCES:
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
