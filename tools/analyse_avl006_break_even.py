#!/usr/bin/env python3
"""Compute an auditable AVL-006-A cost break-even table from AVL-005 timing."""
import json
import math
from pathlib import Path

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]


def main():
    cost_path = Path("outputs/lilocbench/AVL-005/unified_cost/summary.json")
    reloc_path = Path("outputs/lilocbench/AVL-005/unified_cost/reloc3r_timing.json")
    context_path = Path("outputs/lilocbench/AVL-006/early_router/context_feature_timing.json")
    eval_path = Path("outputs/lilocbench/AVL-006/early_router_eval.json")
    output = Path("outputs/lilocbench/AVL-006/break_even.json")
    cost = json.loads(cost_path.read_text())
    reloc = json.loads(reloc_path.read_text())
    context = json.loads(context_path.read_text())
    early_eval = json.loads(eval_path.read_text())
    comp = cost["comparison"]
    n = int(sum(cost["sequences"][s]["frames"] for s in SEQUENCES))
    shared = float(comp["retrieval_total_s"])
    context_s = float(context["total_s"])
    # The following three terms were measured on the AVL-005 closed-loop trace
    # for every frame.  We report them separately because a minimal router may
    # choose not to execute all of this state bookkeeping.
    state_s = float(comp["manager"].get("memory_s", 0.0) + comp["manager"].get("candidate_processing_s", 0.0) + comp["manager"].get("verification_s", 0.0))
    reloc_per_seq = {}
    roma_per_seq = {}
    for s in SEQUENCES:
        frames = int(cost["sequences"][s]["frames"])
        rt = reloc["sequences"][s]
        reloc_per_seq[s] = float(rt["mean_preprocess_s"] + rt["mean_inference_s"])
        h = cost["sequences"][s]["hloc"]
        r = cost["sequences"][s]["roma"]
        roma_per_seq[s] = float(h["query_extract_s"]["mean_s"] + h["rerank_s"]["mean_s"] + r["elapsed_s"]["mean_s"])
    weighted_reloc = sum(cost["sequences"][s]["frames"] * reloc_per_seq[s] for s in SEQUENCES)
    weighted_roma = sum(cost["sequences"][s]["frames"] * roma_per_seq[s] for s in SEQUENCES)
    c_reloc = weighted_reloc / n
    c_roma = weighted_roma / n
    baseline = float(comp["always_roma"]["total_s"])
    # A route cost includes shared retrieval, cheap current/previous RGB
    # features, measured state bookkeeping, and the selected per-frame path.
    route_rows = {}
    for policy, result in early_eval["policies"].items():
        call_counts = result["per_sequence"]
        path_s = 0.0
        for s in SEQUENCES:
            g = int(call_counts[s]["roma_calls"])
            f = int(call_counts[s]["frames"])
            path_s += g * roma_per_seq[s] + (f - g) * reloc_per_seq[s]
        route_cost = shared + context_s + state_s + path_s
        # The no-state variant makes the sensitivity to the chosen memory
        # accounting explicit; it is not the primary cost claim.
        route_cost_no_state = shared + context_s + path_s
        route_rows[policy] = {
            "roma_calls": int(result["roma_calls"]),
            "roma_call_rate": float(result["roma_call_rate"]),
            "success": float(result["success"]),
            "success_count": int(result["success_count"]),
            "route_cost_with_state_s": route_cost,
            "route_cost_without_state_s": route_cost_no_state,
            "delta_vs_always_roma_with_state_s": route_cost - baseline,
            "strict_budget_margin_s": baseline - route_cost,
        }
    # Break-even is expressed for a hypothetical number of RoMa calls while
    # retaining all N frames and the same measured shared/state overhead.
    fixed = shared + context_s + state_s + n * c_reloc
    delta = c_roma - c_reloc
    budget = baseline
    max_calls = math.floor((budget - fixed) / delta) if delta > 0 else n
    # Strictly below baseline requires one fewer call when the quotient is exact.
    while fixed + max_calls * delta >= budget and max_calls >= 0:
        max_calls -= 1
    min_calls_to_match = math.ceil((baseline - fixed) / delta) if delta > 0 else 0
    # Post-hoc oracle for the two available actions.  This is an upper bound,
    # not an online route: it reads both outcomes only here, after routing.
    union_success = 0
    memory_success_total = 0
    memory_only = roma_only = both_success = neither = 0
    oracle_union_calls = 0
    oracle_union_path_s = 0.0
    oracle_min_calls = 0
    oracle_min_path_s = 0.0
    for s in SEQUENCES:
        mem_path = Path(f"outputs/lilocbench/AVL-003/{s}/frame_results.jsonl")
        roma_path = Path(f"outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0/{s}.json")
        mem = [bool(json.loads(line)["methods"]["memory3"].get("success_0.25m_5deg", False)) for line in mem_path.read_text().splitlines()]
        roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False)) for x in json.loads(roma_path.read_text())["frames"]]
        for m, r in zip(mem, roma):
            union_success += m or r
            memory_success_total += m
            both_success += m and r
            memory_only += m and not r
            roma_only += r and not m
            neither += not m and not r
        # Maximum-call accuracy oracle: use memory only on memory-only frames,
        # and RoMa everywhere else.  This preserves the union success while
        # maximizing calls without sacrificing any of the 54 memory-only wins.
        mem_only_seq = sum(m and not r for m, r in zip(mem, roma))
        oracle_union_calls += len(mem) - mem_only_seq
        oracle_union_path_s += (len(mem) - mem_only_seq) * roma_per_seq[s] + mem_only_seq * reloc_per_seq[s]
        # Minimum-call upper-bound route: memory everywhere, then call RoMa on
        # Roma-only frames until the strict Always-RoMa count is exceeded.
        roma_only_seq = sum(r and not m for m, r in zip(mem, roma))
        oracle_min_calls += roma_only_seq
        # The minimal path cost is not additive across folds until the global
        # number of needed Roma-only wins is known; store this count below.
    best_frozen_policy = max(int(result["success_count"]) for result in early_eval["policies"].values())
    target_success_count = 1552 + 1
    minimum_calls_to_exceed = max(0, target_success_count - int(memory_success_total))
    # For the minimum-call route use the cheapest available Roma-only frames;
    # this is an optimistic cost lower bound, not an online policy.
    roma_only_costs = []
    for s in SEQUENCES:
        mem_path = Path(f"outputs/lilocbench/AVL-003/{s}/frame_results.jsonl")
        roma_path = Path(f"outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0/{s}.json")
        mem = [bool(json.loads(line)["methods"]["memory3"].get("success_0.25m_5deg", False)) for line in mem_path.read_text().splitlines()]
        roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False)) for x in json.loads(roma_path.read_text())["frames"]]
        for m, r in zip(mem, roma):
            if r and not m:
                roma_only_costs.append((roma_per_seq[s] - reloc_per_seq[s], s))
    roma_only_costs.sort()
    selected_deltas = roma_only_costs[:minimum_calls_to_exceed]
    # Base memory-everywhere cost plus replacing selected frames with RoMa.
    memory_base_path = sum(cost["sequences"][s]["frames"] * reloc_per_seq[s] for s in SEQUENCES)
    oracle_min_path_s = memory_base_path + sum(x[0] for x in selected_deltas)
    oracle_min_calls = minimum_calls_to_exceed
    oracle_union_cost = shared + context_s + state_s + oracle_union_path_s
    oracle_min_cost = shared + context_s + state_s + oracle_min_path_s
    # Accuracy-constrained maximum-call route under the strict budget: keep
    # only enough memory-only frames as memory to gain > Always-RoMa success,
    # and run RoMa on every other frame.  Select the skipped memory-only frames
    # with largest measured path-cost saving; this is an optimistic post-hoc
    # upper-bound, not a deployable router.
    memory_only_items = []
    all_roma_path_s = 0.0
    for s in SEQUENCES:
        mem_path = Path(f"outputs/lilocbench/AVL-003/{s}/frame_results.jsonl")
        roma_path = Path(f"outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0/{s}.json")
        mem = [bool(json.loads(line)["methods"]["memory3"].get("success_0.25m_5deg", False)) for line in mem_path.read_text().splitlines()]
        roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False)) for x in json.loads(roma_path.read_text())["frames"]]
        all_roma_path_s += len(mem) * roma_per_seq[s]
        if any(m and not r for m, r in zip(mem, roma)):
            memory_only_items.extend([(roma_per_seq[s] - reloc_per_seq[s], s) for m, r in zip(mem, roma) if m and not r])
    budget_max_calls = max_calls
    memory_frames_to_keep = max(0, n - budget_max_calls)
    skipped_for_budget = sorted(memory_only_items, reverse=True)[:memory_frames_to_keep]
    accuracy_max_call_path = all_roma_path_s - sum(x[0] for x in skipped_for_budget)
    accuracy_max_call_cost = shared + context_s + state_s + accuracy_max_call_path
    accuracy_max_call_success = 1552 + len(skipped_for_budget)
    out = {
        "protocol": {
            "query_gt_read": False,
            "timing_sources": [str(cost_path), str(reloc_path), str(context_path)],
            "route_cost_formula": "retrieval + context + state + sum(nonroma*Reloc3r + roma*(query_extract+rerank+RoMa))",
            "reloc3r_note": "per-sequence mean of 8 fixed temporal samples, scaled by frame count",
            "state_note": "memory + candidate processing + verification from AVL-005 closed-loop trace; sensitivity variant without state is also reported",
        },
        "frames": n,
        "shared_retrieval_s": shared,
        "context_feature_s": context_s,
        "state_overhead_s": state_s,
        "per_sequence_reloc3r_s_per_frame": reloc_per_seq,
        "per_sequence_roma_path_s_per_frame": roma_per_seq,
        "weighted_reloc3r_s_per_frame": c_reloc,
        "weighted_roma_path_s_per_frame": c_roma,
        "always_roma": {
            "success": 1552 / n,
            "success_count": 1552,
            "cost_s": baseline,
        },
        "break_even": {
            "budget_s_strict": baseline,
            "fixed_cost_at_zero_roma_s": fixed,
            "delta_roma_minus_reloc_s_per_call": delta,
            "max_roma_calls_for_strictly_below_always_roma": max_calls,
            "max_roma_call_rate_for_strictly_below_always_roma": max_calls / n,
            "minimum_roma_calls_to_match_budget_ignoring_strictness": min_calls_to_match,
            "maximum_manager_overhead_s_at_zero_extra_calls": baseline - (shared + n * c_roma),
        },
        "early_router_policies": route_rows,
        "two_action_empirical_upper_bound": {
            "cached_always_roma_success_count": 1552,
            "best_frozen_early_policy_success_count": best_frozen_policy,
            "posthoc_memory3_or_roma_union_count": int(union_success),
            "posthoc_memory3_or_roma_union_pct": union_success / n,
            "confusion": {"both": int(both_success), "memory_only": int(memory_only), "roma_only": int(roma_only), "neither": int(neither)},
            "strictly_exceeds_91_51_percent_possible_in_posthoc_union": union_success > 1552,
            "online_claim": "union is an upper bound only; the pre-router did not read either outcome",
            "accuracy_constrained_oracle_routes": {
                "strict_target_success_count": target_success_count,
                "memory_only_count": int(memory_only),
                "minimum_roma_calls_to_exceed_always_roma": int(minimum_calls_to_exceed),
                "minimum_call_route_is_posthoc_only": True,
                "minimum_call_route_cost_lower_bound_s": oracle_min_cost,
                "maximum_roma_calls_while_exceeding_and_below_budget": int(budget_max_calls),
                "maximum_call_route_success_count": int(accuracy_max_call_success),
                "maximum_call_route_cost_s": accuracy_max_call_cost,
                "maximum_call_route_extra_manager_overhead_budget_s": baseline - accuracy_max_call_cost,
                "budget_feasible": accuracy_max_call_cost < baseline,
                "maximum_call_route_selection": "posthoc: keep memory on the highest-saving memory-only frames; call RoMa elsewhere",
                "union_preserving_route_calls": int(oracle_union_calls),
                "union_preserving_route_success_count": int(union_success),
                "union_preserving_route_cost_s": oracle_union_cost,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"break_even": out["break_even"], "policies": route_rows}, indent=2))


if __name__ == "__main__":
    main()
