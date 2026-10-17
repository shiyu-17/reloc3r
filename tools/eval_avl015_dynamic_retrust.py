#!/usr/bin/env python3
"""AVL-015 dynamic re-trust replay on frozen causal candidates.

This evaluator applies source-specific thresholds to every existing AVL-013
v3 candidate frame and maintains a new reject streak. It does not regenerate
a pose or route; its purpose is to measure whether stale-state evidence should
be discounted after a reject run without hiding the sparse-candidate limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_avl015_source_verifiers import direct_score, source_name


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = {"consensus": 0.75, "memory": 1.0, "geometry": 0.5}


def state_trust(anchor_age, reject_streak):
    """Causal trust in the previous global anchor before current evidence."""
    return math.exp(-float(anchor_age) / 20.0 - float(reject_streak) / 3.0)


def relaxed_score(e, anchor_age, reject_streak):
    score = direct_score(e)
    trust = state_trust(anchor_age, reject_streak)
    if source_name(e) in ("consensus", "memory"):
        return max(score, float(e.get("temporal_risk", 0.0)))
    src = source_name(e)
    if src == "geometry":
        # Only the state-dependent component is decayed.  New AVL-012 traces
        # expose disjoint risk terms; the fallback keeps old traces usable but
        # is explicitly marked as legacy because their combined geometry_risk
        # cannot support a valid re-trust experiment.
        intrinsic = e.get("geometry_intrinsic_risk")
        cross_source = e.get("geometry_cross_source_risk")
        previous = e.get("geometry_previous_state_risk")
        if intrinsic is None or cross_source is None or previous is None:
            raise ValueError(
                "dynamic re-trust requires componentized geometry risks; "
                "rerun tools/run_avl012_reliability.py first")
        return max(float(intrinsic), float(cross_source), trust * float(previous))
    return score


def summarize(rows):
    accepted = [r for r in rows if r["verifier_accept"]]
    good = sum(r["success"] for r in accepted)
    wrong = len(accepted) - good
    reject_runs = []
    recovery_times = []
    current_reject = 0
    for r in rows:
        if not r["verifier_accept"]:
            current_reject += 1
        else:
            if current_reject:
                reject_runs.append(current_reject)
                if r["success"]:
                    recovery_times.append(current_reject)
            current_reject = 0
    if current_reject:
        reject_runs.append(current_reject)
    return {"frames": len(rows), "accepted": len(accepted),
            "coverage": len(accepted) / len(rows) if rows else 0.0,
            "accepted_correct": good,
            "accepted_accuracy": good / len(accepted) if accepted else 0.0,
            "false_accept_rate": wrong / len(accepted) if accepted else 0.0,
            "full_frame_success": good / len(rows) if rows else 0.0,
            "mean_recovery_time_frames": (sum(recovery_times) / len(recovery_times)
                                           if recovery_times else None),
            "max_reject_length": max(reject_runs) if reject_runs else 0,
            "recovered_reject_segments": len(recovery_times),
            "right_censored_reject_segments": len(reject_runs) - len(recovery_times),
            "stale_frames": sum(r["stale_state"] for r in rows),
            "accepted_stale": sum(r["verifier_accept"] and r["stale_state"] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/lilocbench/AVL-014/AVL-012_selective"))
    p.add_argument("--trace-root", type=Path, default=Path("outputs/lilocbench/AVL-014/AVL-013_v3"))
    p.add_argument("--output", type=Path, default=Path("outputs/lilocbench/AVL-015/dynamic_retrust/evaluation.json"))
    args = p.parse_args()
    report = {"experiment": "AVL-015", "component": "Q2_dynamic_retrust_replay",
              "protocol": {"thresholds": THRESHOLDS, "stale_after_reject_streak": 3,
                            "state_trust": "exp(-anchor_age/20 - reject_streak/3); previous-state disagreement is weighted by this trust",
                            "route_regeneration": False, "gt_used_only_for_evaluation": True,
                            "requires_componentized_geometry_risk": True,
                            "relaxation": "stale state no longer contributes previous-pose/temporal veto; current intrinsic and cross-source evidence remain"},
              "sequences": {}, "aggregate": {},
              "audit": {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False}}
    all_rows = {"all": [], "validation": [], "test": []}
    for seq in SEQUENCES:
        ev = {int(json.loads(l)["query_index"]): json.loads(l)
              for l in (args.root / seq / "evidence.jsonl").read_text().splitlines() if l.strip()}
        tr = {int(json.loads(l)["query_index"]): json.loads(l)
              for l in (args.trace_root / seq / "closed_loop_trace.jsonl").read_text().splitlines() if l.strip()}
        evaluator = {int(json.loads(l)["query_index"]): json.loads(l)
                     for l in (args.root / seq / "evaluator_rows.jsonl").read_text().splitlines() if l.strip()}
        rows = []
        reject_streak = 0
        last_anchor = -1
        for qi in sorted(ev):
            e = ev[qi]; raw = tr[qi]; src = source_name(e)
            candidate_available = e.get("route_pose") is not None
            anchor_age = qi - last_anchor if last_anchor >= 0 else qi + 1
            trust = state_trust(anchor_age, reject_streak)
            stale = reject_streak >= 3 or anchor_age >= 10
            score = relaxed_score(e, anchor_age, reject_streak)
            row = {"sequence": seq, "query_index": qi, "source": src,
                   "score": score, "threshold": THRESHOLDS.get(src, None),
                   "stale_state": stale, "reject_streak": reject_streak,
                   "anchor_age": anchor_age, "state_trust": trust,
                   "motion_uncertainty": float(e.get("temporal_risk", 0.0)),
                   "verifier_accept": bool(src is not None and candidate_available and score <= THRESHOLDS[src]),
                   "success": bool(evaluator[qi].get("success", False))}
            # The source-specific score is evidence-only; the success bit is
            # joined from an independent evaluator and never drives decisions.
            row["candidate_available"] = candidate_available
            rows.append(row)
            report["audit"]["query_gt_read"] |= bool(raw.get("query_gt_read", False))
            report["audit"]["uses_future"] |= bool(raw.get("uses_future", False))
            report["audit"]["selection_uses_gt"] |= bool(raw.get("selection_uses_gt", False))
            if row["verifier_accept"]:
                reject_streak = 0
                last_anchor = qi
            else:
                reject_streak += 1
            row["reject_streak_after"] = reject_streak
        rows.sort(key=lambda x: x["query_index"]); cut = int(math.ceil(len(rows) * 0.4))
        # Source rows are sparse; split by original query position.
        report["sequences"][seq] = {"all": summarize(rows),
                                     "validation": summarize([r for r in rows if r["query_index"] < cut]),
                                     "test": summarize([r for r in rows if r["query_index"] >= cut])}
        all_rows["all"].extend(rows); all_rows["validation"].extend([r for r in rows if r["query_index"] < cut]); all_rows["test"].extend([r for r in rows if r["query_index"] >= cut])
    for split, rows in all_rows.items(): report["aggregate"][split] = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "audit": report["audit"]}, indent=2))


if __name__ == "__main__":
    main()
