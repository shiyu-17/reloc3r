#!/usr/bin/env python3
"""GT-only evaluator for AVL-013 causal Reject/recovery traces."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")

def runs(bits):
    out=[]; cur=0
    for b in bits:
        cur=cur+1 if b else 0
        if b is False and cur==0: pass
        if not b and cur == 0: pass
    cur=0
    for b in bits:
        if b: cur += 1
        elif cur: out.append(cur); cur=0
    if cur: out.append(cur)
    return out

def evaluate(rows, gt):
    for r, g in zip(rows, gt):
        if r.get("pose") is None:
            r["success"] = False; r["translation_error_m"] = None; r["rotation_error_deg"] = None
        else:
            t, q = pose_error(np.asarray(r["pose"], dtype=np.float64), g)
            r["translation_error_m"], r["rotation_error_deg"] = float(t), float(q)
            r["success"] = bool(t <= .25 and q <= 5.0)
    accepted = [r for r in rows if r.get("pose") is not None]
    good = [r for r in accepted if r["success"]]
    rejected = [r for r in rows if r.get("pose") is None]
    failure_bits = [not r["success"] for r in rows]
    reject_bits = [r.get("pose") is None for r in rows]
    reject_runs = runs(reject_bits)
    failure_runs = runs(failure_bits)
    recovery_first = []
    recovery_correct = []
    recovery_attempts = 0
    right_censored_recovery_events = 0
    for i, r in enumerate(rows):
        if r.get("pose") is None and (i == 0 or rows[i-1].get("pose") is not None):
            recovery_attempts += 1
            j=i
            while j < len(rows) and rows[j].get("pose") is None: j += 1
            if j < len(rows):
                recovery_first.append(j-i)
                k=j
                while k < len(rows) and not rows[k]["success"]: k += 1
                if k < len(rows): recovery_correct.append(k-i)
            else:
                right_censored_recovery_events += 1
    previous_accepted_wrong = False; wrong_after_wrong = 0; state_updates_wrong = 0
    for r in rows:
        if r.get("pose") is not None:
            if not r["success"]: state_updates_wrong += 1
            if previous_accepted_wrong and not r["success"]: wrong_after_wrong += 1
            previous_accepted_wrong = not r["success"]
    n=len(rows); ac=len(accepted)
    return {
        "frames": n, "accepted": ac, "rejected": n-ac, "correct_accepted": len(good),
        "coverage": ac/n if n else 0.0,
        "success_all_frames": len(good)/n if n else 0.0,
        "accepted_pose_accuracy": len(good)/ac if ac else None,
        "wrong_accepted": ac-len(good), "false_accept_rate": (ac-len(good))/ac if ac else None,
        "max_reject_streak": max(reject_runs, default=0), "mean_reject_streak": float(np.mean(reject_runs)) if reject_runs else 0.0,
        "reject_run_count": len(reject_runs), "reject_run_total": int(sum(reject_runs)),
        "max_failure_streak": max(failure_runs, default=0), "mean_failure_streak": float(np.mean(failure_runs)) if failure_runs else 0.0,
        "failure_run_count": len(failure_runs), "failure_run_total": int(sum(failure_runs)),
        "recovery_to_first_accepted_mean": float(np.mean(recovery_first)) if recovery_first else 0.0,
        "recovery_to_first_correct_mean": float(np.mean(recovery_correct)) if recovery_correct else 0.0,
        "recovery_events": len(recovery_first), "correct_recovery_events": len(recovery_correct),
        "recovery_attempts": recovery_attempts,
        "right_censored_recovery_events": right_censored_recovery_events,
        "state_updates_wrong": state_updates_wrong, "wrong_after_wrong_state": wrong_after_wrong,
        "action_counts": {a: sum(r.get("action") == a for r in rows) for a in ("accept_consensus","use_memory","call_geometry","reject")},
        "final_decision_counts": {
            "accept": sum(r.get("pose") is not None and r.get("pose_source") in ("consensus", "memory") for r in rows),
            "recovery": sum(r.get("pose") is not None and r.get("pose_source") in ("geometry_recovery", "consensus_after_geometry_reject", "memory_after_geometry_reject") for r in rows),
            "reject": sum(r.get("pose") is None for r in rows),
        },
        "source_counts": {str(a): sum(r.get("pose_source") == a for r in rows) for a in ("consensus","memory","geometry_recovery","consensus_after_geometry_reject","memory_after_geometry_reject","reject","reject_geometry","reject_risk")},
        "geometry_call_rate": sum(r.get("action") == "call_geometry" for r in rows)/n if n else 0.0,
    }

def aggregate_reports(parts):
    """Aggregate per-sequence reports without creating cross-sequence runs."""
    parts = [p for p in parts if p is not None]
    n = sum(p["frames"] for p in parts)
    accepted = sum(p["accepted"] for p in parts)
    good = sum(p["correct_accepted"] for p in parts)
    def weighted_by(parts, count_key, mean_key):
        den = sum(p[count_key] for p in parts)
        return (sum(p[count_key] * p[mean_key] for p in parts) / den) if den else 0.0
    out = {
        "frames": n, "accepted": accepted, "rejected": n - accepted,
        "coverage": accepted / n if n else 0.0,
        "success_all_frames": good / n if n else 0.0,
        "accepted_pose_accuracy": good / accepted if accepted else None,
        "wrong_accepted": accepted - good,
        "false_accept_rate": (accepted - good) / accepted if accepted else None,
        "max_reject_streak": max((p["max_reject_streak"] for p in parts), default=0),
        "mean_reject_streak": weighted_by(parts, "reject_run_count", "mean_reject_streak"),
        "reject_run_count": sum(p["reject_run_count"] for p in parts),
        "reject_run_total": sum(p["reject_run_total"] for p in parts),
        "max_failure_streak": max((p["max_failure_streak"] for p in parts), default=0),
        "mean_failure_streak": weighted_by(parts, "failure_run_count", "mean_failure_streak"),
        "failure_run_count": sum(p["failure_run_count"] for p in parts),
        "failure_run_total": sum(p["failure_run_total"] for p in parts),
        "recovery_to_first_accepted_mean": weighted_by(parts, "recovery_events", "recovery_to_first_accepted_mean"),
        "recovery_to_first_correct_mean": weighted_by(parts, "correct_recovery_events", "recovery_to_first_correct_mean"),
        "recovery_events": sum(p["recovery_events"] for p in parts),
        "correct_recovery_events": sum(p["correct_recovery_events"] for p in parts),
        "recovery_attempts": sum(p["recovery_attempts"] for p in parts),
        "right_censored_recovery_events": sum(p["right_censored_recovery_events"] for p in parts),
        "state_updates_wrong": sum(p["state_updates_wrong"] for p in parts),
        "wrong_after_wrong_state": sum(p["wrong_after_wrong_state"] for p in parts),
        "geometry_call_rate": sum(p["geometry_call_rate"] * p["frames"] for p in parts) / n if n else 0.0,
    }
    for key in ("action_counts", "final_decision_counts", "source_counts"):
        keys = sorted({k for p in parts for k in p[key]})
        out[key] = {k: sum(p[key].get(k, 0) for p in parts) for k in keys}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path('data/lilocbench/prepared_1hz')); p.add_argument('--gt-root',type=Path,default=Path('data/lilocbench/gt')); p.add_argument('--trace-root',type=Path,default=Path('outputs/lilocbench/AVL-013/closed_loop_t1.5')); p.add_argument('--baseline-root',type=Path,default=Path('outputs/lilocbench/AVL-011/final_v1')); p.add_argument('--roma-root',type=Path,default=Path('outputs/lilocbench/AVL-004/always_geometry/roma')); p.add_argument('--output',type=Path,default=Path('outputs/lilocbench/AVL-013/evaluation.json')); p.add_argument('--max-gt-delta',type=float,default=.1)
    args=p.parse_args(); report={'experiment':'AVL-013','threshold':1.5,'validation':{},'test':{},'sequences':{},'baseline_no_reject':{},'audit':{'query_gt_read_in_trace':False,'uses_future_in_trace':False,'selection_uses_gt_in_trace':False}}
    split_reports={'validation':[],'test':[]}; split_base_reports={'validation':[],'test':[]}
    for seq in SEQUENCES:
        _,_,qt,_,qr=read_manifest(args.data_root,seq); gt,_=tum_camera_poses(args.gt_root/f'{seq}_gt_poses.txt',qt,base_to_front_rgb_optical(qr),args.max_gt_delta)
        rows=[json.loads(x) for x in (args.trace_root/seq/'closed_loop_trace.jsonl').read_text().splitlines() if x.strip()]
        base=[json.loads(x) for x in (args.baseline_root/seq/'closed_loop_trace.jsonl').read_text().splitlines() if x.strip()]
        for r in rows: report['audit']['query_gt_read_in_trace'] |= bool(r.get('query_gt_read')); report['audit']['uses_future_in_trace'] |= bool(r.get('uses_future')); report['audit']['selection_uses_gt_in_trace'] |= bool(r.get('selection_uses_gt'))
        cut=int(math.ceil(len(rows)*.4)); evaluate(rows,gt); evaluate(base,gt)
        report['sequences'][seq]=evaluate(rows,gt); report['baseline_no_reject'][seq]=evaluate(base,gt)
        report['validation'][seq]=evaluate(rows[:cut],gt[:cut]); report['test'][seq]=evaluate(rows[cut:],gt[cut:])
        split_reports['validation'].append(report['validation'][seq]); split_reports['test'].append(report['test'][seq])
        split_base_reports['validation'].append(evaluate(base[:cut], gt[:cut]))
        split_base_reports['test'].append(evaluate(base[cut:], gt[cut:]))
    report['validation_aggregate']=aggregate_reports(split_reports['validation'])
    report['test_aggregate']=aggregate_reports(split_reports['test'])
    report['baseline_validation_aggregate']=aggregate_reports(split_base_reports['validation'])
    report['baseline_test_aggregate']=aggregate_reports(split_base_reports['test'])
    # Costs are kept in separate buckets: replay wall time is real for this state machine;
    # cached neural elapsed_s is historical tool time, not a new end-to-end inference run.
    cost={'cost_complete_online_inference':False,'note':'AVL-013 state machine replay uses frozen AVL-002/004 tool outputs; neural times are reported as historical measured cache elapsed, not claimed as new online latency.','sequences':{}}
    for seq in SEQUENCES:
        s=json.loads((args.trace_root/seq/'summary.json').read_text()); rows=[json.loads(x) for x in (args.trace_root/seq/'closed_loop_trace.jsonl').read_text().splitlines() if x.strip()]
        roma={int(json.loads(x)['query_index']):json.loads(x) for x in (args.roma_root/f'{seq}_top100_merged.jsonl').read_text().splitlines() if x.strip()}
        geo=sum(float(roma[r['query_index']].get('elapsed_s',0.0)) for r in rows if r['action']=='call_geometry' and r['query_index'] in roma)
        cost['sequences'][seq]={'state_machine_replay_wall_s':s.get('sequence_wall_s'),'candidate_processing_s':s.get('stage_wall_s',{}).get('candidate_and_selection'),'verification_risk_s':s.get('stage_wall_s',{}).get('verification_and_risk'),'state_update_s':s.get('stage_wall_s',{}).get('state_update'),'historical_cached_geometry_elapsed_s':geo,'geometry_calls':sum(r['action']=='call_geometry' for r in rows)}
    cost['sum_state_machine_replay_wall_s']=sum(v['state_machine_replay_wall_s'] or 0 for v in cost['sequences'].values()); cost['sum_historical_cached_geometry_elapsed_s']=sum(v['historical_cached_geometry_elapsed_s'] for v in cost['sequences'].values())
    report['cost']=cost; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); (args.output.parent/'cost_ledger.json').write_text(json.dumps(cost,indent=2)+'\n')
    print(json.dumps({'validation':report['validation_aggregate'],'test':report['test_aggregate'],'baseline_test':report['baseline_test_aggregate'],'cost':cost,'audit':report['audit']},indent=2))
if __name__=='__main__': main()
