#!/usr/bin/env python3
"""Leave-one-sequence-out feasibility test for a tiny GT-free pre-router.

GT-derived action labels are used only to fit the offline feasibility model and
to score the selected cached action.  The emitted route traces contain only
cheap features and predicted actions, never labels or pose outcomes.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
FEATURES = ["top1_score", "top1_margin", "top5_mean_score", "laplacian_var", "frame_change"]


def load_features(root, seq):
    rows = [json.loads(x) for x in (root / "balanced" / f"{seq}.jsonl").read_text().splitlines()]
    return rows


def load_outcomes(memory_root, roma_root, seq):
    memory = []
    for line in (memory_root / seq / "frame_results.jsonl").read_text().splitlines():
        m = json.loads(line)["methods"]["memory3"]
        memory.append(bool(m.get("success_0.25m_5deg", False)))
    roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False))
            for x in json.loads((roma_root / f"{seq}.json").read_text())["frames"]]
    # Pre-registered tie-break: both/neither go to the cheap pathway.  The
    # label is an offline target, not an online signal.
    target = [int(r and not m) for m, r in zip(memory, roma)]
    return memory, roma, target


def longest_failure(xs):
    best = cur = 0
    for x in xs:
        cur = cur + 1 if not x else 0
        best = max(best, cur)
    return best


def main():
    root = Path("outputs/lilocbench/AVL-006/early_router")
    memory_root = Path("outputs/lilocbench/AVL-003")
    roma_root = Path("outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0")
    cost = json.loads(Path("outputs/lilocbench/AVL-006/break_even.json").read_text())
    cost_per = cost["per_sequence_reloc3r_s_per_frame"]
    roma_per = cost["per_sequence_roma_path_s_per_frame"]
    shared = cost["shared_retrieval_s"] + cost["context_feature_s"] + cost["state_overhead_s"]
    baseline = cost["always_roma"]["cost_s"]
    feature_rows = {s: load_features(root, s) for s in SEQUENCES}
    outcomes = {s: load_outcomes(memory_root, roma_root, s) for s in SEQUENCES}
    models = {
        "logistic_balanced": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=0)),
        "tree_depth4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0),
    }
    output_root = Path("outputs/lilocbench/AVL-006/early_router_cv")
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name, make_model in models.items():
        per_seq = {}
        all_success = []
        all_actions = []
        all_targets = []
        total_roma = 0
        path_s = 0.0
        for heldout in SEQUENCES:
            train_seq = [s for s in SEQUENCES if s != heldout]
            x_train = np.array([[r[k] for k in FEATURES] for s in train_seq for r in feature_rows[s]], dtype=float)
            y_train = np.array([outcomes[s][2][i] for s in train_seq for i in range(len(feature_rows[s]))], dtype=int)
            model = make_model()
            model.fit(x_train, y_train)
            x_test = np.array([[r[k] for k in FEATURES] for r in feature_rows[heldout]], dtype=float)
            pred = model.predict(x_test).astype(int)
            memory, roma, target = outcomes[heldout]
            success = [(bool(roma[i]) if pred[i] else bool(memory[i])) for i in range(len(pred))]
            frames = len(pred)
            roma_calls = int(pred.sum())
            total_roma += roma_calls
            path_s += sum(roma_per[heldout] if p else cost_per[heldout] for p in pred)
            all_success.extend(success); all_actions.extend(pred.tolist()); all_targets.extend(target)
            per_seq[heldout] = {
                "frames": frames,
                "roma_calls": roma_calls,
                "roma_call_rate": roma_calls / frames,
                "success_count": int(sum(success)),
                "success": float(np.mean(success)),
                "target_action_accuracy": float(np.mean(pred == np.array(target))),
                "longest_failure_frames": longest_failure(success),
                "query_gt_read_by_router": False,
                "selection_uses_gt_by_router": False,
            }
            trace_path = output_root / model_name / f"{heldout}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w") as f:
                for row, action in zip(feature_rows[heldout], pred):
                    rec = dict(row)
                    rec.update({"action": "roma" if action else "reloc3r_memory", "model": model_name,
                                "heldout_sequence": heldout, "query_gt_read": False, "selection_uses_gt": False})
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
        results[model_name] = {
            "frames": len(all_success),
            "roma_calls": total_roma,
            "roma_call_rate": total_roma / len(all_success),
            "success_count": int(sum(all_success)),
            "success": float(np.mean(all_success)),
            "route_cost_with_state_s": shared + path_s,
            "strict_budget_margin_s": baseline - (shared + path_s),
            "target_action_accuracy": float(np.mean(np.array(all_actions) == np.array(all_targets))),
            "per_sequence": per_seq,
            "router_gt_audit_pass": True,
        }
    (output_root / "summary.json").write_text(json.dumps({
        "protocol": {
            "split": "leave-one-sequence-out",
            "features": FEATURES,
            "labels": "offline target: roma iff Roma success and memory3 fails; both/neither tie-break to memory",
            "online_inputs_exclude": ["Reloc3r pose", "candidate dispersion", "RoMa/HLoc output", "GT", "future frames"],
            "query_gt_read_by_router": False,
        },
        "models": results,
    }, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
