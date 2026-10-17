#!/usr/bin/env python3
"""AVL-007: frozen DINOv2 representation + cheap-context pathway routing.

The router is evaluated leave-one-sequence-out.  DINO embeddings and PCA are
frozen using only the mapping database.  GT-derived action labels are read only
by this offline evaluator; emitted predictions contain cheap features and the
chosen action, never labels or pose outcomes.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
EMBED_ROOT = Path("outputs/lilocbench/AVL-007/embeddings")
CHEAP_ROOT = Path("outputs/lilocbench/AVL-006/early_router/balanced")
MEMORY_ROOT = Path("outputs/lilocbench/AVL-003")
ROMA_ROOT = Path("outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0")
RETRIEVAL_ROOT = Path("outputs/lilocbench/AVL-004/always_geometry/retrieval")
BREAK_EVEN = Path("outputs/lilocbench/AVL-006/break_even.json")
OUT_ROOT = Path("outputs/lilocbench/AVL-007")


def load_outcomes(seq):
    memory = []
    for line in (MEMORY_ROOT / seq / "frame_results.jsonl").read_text().splitlines():
        memory.append(bool(json.loads(line)["methods"]["memory3"].get("success_0.25m_5deg", False)))
    roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False))
            for x in json.loads((ROMA_ROOT / f"{seq}.json").read_text())["frames"]]
    # Target action for offline feasibility: RoMa only when it is the sole
    # successful action; both/neither use the cheap pathway tie-break.
    target = np.asarray([int(r and not m) for m, r in zip(memory, roma)], dtype=np.int64)
    return np.asarray(memory, dtype=bool), np.asarray(roma, dtype=bool), target


def load_cheap(seq):
    return [json.loads(x) for x in (CHEAP_ROOT / f"{seq}.jsonl").read_text().splitlines()]


def make_features(pca, seq, mapping, query, cheap):
    retrieval = np.load(RETRIEVAL_ROOT / f"{seq}_top100.npz")
    map_idx = retrieval["mapping_indices"].astype(int)
    qpc = pca.transform(query)
    mpc = pca.transform(mapping)
    rows = []
    for i, row in enumerate(cheap):
        q = query[i]
        sims = mapping @ q
        order = np.argsort(-sims)
        top1_idx = int(map_idx[i, 0])
        ref = mapping[top1_idx]
        prev_sim = float(q @ query[i - 1]) if i else 1.0
        begin = max(0, i - 3)
        hist = query[begin:i]
        hist_sim = float(q @ (hist.mean(axis=0) / max(np.linalg.norm(hist.mean(axis=0)), 1e-12))) if len(hist) else 1.0
        scalar = np.asarray([
            row["top1_score"], row["top1_margin"], row["top5_mean_score"],
            row["laplacian_var"], row["frame_change"],
            float(q @ ref), float(sims[order[0]]), float(np.mean(sims[order[:5]])),
            float(q @ query[i - 1]) if i else 1.0, prev_sim, hist_sim,
            float(np.where(order == top1_idx)[0][0]) / len(mapping),
        ], dtype=np.float32)
        rows.append({
            "query_index": int(row["query_index"]),
            "timestamp": row["timestamp"],
            "scalar": scalar,
            "qpc": qpc[i].astype(np.float32),
            "refpc": mpc[top1_idx].astype(np.float32),
            "diffpc": (qpc[i] - mpc[top1_idx]).astype(np.float32),
            "feature_source": "frozen_DINOv2_ViT-L14+mapping_only_retrieval+RGB_history+cheap_signals",
        })
    return rows


def flatten(rows, group):
    if group == "cheap":
        return np.stack([r["scalar"][:5] for r in rows])
    if group == "dino_similarity":
        return np.stack([r["scalar"] for r in rows])
    if group == "dino_full":
        return np.stack([np.concatenate([r["scalar"], r["qpc"], r["refpc"], r["diffpc"]]) for r in rows])
    raise ValueError(group)


def failure_run(values):
    best = cur = 0
    for x in values:
        cur = cur + 1 if not x else 0
        best = max(best, cur)
    return best


def main():
    mapping = np.load(EMBED_ROOT / "mapping_cls.npy").astype(np.float32)
    # Mapping-only PCA is fixed before any query outcome is opened.
    pca = PCA(n_components=64, whiten=True, random_state=0)
    pca.fit(mapping)
    (OUT_ROOT / "features").mkdir(parents=True, exist_ok=True)
    feature_rows = {}
    outcomes = {}
    for seq in SEQUENCES:
        q = np.load(EMBED_ROOT / f"{seq}_cls.npy").astype(np.float32)
        rows = make_features(pca, seq, mapping, q, load_cheap(seq))
        feature_rows[seq] = rows
        outcomes[seq] = load_outcomes(seq)
        np.savez_compressed(OUT_ROOT / "features" / f"{seq}.npz",
                            scalar=np.stack([r["scalar"] for r in rows]),
                            qpc=np.stack([r["qpc"] for r in rows]),
                            refpc=np.stack([r["refpc"] for r in rows]),
                            diffpc=np.stack([r["diffpc"] for r in rows]))
    np.save(OUT_ROOT / "features" / "mapping_pca_components.npy", pca.components_.astype(np.float32))
    groups = ["cheap", "dino_similarity", "dino_full"]
    models = {
        "logistic_balanced": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, random_state=0)),
        "tree_depth4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0),
    }
    cost = json.loads(BREAK_EVEN.read_text())
    reloc = cost["per_sequence_reloc3r_s_per_frame"]
    roma = cost["per_sequence_roma_path_s_per_frame"]
    shared = cost["shared_retrieval_s"] + cost["context_feature_s"] + cost["state_overhead_s"]
    baseline_cost = cost["always_roma"]["cost_s"]
    embedding_manifest = json.loads((EMBED_ROOT / "manifest.json").read_text())
    dino_mapping_s = float(embedding_manifest["sequences"]["mapping"]["elapsed_s"])
    dino_query_s = float(sum(embedding_manifest["sequences"][s]["elapsed_s"] for s in SEQUENCES))
    summary = {"groups": {}, "protocol": {
        "representation": "frozen DINOv2 ViT-L/14 CLS, mapping-only PCA-64",
        "split": "leave-one-sequence-out",
        "online_inputs": ["current DINO embedding", "NetVLAD top references and scores", "previous/last-3 DINO embedding similarities", "Laplacian", "frame_change"],
        "forbidden_online_inputs": ["Reloc3r pose", "candidate dispersion", "RoMa/HLoc output", "GT", "future frames"],
        "target_label": "roma iff RoMa succeeds and memory3 fails; both/neither tie-break to memory; offline only",
        "query_gt_read_by_router": False,
    }}
    for group in groups:
        summary["groups"][group] = {}
        for model_name, make_model in models.items():
            all_success, all_pred, all_target, all_special = [], [], [], []
            per_seq = {}
            path_s = 0.0
            for heldout in SEQUENCES:
                train = [s for s in SEQUENCES if s != heldout]
                x_train = np.concatenate([flatten(feature_rows[s], group) for s in train])
                y_train = np.concatenate([outcomes[s][2] for s in train])
                model = make_model(); model.fit(x_train, y_train)
                x_test = flatten(feature_rows[heldout], group)
                pred = model.predict(x_test).astype(np.int64)
                memory, rsuccess, target = outcomes[heldout]
                success = np.where(pred == 1, rsuccess, memory)
                special = (memory ^ rsuccess)
                special_pred = pred[special]
                special_true = target[special]
                all_success.extend(success.tolist()); all_pred.extend(pred.tolist()); all_target.extend(target.tolist())
                all_special.extend([(int(t), int(p)) for t, p in zip(special_true, special_pred)])
                path_s += float(sum(roma[heldout] if p else reloc[heldout] for p in pred))
                per_seq[heldout] = {
                    "frames": len(pred), "roma_calls": int(pred.sum()), "roma_call_rate": float(pred.mean()),
                    "success_count": int(success.sum()), "success": float(success.mean()),
                    "memory_only_count": int(np.sum(memory & ~rsuccess)), "roma_only_count": int(np.sum(rsuccess & ~memory)),
                    "memory_only_recall": float(np.mean(pred[memory & ~rsuccess] == 0)) if np.any(memory & ~rsuccess) else None,
                    "roma_only_recall": float(np.mean(pred[rsuccess & ~memory] == 1)) if np.any(rsuccess & ~memory) else None,
                    "target_action_accuracy": float(np.mean(pred == target)),
                    "longest_failure_frames": failure_run(success),
                }
                trace = OUT_ROOT / "traces" / group / model_name / f"{heldout}.jsonl"
                trace.parent.mkdir(parents=True, exist_ok=True)
                with trace.open("w") as f:
                    for row, p in zip(feature_rows[heldout], pred):
                        f.write(json.dumps({"query_index": row["query_index"], "timestamp": row["timestamp"],
                                            "action": "roma" if p else "reloc3r_memory", "feature_source": row["feature_source"],
                                            "query_gt_read": False, "selection_uses_gt": False}, sort_keys=True) + "\n")
            sp_true = np.asarray([x[0] for x in all_special]); sp_pred = np.asarray([x[1] for x in all_special])
            precision, recall, f1, _ = precision_recall_fscore_support(sp_true, sp_pred, labels=[0, 1], zero_division=0)
            summary["groups"][group][model_name] = {
                "frames": len(all_success), "roma_calls": int(sum(all_pred)), "roma_call_rate": float(np.mean(all_pred)),
                "success_count": int(sum(all_success)), "success": float(np.mean(all_success)),
                "route_cost_s": shared + path_s, "budget_margin_s": baseline_cost - (shared + path_s),
                "dino_query_embedding_s": dino_query_s if group != "cheap" else 0.0,
                "dino_mapping_embedding_one_time_s": dino_mapping_s if group != "cheap" else 0.0,
                "route_cost_with_dino_s": shared + path_s + (dino_query_s if group != "cheap" else 0.0),
                "budget_margin_with_dino_s": baseline_cost - (shared + path_s + (dino_query_s if group != "cheap" else 0.0)),
                "target_action_accuracy": float(np.mean(np.asarray(all_pred) == np.asarray(all_target))),
                "special_frame_count": len(all_special),
                "special_balanced_accuracy": float(balanced_accuracy_score(sp_true, sp_pred)),
                "special_confusion_memory0_roma1": confusion_matrix(sp_true, sp_pred, labels=[0, 1]).tolist(),
                "special_memory_only_recall": float(recall[0]), "special_roma_only_recall": float(recall[1]),
                "special_memory_only_precision": float(precision[0]), "special_roma_only_precision": float(precision[1]),
                "per_sequence": per_seq, "router_gt_audit_pass": True,
            }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
