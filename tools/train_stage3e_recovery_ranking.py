#!/usr/bin/env python3
"""Train Stage 3E P(recover)+cost pairwise acquisition ranker."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from tools.stage3b_policy import action_features
from tools.stage3c6_utility_policy import UtilityComponentPolicy, policy_state_features
from tools.stage3e_recovery_ranking_policy import RecoveryCostRanker


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty recovery/cost ranking data")
    return rows


def split_rows(rows: Sequence[Mapping[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    # Newly collected DAgger states are deliberately all train data.  The
    # validation tail is reserved from frozen 0--79 Oracle states, so training
    # never tunes against the final held-out rollout split.
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("source_type") != "dagger":
            by_sequence[str(row["sequence"])].append(int(row["frame_index"]))
    validation_frames = {}
    for sequence, frames in by_sequence.items():
        unique = sorted(set(frames)); count = max(1, round(len(unique) * fraction))
        validation_frames[sequence] = set(unique[-count:])
    train, validation = [], []
    for row in rows:
        if row.get("source_type") != "dagger" and int(row["frame_index"]) in validation_frames[str(row["sequence"])]:
            validation.append(dict(row))
        else:
            train.append(dict(row))
    if not train or not validation:
        raise ValueError("temporal split produced empty partition")
    return train, validation, {key: sorted(value) for key, value in validation_frames.items()}


def normalizer(rows: Sequence[Mapping[str, Any]], feature_level: str) -> dict[str, list[float]]:
    states = np.stack([policy_state_features(row["observation"], feature_level) for row in rows])
    actions = np.concatenate([np.stack([action_features(row["observation"], action) for action in row["actions"]])
                              for row in rows])
    return {"state_mean": states.mean(0).astype(np.float32).tolist(),
            "state_std": np.maximum(states.std(0), 1e-4).astype(np.float32).tolist(),
            "action_mean": actions.mean(0).astype(np.float32).tolist(),
            "action_std": np.maximum(actions.std(0), 1e-4).astype(np.float32).tolist()}


def vectorize(rows: Sequence[Mapping[str, Any]], norm: Mapping[str, Any], feature_level: str) -> list[dict[str, Any]]:
    sm, ss = np.asarray(norm["state_mean"], np.float32), np.asarray(norm["state_std"], np.float32)
    am, ast = np.asarray(norm["action_mean"], np.float32), np.asarray(norm["action_std"], np.float32)
    result = []
    for row in rows:
        actions = row["actions"]
        result.append({
            "state": (policy_state_features(row["observation"], feature_level) - sm) / ss,
            "actions": np.stack([(action_features(row["observation"], a) - am) / ast for a in actions]),
            "recover": np.asarray([float(a["recovery_target"]) for a in actions], np.float32),
            "log_cost": np.log1p(np.asarray([max(0.0, float(a["incremental_cost_s"])) for a in actions], np.float32)),
            "oracle": np.asarray([float(a["oracle_value"]) for a in actions], np.float32),
            "kinds": [str(a["kind"]) for a in actions], "weight": float(row.get("training_weight", 1.0)),
        })
    return result


def batch(rows: Sequence[Mapping[str, Any]], device: torch.device):
    maximum = max(len(row["actions"]) for row in rows); count = len(rows)
    actions = np.zeros((count, maximum, rows[0]["actions"].shape[-1]), np.float32)
    recover = np.zeros((count, maximum), np.float32); costs = recover.copy(); oracle = recover.copy(); mask = np.zeros((count, maximum), bool)
    for i, row in enumerate(rows):
        n = len(row["actions"]); actions[i, :n] = row["actions"]; recover[i, :n] = row["recover"]
        costs[i, :n] = row["log_cost"]; oracle[i, :n] = row["oracle"]; mask[i, :n] = True
    return (torch.from_numpy(np.stack([row["state"] for row in rows]).astype(np.float32)).to(device),
            torch.from_numpy(actions).to(device), torch.from_numpy(recover).to(device), torch.from_numpy(costs).to(device),
            torch.from_numpy(oracle).to(device), torch.from_numpy(mask).to(device),
            torch.tensor([row["weight"] for row in rows], dtype=torch.float32, device=device))


def weighted_state(loss: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    per = loss.masked_fill(~mask, 0.0).sum(1) / mask.sum(1).clamp_min(1)
    return per.mean() if weights is None else (per * weights).sum() / weights.sum().clamp_min(1e-6)


def pairwise(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    delta_target = target[:, :, None] - target[:, None, :]
    valid = mask[:, :, None] & mask[:, None, :] & (delta_target.abs() > 1e-6)
    signed = torch.sign(delta_target) * (predicted[:, :, None] - predicted[:, None, :])
    # Large Oracle regret pairs are the costly errors this stage is intended
    # to fix; cap their influence so targeted data does not erase the prior.
    scale = delta_target.abs().clamp(0.05, 1.0)
    loss = (nn.functional.softplus(-signed) * scale).masked_fill(~valid, 0.0)
    per = loss.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return per.mean() if weights is None else (per * weights).sum() / weights.sum().clamp_min(1e-6)


def losses(model, item, lam: float, weights: torch.Tensor | None):
    states, actions, recover, log_cost, oracle, mask, _ = item
    b, n, _ = actions.shape
    # The policy model operates per state; flattening action rows preserves
    # its shared encoder while keeping a batched loss below.
    flat_state = states[:, None, :].expand(-1, n, -1).reshape(-1, states.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    encoded = model.encoder(torch.cat((flat_state, flat_actions), dim=-1)).reshape(b, n, -1)
    logits = model.recovery_head(encoded).squeeze(-1); predicted_log_cost = model.log_cost_head(encoded).squeeze(-1)
    score = torch.sigmoid(logits) - lam * torch.expm1(predicted_log_cost).clamp_min(0.0)
    return (weighted_state(nn.functional.binary_cross_entropy_with_logits(logits, recover, reduction="none"), mask, weights),
            weighted_state(nn.functional.smooth_l1_loss(predicted_log_cost, log_cost, reduction="none"), mask, weights),
            pairwise(score, oracle, mask, weights), score, logits, predicted_log_cost)


def evaluate(model, rows, device, lam, batch_size, geometry_margin_threshold: float = 0.0):
    model.eval(); totals = defaultdict(float)
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            subset = rows[begin:begin + batch_size]; item = batch(subset, device)
            rec, cost, rank, score, logits, predicted_cost = losses(model, item, lam, None)
            states, actions, targets, log_cost, oracle, mask, _ = item
            totals["states"] += len(subset); totals["recovery_loss"] += float(rec) * len(subset)
            totals["cost_loss"] += float(cost) * len(subset); totals["pairwise_loss"] += float(rank) * len(subset)
            predicted = torch.expm1(predicted_cost).clamp_min(0.0)
            totals["cost_abs"] += float((predicted.sub(torch.expm1(log_cost)).abs().masked_fill(~mask, 0.0)).sum())
            totals["actions"] += int(mask.sum())
            for i, row in enumerate(subset):
                g = [j for j,k in enumerate(row["kinds"]) if k == "geometry"]
                c = [j for j,k in enumerate(row["kinds"]) if k in {"localize", "memory"}]
                if not g or not c: continue
                pg, tg = float(score[i, g].max()), float(oracle[i, g].max())
                pc, tc = float(score[i, c].max()), float(oracle[i, c].max())
                truth, selected = tg >= tc, pg - pc >= geometry_margin_threshold
                totals["geometry_states"] += 1; totals["geometry_correct"] += int(truth == selected)
                totals["geometry_positive"] += int(truth); totals["geometry_selected"] += int(selected)
                totals["geometry_hits"] += int(truth and selected)
    n = max(1, totals["states"])
    return {"states": int(totals["states"]), "recovery_loss": totals["recovery_loss"] / n,
            "cost_loss": totals["cost_loss"] / n, "pairwise_loss": totals["pairwise_loss"] / n,
            "cost_mae_s": totals["cost_abs"] / max(1, totals["actions"]),
            "geometry_margin_sign_accuracy": totals["geometry_correct"] / max(1, totals["geometry_states"]),
            "geometry_recall": totals["geometry_hits"] / max(1, totals["geometry_positive"]),
            "geometry_precision": totals["geometry_hits"] / max(1, totals["geometry_selected"]),
            "geometry_positive": int(totals["geometry_positive"]), "geometry_selected": int(totals["geometry_selected"])}


def calibrate_geometry_margin(model, rows, device, lam: float, minimum_recall: float) -> dict[str, float | int]:
    """Calibrate one threshold on the same explicit controller score.

    The objective is the validation Oracle value increment from choosing
    Geometry rather than the best cheap action.  It introduces no auxiliary
    classifier and is selected before any fresh split rollout.
    """
    model.eval(); margins = []
    with torch.no_grad():
        for row in rows:
            state = torch.from_numpy(row["state"].astype(np.float32)).to(device)
            actions = torch.from_numpy(row["actions"].astype(np.float32)).to(device)
            logits, log_cost = model(state, actions)
            predicted = torch.sigmoid(logits) - lam * torch.expm1(log_cost).clamp_min(0.0)
            geometry = [i for i,k in enumerate(row["kinds"]) if k == "geometry"]
            cheap = [i for i,k in enumerate(row["kinds"]) if k in {"localize", "memory"}]
            if not geometry or not cheap:
                continue
            pg, tg = float(predicted[geometry].max()), float(row["oracle"][geometry].max())
            pc, tc = float(predicted[cheap].max()), float(row["oracle"][cheap].max())
            margins.append((pg - pc, tg - tc))
    if not margins:
        return {"threshold": 0.0, "states": 0, "selected": 0, "oracle_incremental_gain": 0.0}
    values = sorted({0.0, *(margin for margin, _ in margins), max(margin for margin, _ in margins) + 1e-6})
    positives = sum(truth >= 0.0 for _, truth in margins)
    candidates = []
    for threshold in values:
        selected = [(predicted, truth) for predicted, truth in margins if predicted >= threshold]
        gain = sum(truth for _, truth in selected)
        hits = sum(truth >= 0.0 for _, truth in selected)
        recall = 0.0 if positives == 0 else hits / positives
        candidates.append((threshold, selected, gain, recall))
    eligible = [candidate for candidate in candidates if candidate[3] >= minimum_recall]
    # If rank ordering cannot reach the target, surface that limitation by
    # choosing the highest-recall threshold rather than silently collapsing to
    # a no-Geometry controller.
    pool = eligible or candidates
    threshold, selected, _, _ = max(pool, key=lambda candidate: (candidate[2], -len(candidate[1]), candidate[0])
                                     if eligible else (candidate[3], candidate[2], -len(candidate[1]), candidate[0]))
    hits = sum(truth >= 0.0 for _, truth in selected)
    return {"threshold": float(threshold), "states": len(margins), "selected": len(selected),
            "oracle_incremental_gain": float(sum(truth for _, truth in selected)),
            "geometry_recall": 0.0 if positives == 0 else hits / positives,
            "geometry_precision": 0.0 if not selected else hits / len(selected),
            "minimum_recall_target": minimum_recall,
            "recall_constraint_satisfied": bool((0.0 if positives == 0 else hits / positives) >= minimum_recall)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=96); parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cost-loss-weight", type=float, default=0.5); parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--minimum-validation-geometry-recall", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2); parser.add_argument("--seed", type=int, default=1777)
    parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device = torch.device(args.device)
    base = UtilityComponentPolicy.load(args.base_checkpoint); rows = read_rows(args.dataset)
    train_rows, val_rows, validation_frames = split_rows(rows, args.validation_fraction)
    norm = normalizer(train_rows, base.feature_level); train = vectorize(train_rows, norm, base.feature_level); val = vectorize(val_rows, norm, base.feature_level)
    config = {"state_dim": len(norm["state_mean"]), "action_dim": len(norm["action_mean"]), "hidden_dim": args.hidden_dim}
    model = RecoveryCostRanker(**config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best, best_score, history = None, float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = np.random.permutation(len(train))
        for begin in range(0, len(order), args.batch_size):
            item = batch([train[int(i)] for i in order[begin:begin + args.batch_size]], device)
            recovery, cost, rank, *_ = losses(model, item, args.lambda_cost, item[-1])
            loss = recovery + args.cost_loss_weight * cost + args.rank_loss_weight * rank
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        metrics = evaluate(model, val, device, args.lambda_cost, args.batch_size); history.append({"epoch": epoch, "validation": metrics})
        score = metrics["recovery_loss"] + metrics["pairwise_loss"]
        if score < best_score:
            best_score = score; best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert best is not None; model.load_state_dict(best); args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_geometry_margin(model, val, device, args.lambda_cost, args.minimum_validation_geometry_recall)
    torch.save({"schema_version": 1, "family": "stage3e_recovery_cost_ranking", "model_config": config,
                "model_state": model.state_dict(), "normalizer": norm, "lambda_cost": args.lambda_cost,
                "geometry_margin_threshold": calibration["threshold"], "base_checkpoint": str(args.base_checkpoint)}, args.output_dir / "recovery_cost_ranking.pt")
    summary = {"schema_version": 1, "family": "stage3e_recovery_cost_ranking", "base_checkpoint": str(args.base_checkpoint),
               "feature_level": base.feature_level, "training": {"dataset": str(args.dataset), "train_states": len(train), "validation_states": len(val),
               "validation_frames": validation_frames, "dagger_states_train_only": sum(r.get("source_type") == "dagger" for r in train_rows),
               "seed": args.seed, "epochs": args.epochs, "cost_loss_weight": args.cost_loss_weight, "rank_loss_weight": args.rank_loss_weight},
               "controller": {"formula": "P_recover - lambda_cost * predicted_incremental_cost_s", "lambda_cost": args.lambda_cost,
                              "geometry_margin_threshold": calibration["threshold"], "terminal_boundary": "frozen Stage3D component controller"},
               "calibration": calibration, "best_validation_score": best_score,
               "final_validation": evaluate(model, val, device, args.lambda_cost, args.batch_size, calibration["threshold"]), "history": history,
               "online_boundary": "only current pre-action F2 observation/action descriptors are used online; Oracle targets are offline only"}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
