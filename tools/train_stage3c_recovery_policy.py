#!/usr/bin/env python3
"""Train Stage 3C recovery-probability + incremental-cost ranking policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from tools.stage3b_policy import ACTION_DIM, STATE_DIM, action_features, state_features
from tools.stage3c_recovery_policy import create_model


def read_records(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("recovery dataset is empty")
    return rows


def temporal_split(records: Sequence[Mapping[str, Any]], validation_fraction: float):
    by_sequence: Dict[str, List[int]] = {}
    for row in records:
        if row.get("dagger_diagnostics"):
            continue
        by_sequence.setdefault(str(row["sequence"]), []).append(int(row["frame_index"]))
    validation_frames: Dict[str, set[int]] = {}
    for sequence, frames in by_sequence.items():
        unique = sorted(set(frames))
        count = max(1, int(round(len(unique) * validation_fraction)))
        validation_frames[sequence] = set(unique[-count:])
    train = [row for row in records if row.get("dagger_diagnostics") or
             int(row["frame_index"]) not in validation_frames[str(row["sequence"])]]
    validation = [row for row in records if not row.get("dagger_diagnostics") and
                  int(row["frame_index"]) in validation_frames[str(row["sequence"])]]
    if not train or not validation:
        raise ValueError("temporal split produced an empty partition")
    return train, validation, {key: sorted(value) for key, value in validation_frames.items()}


def normalizer(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    states = np.stack([state_features(row["observation"]) for row in records])
    actions = np.concatenate([np.stack([action_features(row["observation"], action) for action in row["actions"]])
                              for row in records])
    def stats(array: np.ndarray):
        return array.mean(axis=0).astype(np.float32), np.maximum(array.std(axis=0), 1e-4).astype(np.float32)
    sm, ss = stats(states); am, ast = stats(actions)
    return {"state_mean": sm.tolist(), "state_std": ss.tolist(), "action_mean": am.tolist(), "action_std": ast.tolist()}


def vectorize(records: Sequence[Mapping[str, Any]], norm: Mapping[str, Any]):
    sm = np.asarray(norm["state_mean"], dtype=np.float32); ss = np.asarray(norm["state_std"], dtype=np.float32)
    am = np.asarray(norm["action_mean"], dtype=np.float32); ast = np.asarray(norm["action_std"], dtype=np.float32)
    result = []
    for row in records:
        actions = row["actions"]
        result.append({
            "state": (state_features(row["observation"]) - sm) / ss,
            "actions": np.stack([(action_features(row["observation"], action) - am) / ast for action in actions]),
            "recovery": np.asarray([float(action["recovery_target"]) for action in actions], dtype=np.float32),
            "log_cost": np.log1p(np.asarray([max(0.0, float(action["incremental_cost_s"])) for action in actions], dtype=np.float32)),
            "advantage": np.asarray([float(action["oracle_advantage"]) for action in actions], dtype=np.float32),
            "kinds": [str(action["kind"]) for action in actions],
            "weight": float(row.get("training_weight", 1.0)),
        })
    return result


def collate(rows: Sequence[Mapping[str, Any]], device: torch.device):
    maximum = max(len(row["actions"]) for row in rows)
    count = len(rows)
    actions = np.zeros((count, maximum, ACTION_DIM), dtype=np.float32)
    recovery = np.zeros((count, maximum), dtype=np.float32)
    log_cost = np.zeros((count, maximum), dtype=np.float32)
    advantage = np.zeros((count, maximum), dtype=np.float32)
    mask = np.zeros((count, maximum), dtype=bool)
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=np.float32)
    for index, row in enumerate(rows):
        size = len(row["actions"])
        actions[index, :size] = row["actions"]
        recovery[index, :size] = row["recovery"]
        log_cost[index, :size] = row["log_cost"]
        advantage[index, :size] = row["advantage"]
        mask[index, :size] = True
    return (
        torch.from_numpy(np.stack([row["state"] for row in rows]).astype(np.float32)).to(device),
        torch.from_numpy(actions).to(device), torch.from_numpy(recovery).to(device),
        torch.from_numpy(log_cost).to(device), torch.from_numpy(advantage).to(device),
        torch.from_numpy(mask).to(device), torch.from_numpy(weights).to(device),
    )


def weighted_per_state(losses: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    per_state = losses.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_state.mean() if weights is None else (per_state * weights).sum() / weights.sum().clamp_min(1e-6)


def pairwise_loss(predicted_advantage: torch.Tensor, target_advantage: torch.Tensor,
                  mask: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    diff_pred = predicted_advantage[:, :, None] - predicted_advantage[:, None, :]
    diff_target = target_advantage[:, :, None] - target_advantage[:, None, :]
    valid = mask[:, :, None] & mask[:, None, :] & (diff_target.abs() > 1e-7)
    signed = torch.sign(diff_target) * diff_pred
    losses = nn.functional.softplus(-signed).masked_fill(~valid, 0.0)
    per_state = losses.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    return per_state.mean() if weights is None else (per_state * weights).sum() / weights.sum().clamp_min(1e-6)


def losses(model, batch, *, lambda_cost: float, weights: torch.Tensor | None):
    states, actions, recovery, log_cost, advantage, mask, _ = batch
    logits, predicted_log_cost = model(states, actions, mask)
    recovery_loss = weighted_per_state(nn.functional.binary_cross_entropy_with_logits(logits, recovery, reduction="none"), mask, weights)
    cost_loss = weighted_per_state(nn.functional.smooth_l1_loss(predicted_log_cost, log_cost, reduction="none"), mask, weights)
    predicted_advantage = torch.sigmoid(logits) - lambda_cost * torch.expm1(predicted_log_cost).clamp_min(0.0)
    rank_loss = pairwise_loss(predicted_advantage, advantage, mask, weights)
    return recovery_loss, cost_loss, rank_loss, predicted_advantage


def evaluate(model, rows: Sequence[Mapping[str, Any]], device: torch.device, *, lambda_cost: float, batch_size: int):
    model.eval()
    sums = {"recovery_loss": 0.0, "cost_loss": 0.0, "rank_loss": 0.0, "count": 0,
            "pair_correct": 0, "pair_total": 0, "geometry_margin_correct": 0, "geometry_margin_total": 0,
            "cost_abs_error": 0.0, "cost_count": 0, "brier": 0.0, "recovery_count": 0}
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            subset = rows[begin:begin + batch_size]
            batch = collate(subset, device)
            states, actions, recovery, log_cost, advantage, mask, _ = batch
            rl, cl, pl, predicted_advantage = losses(model, batch, lambda_cost=lambda_cost, weights=None)
            logits, predicted_log_cost = model(states, actions, mask)
            probability = torch.sigmoid(logits)
            predicted_cost = torch.expm1(predicted_log_cost).clamp_min(0.0)
            sums["recovery_loss"] += float(rl) * len(subset); sums["cost_loss"] += float(cl) * len(subset); sums["rank_loss"] += float(pl) * len(subset); sums["count"] += len(subset)
            valid_pair = mask[:, :, None] & mask[:, None, :] & ((advantage[:, :, None] - advantage[:, None, :]).abs() > 1e-7)
            sign_correct = ((predicted_advantage[:, :, None] - predicted_advantage[:, None, :]) *
                            (advantage[:, :, None] - advantage[:, None, :]) > 0) & valid_pair
            sums["pair_correct"] += int(sign_correct.sum().item()); sums["pair_total"] += int(valid_pair.sum().item())
            sums["cost_abs_error"] += float((predicted_cost.sub(torch.expm1(log_cost)).abs().masked_fill(~mask, 0.0)).sum().item())
            sums["cost_count"] += int(mask.sum().item())
            sums["brier"] += float(((probability - recovery).square().masked_fill(~mask, 0.0)).sum().item()); sums["recovery_count"] += int(mask.sum().item())
            for batch_index, row in enumerate(subset):
                geometry = [index for index, kind in enumerate(row["kinds"]) if kind == "geometry"]
                alternatives = [index for index, kind in enumerate(row["kinds"]) if kind != "geometry"]
                if geometry and alternatives:
                    g = geometry[0]
                    pred_margin = float(predicted_advantage[batch_index, g] - predicted_advantage[batch_index, alternatives].max())
                    true_margin = float(advantage[batch_index, g] - advantage[batch_index, alternatives].max())
                    sums["geometry_margin_correct"] += int((pred_margin >= 0.0) == (true_margin >= 0.0))
                    sums["geometry_margin_total"] += 1
    n = max(1, sums["count"])
    return {
        "states": sums["count"], "recovery_loss": sums["recovery_loss"] / n, "cost_loss": sums["cost_loss"] / n,
        "pairwise_loss": sums["rank_loss"] / n,
        "pairwise_sign_accuracy": sums["pair_correct"] / max(1, sums["pair_total"]),
        "geometry_margin_sign_accuracy": sums["geometry_margin_correct"] / max(1, sums["geometry_margin_total"]),
        "geometry_margin_states": sums["geometry_margin_total"],
        "cost_mae_s": sums["cost_abs_error"] / max(1, sums["cost_count"]),
        "recovery_brier": sums["brier"] / max(1, sums["recovery_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cost-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, required=True)
    parser.add_argument("--defer-threshold", type=float, default=0.0)
    parser.add_argument("--terminal-continue-margin", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = read_records(args.dataset)
    train_records, validation_records, validation_frames = temporal_split(records, args.validation_fraction)
    norm = normalizer(train_records)
    train, validation = vectorize(train_records, norm), vectorize(validation_records, norm)
    config = {"hidden_dim": args.hidden_dim}
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []; best_state = None; best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        for begin in range(0, len(order), args.batch_size):
            batch = collate([train[int(index)] for index in order[begin:begin + args.batch_size]], device)
            weights = batch[-1]
            recovery_loss, cost_loss, rank_loss, _ = losses(model, batch, lambda_cost=args.lambda_cost, weights=weights)
            loss = recovery_loss + args.cost_loss_weight * cost_loss + args.rank_loss_weight * rank_loss
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_metrics = evaluate(model, train, device, lambda_cost=args.lambda_cost, batch_size=args.batch_size)
        validation_metrics = evaluate(model, validation, device, lambda_cost=args.lambda_cost, batch_size=args.batch_size)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        score = validation_metrics["pairwise_loss"] + validation_metrics["recovery_loss"]
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "family": "stage3c_recovery", "model_config": config,
        "model_state": model.state_dict(), "normalizer": norm,
        "controller": {"lambda_cost": args.lambda_cost, "defer_threshold": args.defer_threshold,
                       "terminal_continue_margin": args.terminal_continue_margin},
        "training": {"dataset": str(args.dataset), "seed": args.seed, "epochs": args.epochs,
                     "split": "per-sequence temporal tail; all DAgger states train", "validation_frames": validation_frames,
                     "train_states": len(train), "validation_states": len(validation),
                     "cost_loss_weight": args.cost_loss_weight, "rank_loss_weight": args.rank_loss_weight},
    }
    torch.save(checkpoint, args.output_dir / "recovery_cost_policy.pt")
    summary = {"schema_version": 1, "family": "stage3c_recovery", "training": checkpoint["training"],
               "controller": checkpoint["controller"], "best_validation_score": best_score,
               "final_train": evaluate(model, train, device, lambda_cost=args.lambda_cost, batch_size=args.batch_size),
               "final_validation": evaluate(model, validation, device, lambda_cost=args.lambda_cost, batch_size=args.batch_size),
               "history": history}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
