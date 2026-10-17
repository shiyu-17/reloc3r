#!/usr/bin/env python3
"""Train a masked structured policy + Q-value baseline from Stage 3A labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from tools.stage3b_policy import (ACTION_DIM, STATE_DIM, action_features,
                                  create_model, is_deployment_legal, state_features)


def read_dataset(path: Path) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("dataset is empty")
    return records


def temporal_split(records: Sequence[Mapping[str, Any]], validation_fraction: float):
    frames_by_sequence: Dict[str, List[int]] = {}
    for row in records:
        # Policy-induced DAgger states are the training intervention.  Putting
        # their later frame numbers into a generic temporal tail silently
        # turns the entire DAgger round into validation-only data.
        if row.get("dagger_diagnostics"):
            continue
        frames_by_sequence.setdefault(str(row["sequence"]), []).append(int(row["frame_index"]))
    validation_frames: Dict[str, set[int]] = {}
    for sequence, frames in frames_by_sequence.items():
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
    actions = np.concatenate([np.stack([action_features(row["observation"], action)
                                        for action in row["action_values"]
                                        if is_deployment_legal(row["observation"], action)]) for row in records])
    values = np.asarray([action["q_star"] for row in records for action in row["action_values"]
                         if is_deployment_legal(row["observation"], action)], dtype=np.float32)
    def stats(array: np.ndarray):
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)
    state_mean, state_std = stats(states)
    action_mean, action_std = stats(actions)
    return {"state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
            "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
            "value_mean": float(values.mean()), "value_std": float(max(values.std(), 1e-4))}


def training_weight(record: Mapping[str, Any], targeting: Mapping[str, Any] | None) -> float:
    if targeting is None:
        return 1.0
    weights = targeting.get("tag_weights", {})
    selected = [float(weights[tag]) for tag in record.get("target_tags", []) if tag in weights]
    value = max([float(targeting.get("default_weight", 1.0)), *selected])
    return min(value, float(targeting.get("maximum_weight", value)))


def vectorize(records: Sequence[Mapping[str, Any]], norm: Mapping[str, Any],
              targeting: Mapping[str, Any] | None = None):
    s_mean = np.asarray(norm["state_mean"], dtype=np.float32)
    s_std = np.asarray(norm["state_std"], dtype=np.float32)
    a_mean = np.asarray(norm["action_mean"], dtype=np.float32)
    a_std = np.asarray(norm["action_std"], dtype=np.float32)
    q_mean, q_std = float(norm["value_mean"]), float(norm["value_std"])
    examples = []
    for row in records:
        actions = [action for action in row["action_values"]
                   if is_deployment_legal(row["observation"], action)]
        if not actions:
            raise ValueError("deployment mask removed every action")
        values = np.asarray([float(action["q_star"]) for action in actions], dtype=np.float32)
        optimal = int(np.argmax(values))
        optimal_mask = np.isclose(values, values.max(), rtol=0.0, atol=1e-12)
        examples.append({
            "state": (state_features(row["observation"]) - s_mean) / s_std,
            "actions": np.stack([(action_features(row["observation"], action) - a_mean) / a_std
                                 for action in actions]),
            "values": (values - q_mean) / q_std,
            "optimal": optimal,
            "optimal_mask": optimal_mask,
            "raw_values": values,
            "training_weight": training_weight(row, targeting),
            "sequence": row["sequence"], "frame_index": row["frame_index"],
        })
    return examples


def collate(examples: Sequence[Mapping[str, Any]], device: torch.device):
    maximum = max(len(item["actions"]) for item in examples)
    states = np.stack([item["state"] for item in examples]).astype(np.float32)
    actions = np.zeros((len(examples), maximum, ACTION_DIM), dtype=np.float32)
    values = np.zeros((len(examples), maximum), dtype=np.float32)
    mask = np.zeros((len(examples), maximum), dtype=bool)
    targets = np.zeros(len(examples), dtype=np.int64)
    optimal_masks = np.zeros((len(examples), maximum), dtype=bool)
    raw_values = np.full((len(examples), maximum), np.nan, dtype=np.float32)
    weights = np.ones(len(examples), dtype=np.float32)
    for index, item in enumerate(examples):
        count = len(item["actions"])
        actions[index, :count] = item["actions"]
        values[index, :count] = item["values"]
        raw_values[index, :count] = item["raw_values"]
        mask[index, :count] = True
        targets[index] = item["optimal"]
        optimal_masks[index, :count] = item["optimal_mask"]
        weights[index] = float(item.get("training_weight", 1.0))
    return (torch.from_numpy(states).to(device), torch.from_numpy(actions).to(device),
            torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(targets).to(device), torch.from_numpy(optimal_masks).to(device),
            torch.from_numpy(weights).to(device), raw_values)


def multi_optimal_policy_loss(logits: torch.Tensor, optimal: torch.Tensor,
                              weights: torch.Tensor | None = None) -> torch.Tensor:
    """NLL of the total mass assigned to any Q*-tied legal action."""
    losses = -(torch.logsumexp(logits.masked_fill(~optimal, -1e9), dim=1) -
               torch.logsumexp(logits, dim=1))
    return losses.mean() if weights is None else (losses * weights).sum() / weights.sum()


def weighted_value_loss(predicted: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                        weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = nn.functional.smooth_l1_loss(predicted, values, reduction="none")
    losses = losses.masked_fill(~mask, 0.0)
    per_state = losses.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_state.mean() if weights is None else (per_state * weights).sum() / weights.sum()


def evaluate(model: nn.Module, examples: Sequence[Mapping[str, Any]], device: torch.device,
             batch_size: int, value_weight: float) -> Dict[str, float]:
    model.eval()
    total = correct = count = 0
    total_loss = total_policy = total_value = total_regret = 0.0
    with torch.no_grad():
        for begin in range(0, len(examples), batch_size):
            batch = examples[begin:begin + batch_size]
            states, actions, values, mask, target, optimal, _, raw = collate(batch, device)
            logits, predicted = model(states, actions, mask)
            policy_loss = multi_optimal_policy_loss(logits, optimal)
            value_loss = weighted_value_loss(predicted, values, mask)
            chosen = torch.argmax(logits, dim=1).cpu().numpy()
            total += len(batch)
            correct += int(optimal[torch.arange(len(batch), device=device),
                                   torch.from_numpy(chosen).to(device)].sum().item())
            for index, choice in enumerate(chosen):
                valid = raw[index][~np.isnan(raw[index])]
                total_regret += float(valid.max() - valid[int(choice)])
            total_loss += float(policy_loss + value_weight * value_loss) * len(batch)
            total_policy += float(policy_loss) * len(batch)
            total_value += float(value_loss) * len(batch)
            count += len(batch)
    return {"loss": total_loss / count, "policy_loss": total_policy / count,
            "value_loss": total_value / count, "top1_oracle_action_accuracy": correct / total,
            "mean_oracle_regret": total_regret / total, "states": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("mlp", "transformer"), required=True)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--targeting-config", type=Path,
                        help="optional bounded training weights for rare/high-regret DAgger states")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="warm-start model and normalizer from the rollout policy")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = read_dataset(args.dataset)
    targeting = (json.loads(args.targeting_config.read_text(encoding="utf-8"))
                 if args.targeting_config is not None else None)
    train_records, validation_records, validation_frames = temporal_split(records, args.validation_fraction)
    initial = (torch.load(args.init_checkpoint, map_location=device, weights_only=False)
               if args.init_checkpoint is not None else None)
    norm = initial["normalizer"] if initial is not None else normalizer(train_records)
    train = vectorize(train_records, norm, targeting)
    validation = vectorize(validation_records, norm, None)
    config = {"hidden_dim": args.hidden_dim, "layers": args.layers, "heads": args.heads}
    model = create_model(args.model, config).to(device)
    if initial is not None:
        if initial["model_type"] != args.model or initial["model_config"] != config:
            raise ValueError("--init-checkpoint architecture does not match requested model")
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    best_state = None; best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); order = np.random.permutation(len(train))
        for begin in range(0, len(order), args.batch_size):
            batch = [train[int(index)] for index in order[begin:begin + args.batch_size]]
            states, actions, values, mask, target, optimal, weights, _ = collate(batch, device)
            logits, predicted = model(states, actions, mask)
            policy_loss = multi_optimal_policy_loss(logits, optimal, weights)
            value_loss = weighted_value_loss(predicted, values, mask, weights)
            loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_metrics = evaluate(model, train, device, args.batch_size, args.value_weight)
        validation_metrics = evaluate(model, validation, device, args.batch_size, args.value_weight)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        if validation_metrics["mean_oracle_regret"] < best_score:
            best_score = validation_metrics["mean_oracle_regret"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "model_type": args.model, "model_config": config,
        "model_state": model.state_dict(), "normalizer": norm,
        "training": {"seed": args.seed, "epochs": args.epochs, "value_weight": args.value_weight,
                     "split": "per-sequence temporal tail", "validation_frames": validation_frames,
                     "train_states": len(train), "validation_states": len(validation),
                     "dataset": str(args.dataset),
                     "targeting_config": None if args.targeting_config is None else str(args.targeting_config),
                     "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
                     "target_weight_counts": dict(Counter(
                         str(training_weight(row, targeting)) for row in train_records))},
    }
    torch.save(checkpoint, args.output_dir / f"{args.model}_policy_value.pt")
    final = {"model": args.model, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
             "training": checkpoint["training"], "best_validation_oracle_regret": best_score,
             "final_train": evaluate(model, train, device, args.batch_size, args.value_weight),
             "final_validation": evaluate(model, validation, device, args.batch_size, args.value_weight),
             "action_count_train": dict(Counter(action["kind"] for row in train_records for action in row["action_values"]
                                                if is_deployment_legal(row["observation"], action))),
             "history": history}
    (args.output_dir / f"{args.model}_training_summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in final.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
