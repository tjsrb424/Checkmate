from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .alphazero_dataset import AlphaZeroJsonlDataset
from .alphazero_model import AlphaZeroNet
from .model import count_parameters


@dataclass
class AlphaZeroEpochMetrics:
    epoch: int
    policy_loss: float
    value_loss: float
    total_loss: float
    value_mae: float
    policy_top1_against_argmax: float
    val_policy_loss: float
    val_value_loss: float
    val_total_loss: float
    val_value_mae: float
    val_policy_top1_against_argmax: float


def train_alphazero(
    data: str | Path,
    output: str | Path,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 0.001,
    limit: int | None = None,
    seed: int = 1,
    channels: int = 64,
    resume: str | Path | None = None,
) -> dict:
    set_seed(seed)
    dataset = AlphaZeroJsonlDataset(data, limit=limit)
    train_dataset, val_dataset = split_dataset(dataset, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model = AlphaZeroNet(channels=int(checkpoint.get("channels", channels))).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = AlphaZeroNet(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_criterion = nn.MSELoss()
    history: list[AlphaZeroEpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, value_criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, value_criterion, device)
        history.append(
            AlphaZeroEpochMetrics(
                epoch=epoch,
                policy_loss=train_metrics["policy_loss"],
                value_loss=train_metrics["value_loss"],
                total_loss=train_metrics["total_loss"],
                value_mae=train_metrics["value_mae"],
                policy_top1_against_argmax=train_metrics["policy_top1_against_argmax"],
                val_policy_loss=val_metrics["policy_loss"],
                val_value_loss=val_metrics["value_loss"],
                val_total_loss=val_metrics["total_loss"],
                val_value_mae=val_metrics["value_mae"],
                val_policy_top1_against_argmax=val_metrics["policy_top1_against_argmax"],
            )
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": model.channels,
            "metrics": [asdict(row) for row in history],
        },
        output_path,
    )
    metrics = {
        "data": str(data),
        "output": str(output_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "limit": limit,
        "seed": seed,
        "channels": model.channels,
        "resume": str(resume) if resume else None,
        "device": str(device),
        "sample_count": len(dataset),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "parameter_count": count_parameters(model),
        "history": [asdict(row) for row in history],
    }
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_epoch(
    model: AlphaZeroNet,
    loader: DataLoader,
    value_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "value_mae": 0.0, "top1": 0.0}
    total_count = 0

    for features, policy_targets, value_targets in loader:
        features = features.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        policy_logits, value_pred = model(features)
        policy_loss = soft_cross_entropy(policy_logits, policy_targets)
        value_loss = value_criterion(value_pred, value_targets)
        loss = policy_loss + value_loss
        if training:
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_count += batch_size
        totals["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size
        totals["value_loss"] += float(value_loss.detach().cpu()) * batch_size
        totals["total_loss"] += float(loss.detach().cpu()) * batch_size
        totals["value_mae"] += float(torch.abs(value_pred.detach() - value_targets).sum().cpu())
        totals["top1"] += int((policy_logits.detach().argmax(dim=1) == policy_targets.argmax(dim=1)).sum().item())

    if total_count == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "value_mae": 0.0,
            "policy_top1_against_argmax": 0.0,
        }
    return {
        "policy_loss": totals["policy_loss"] / total_count,
        "value_loss": totals["value_loss"] / total_count,
        "total_loss": totals["total_loss"] / total_count,
        "value_mae": totals["value_mae"] / total_count,
        "policy_top1_against_argmax": totals["top1"] / total_count,
    }


def soft_cross_entropy(policy_logits: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(policy_logits, dim=1)
    return -(policy_targets * log_probs).sum(dim=1).mean()


def split_dataset(dataset: Dataset, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) == 1:
        return dataset, dataset
    val_count = max(1, int(round(len(dataset) * 0.2)))
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_count, val_count], generator=generator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first Oetongsu AlphaZero dual-head network.")
    parser.add_argument("--data", default="../data/selfplay/selfplay_latest.jsonl")
    parser.add_argument("--output", default="../data/models/az_model_latest.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = train_alphazero(
        data=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batchSize,
        lr=args.lr,
        limit=args.limit,
        seed=args.seed,
        channels=args.channels,
        resume=args.resume,
    )
    last = metrics["history"][-1]
    print("AlphaZero training complete")
    print(f"output: {metrics['output']}")
    print(f"total_loss: {last['total_loss']:.6f}")
    print(f"policy_loss: {last['policy_loss']:.6f}")
    print(f"value_loss: {last['value_loss']:.6f}")


if __name__ == "__main__":
    main()
