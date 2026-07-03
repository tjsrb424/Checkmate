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

from .model import PolicyNet, count_parameters
from .policy_dataset import PolicyJsonlDataset


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    train_top5: float
    val_loss: float
    val_top1: float
    val_top5: float


def train_policy(
    data: str | Path,
    output: str | Path,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 0.001,
    limit: int | None = None,
    seed: int = 1,
    channels: int = 64,
) -> dict:
    set_seed(seed)
    dataset = PolicyJsonlDataset(data, limit=limit)
    train_dataset, val_dataset = split_dataset(dataset, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_top1=train_metrics["top1"],
                train_top5=train_metrics["top5"],
                val_loss=val_metrics["loss"],
                val_top1=val_metrics["top1"],
                val_top5=val_metrics["top5"],
            )
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": channels,
            "policy_size": model(torch.zeros((1, 16, 10, 9), device=device)).shape[1],
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
        "channels": channels,
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
    model: PolicyNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    top1_count = 0
    top5_count = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        top1, top5 = accuracy_counts(logits.detach(), targets)
        top1_count += top1
        top5_count += top5

    if total_count == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0}
    return {
        "loss": total_loss / total_count,
        "top1": top1_count / total_count,
        "top5": top5_count / total_count,
    }


def accuracy_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    top1 = int((logits.argmax(dim=1) == targets).sum().item())
    k = min(5, logits.shape[1])
    top5_indices = logits.topk(k, dim=1).indices
    top5 = int((top5_indices == targets.unsqueeze(1)).any(dim=1).sum().item())
    return top1, top5


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
    parser = argparse.ArgumentParser(description="Train the first Oetongsu supervised policy network.")
    parser.add_argument("--data", default="../data/ml/policy_samples.jsonl")
    parser.add_argument("--output", default="../data/models/policy_net.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = train_policy(
        data=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batchSize,
        lr=args.lr,
        limit=args.limit,
        seed=args.seed,
        channels=args.channels,
    )
    last = metrics["history"][-1]
    print("Policy training complete")
    print(f"output: {metrics['output']}")
    print(f"loss: {last['train_loss']:.6f}")
    print(f"top1: {last['train_top1']:.4f}")
    print(f"top5: {last['train_top5']:.4f}")


if __name__ == "__main__":
    main()
