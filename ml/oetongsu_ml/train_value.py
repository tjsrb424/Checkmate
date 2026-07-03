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

from .model import ValueNet, count_parameters
from .value_dataset import ValueJsonlDataset


@dataclass
class ValueEpochMetrics:
    epoch: int
    train_mse: float
    train_mae: float
    train_sign_accuracy: float
    val_mse: float
    val_mae: float
    val_sign_accuracy: float


def train_value(
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
    dataset = ValueJsonlDataset(data, limit=limit)
    train_dataset, val_dataset = split_dataset(dataset, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: list[ValueEpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append(
            ValueEpochMetrics(
                epoch=epoch,
                train_mse=train_metrics["mse"],
                train_mae=train_metrics["mae"],
                train_sign_accuracy=train_metrics["sign_accuracy"],
                val_mse=val_metrics["mse"],
                val_mae=val_metrics["mae"],
                val_sign_accuracy=val_metrics["sign_accuracy"],
            )
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": channels,
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
    model: ValueNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_mse = 0.0
    total_abs_error = 0.0
    total_count = 0
    sign_correct = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_mse += float(loss.detach().cpu()) * batch_size
        total_abs_error += float(torch.abs(predictions.detach() - targets).sum().cpu())
        total_count += batch_size
        sign_correct += sign_accuracy_count(predictions.detach(), targets)

    if total_count == 0:
        return {"mse": 0.0, "mae": 0.0, "sign_accuracy": 0.0}
    return {
        "mse": total_mse / total_count,
        "mae": total_abs_error / total_count,
        "sign_accuracy": sign_correct / total_count,
    }


def sign_accuracy_count(predictions: torch.Tensor, targets: torch.Tensor) -> int:
    predicted_sign = torch.sign(predictions)
    target_sign = torch.sign(targets)
    return int((predicted_sign == target_sign).sum().item())


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
    parser = argparse.ArgumentParser(description="Train the first Oetongsu supervised value network.")
    parser.add_argument("--data", default="../data/ml/value_samples.jsonl")
    parser.add_argument("--output", default="../data/models/value_net.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = train_value(
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
    print("Value training complete")
    print(f"output: {metrics['output']}")
    print(f"mse: {last['train_mse']:.6f}")
    print(f"mae: {last['train_mae']:.6f}")
    print(f"sign_accuracy: {last['train_sign_accuracy']:.4f}")


if __name__ == "__main__":
    main()
