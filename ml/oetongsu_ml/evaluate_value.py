from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ValueNet
from .train_value import sign_accuracy_count
from .value_dataset import ValueJsonlDataset


def evaluate_value(model_path: str | Path, data: str | Path, limit: int | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    channels = int(checkpoint.get("channels", 64))
    model = ValueNet(channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = ValueJsonlDataset(data, limit=limit)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.MSELoss()
    total_mse = 0.0
    total_abs_error = 0.0
    total_count = 0
    sign_correct = 0
    sample_prediction: dict | None = None

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            batch_size = targets.shape[0]
            total_mse += float(loss.cpu()) * batch_size
            total_abs_error += float(torch.abs(predictions - targets).sum().cpu())
            total_count += batch_size
            sign_correct += sign_accuracy_count(predictions, targets)

            if sample_prediction is None:
                sample_prediction = {
                    "prediction": float(predictions[0].cpu()),
                    "target": float(targets[0].cpu()),
                }

    metrics = {
        "mse": total_mse / total_count if total_count else 0.0,
        "mae": total_abs_error / total_count if total_count else 0.0,
        "sign_accuracy": sign_correct / total_count if total_count else 0.0,
        "sample_count": float(total_count),
    }
    print("Value evaluation complete")
    print(f"mse: {metrics['mse']:.6f}")
    print(f"mae: {metrics['mae']:.6f}")
    print(f"sign_accuracy: {metrics['sign_accuracy']:.4f}")
    if sample_prediction is not None:
        print(f"sample_prediction: {sample_prediction}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Oetongsu supervised value checkpoint.")
    parser.add_argument("--model", default="../data/models/value_net.pt")
    parser.add_argument("--data", default="../data/ml/value_samples.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_value(args.model, args.data, limit=args.limit)


if __name__ == "__main__":
    main()
