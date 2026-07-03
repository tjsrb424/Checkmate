from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import PolicyNet
from .move_index import index_to_move
from .policy_dataset import PolicyJsonlDataset
from .train_policy import accuracy_counts


def evaluate_policy(model_path: str | Path, data: str | Path, limit: int | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    channels = int(checkpoint.get("channels", 64))
    model = PolicyNet(channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = PolicyJsonlDataset(data, limit=limit)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    top1_count = 0
    top5_count = 0
    sample_prediction: dict | None = None

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            batch_size = targets.shape[0]
            total_loss += float(loss.cpu()) * batch_size
            total_count += batch_size
            top1, top5 = accuracy_counts(logits, targets)
            top1_count += top1
            top5_count += top5

            if sample_prediction is None:
                predicted_index = int(logits.argmax(dim=1)[0].cpu())
                target_index = int(targets[0].cpu())
                sample_prediction = {
                    "predicted_index": predicted_index,
                    "predicted_move": index_to_move(predicted_index).to_json(),
                    "target_index": target_index,
                    "target_move": index_to_move(target_index).to_json(),
                }

    metrics = {
        "loss": total_loss / total_count if total_count else 0.0,
        "top1": top1_count / total_count if total_count else 0.0,
        "top5": top5_count / total_count if total_count else 0.0,
        "sample_count": float(total_count),
    }
    print("Policy evaluation complete")
    print(f"loss: {metrics['loss']:.6f}")
    print(f"top1: {metrics['top1']:.4f}")
    print(f"top5: {metrics['top5']:.4f}")
    if sample_prediction is not None:
        print(f"sample_prediction: {sample_prediction}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Oetongsu supervised policy checkpoint.")
    parser.add_argument("--model", default="../data/models/policy_net.pt")
    parser.add_argument("--data", default="../data/ml/policy_samples.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_policy(args.model, args.data, limit=args.limit)


if __name__ == "__main__":
    main()
