from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .model_registry import find_entry, get_latest_promoted, load_registry, promote_candidate, register_candidate, save_registry
from .train_alphazero import train_alphazero


def bootstrap_champion(
    data: str | Path = "../data/ml/az_supervised_samples.jsonl",
    output: str | Path = "../data/models/checkpoints/supervised_v0001.pt",
    registry_path: str | Path = "../data/models/registry.json",
    version: str = "supervised_v0001",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    channels: int = 64,
    limit: int | None = None,
    seed: int = 1,
    promote: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output)
    registry = load_registry(registry_path)
    existing = find_entry(registry, version)
    if existing is not None and not overwrite:
        raise ValueError(f"model version already exists: {version}")
    if existing is not None and overwrite:
        registry["models"] = [entry for entry in registry.get("models", []) if entry.get("version") != version]

    metrics = train_alphazero(
        data=data,
        output=output_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        limit=limit,
        seed=seed,
        channels=channels,
    )
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    registry_metrics = {
        "bootstrap": True,
        "source": "supervised",
        "data": str(data),
        "sampleCount": metrics["sample_count"],
        "epochs": epochs,
        "batchSize": batch_size,
        "channels": metrics["channels"],
        "latestTrain": (metrics.get("history") or [{}])[-1],
    }
    register_candidate(
        registry,
        version=version,
        path=str(output_path),
        metadata_path=str(metrics_path),
        parent_version=None,
        metrics=registry_metrics,
    )
    if promote:
        promote_candidate(
            registry,
            version,
            {
                "bootstrap": True,
                "source": "supervised",
                "promotedWithoutArena": True,
                "sampleCount": metrics["sample_count"],
            },
        )
    save_registry(registry_path, registry)
    latest = get_latest_promoted(registry)
    return {
        "version": version,
        "checkpoint": str(output_path),
        "metrics": str(metrics_path),
        "registry": str(registry_path),
        "promoted": promote,
        "latestPromotedVersion": latest["version"] if latest else None,
        "trainMetrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap an initial promoted AlphaZero champion from supervised JSONL data.")
    parser.add_argument("--data", default="../data/ml/az_supervised_samples.jsonl")
    parser.add_argument("--output", default="../data/models/checkpoints/supervised_v0001.pt")
    parser.add_argument("--registry", default="../data/models/registry.json")
    parser.add_argument("--version", default="supervised_v0001")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = bootstrap_champion(
        data=args.data,
        output=args.output,
        registry_path=args.registry,
        version=args.version,
        epochs=args.epochs,
        batch_size=args.batchSize,
        lr=args.lr,
        channels=args.channels,
        limit=args.limit,
        seed=args.seed,
        promote=args.promote,
        overwrite=args.overwrite,
    )
    print("Supervised champion bootstrap complete")
    print(f"version: {result['version']}")
    print(f"promoted: {result['promoted']}")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"metrics: {result['metrics']}")
    print(f"registry: {result['registry']}")
    print(f"latestPromotedVersion: {result['latestPromotedVersion']}")


if __name__ == "__main__":
    main()
