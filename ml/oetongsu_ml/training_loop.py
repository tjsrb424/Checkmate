from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .alphazero_model import AlphaZeroNet
from .checkpoint import save_checkpoint
from .inference import RandomPolicyValueModel
from .self_play import SelfPlayConfig, play_self_play_game, self_play_samples_to_jsonl
from .train_alphazero import train_alphazero


@dataclass
class TrainingIterationConfig:
    games: int = 1
    max_plies: int = 4
    simulations: int = 4
    epochs: int = 1
    batch_size: int = 2
    lr: float = 0.001
    seed: int = 1
    channels: int = 8
    output_dir: str = "../data/selfplay"
    model_output: str = "../data/models/az_model_latest.pt"
    checkpoint_dir: str = "../data/models/checkpoints"


def run_training_iteration(config: TrainingIterationConfig | None = None) -> dict:
    cfg = config or TrainingIterationConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selfplay_path = output_dir / "selfplay_latest.jsonl"
    summary_path = output_dir / "selfplay_latest_summary.json"

    model = RandomPolicyValueModel(seed=cfg.seed)
    all_samples = []
    game_summaries = []
    for index in range(cfg.games):
        result = play_self_play_game(
            model,
            SelfPlayConfig(
                game_id=f"az-iter-{index + 1:06d}",
                max_plies=cfg.max_plies,
                mcts_simulations=cfg.simulations,
                temperature=0,
                seed=cfg.seed + index,
            ),
        )
        all_samples.extend(result.samples)
        game_summaries.append(result.to_summary())

    selfplay_path.write_text(self_play_samples_to_jsonl(all_samples), encoding="utf-8")
    summary_path.write_text(json.dumps({"games": cfg.games, "sample_count": len(all_samples), "summaries": game_summaries}, indent=2), encoding="utf-8")

    metrics = train_alphazero(
        data=selfplay_path,
        output=cfg.model_output,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        channels=cfg.channels,
    )
    checkpoint_model = AlphaZeroNet(channels=int(metrics["channels"]))
    checkpoint = torch.load(cfg.model_output, map_location="cpu")
    checkpoint_model.load_state_dict(checkpoint["model_state"])
    checkpoint_path, metadata_path = save_checkpoint(
        checkpoint_model,
        cfg.checkpoint_dir,
        {
            "trainingData": str(selfplay_path),
            "epochs": cfg.epochs,
            "games": cfg.games,
            "simulations": cfg.simulations,
            "metrics": metrics,
        },
    )
    return {
        "selfplay": str(selfplay_path),
        "selfplay_summary": str(summary_path),
        "model": cfg.model_output,
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "sample_count": len(all_samples),
        "metrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny AlphaZero self-play plus training iteration.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--maxPlies", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--outputDir", default="../data/selfplay")
    parser.add_argument("--modelOutput", default="../data/models/az_model_latest.pt")
    parser.add_argument("--checkpointDir", default="../data/models/checkpoints")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_training_iteration(
        TrainingIterationConfig(
            games=args.games,
            max_plies=args.maxPlies,
            simulations=args.simulations,
            epochs=args.epochs,
            batch_size=args.batchSize,
            lr=args.lr,
            seed=args.seed,
            channels=args.channels,
            output_dir=args.outputDir,
            model_output=args.modelOutput,
            checkpoint_dir=args.checkpointDir,
        )
    )
    print("AlphaZero training iteration complete")
    print(f"samples: {result['sample_count']}")
    print(f"model: {result['model']}")
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
