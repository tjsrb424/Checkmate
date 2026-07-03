from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import RandomPolicyValueModel, TorchPolicyValueModel
from .self_play import SelfPlayConfig, play_self_play_game, self_play_samples_to_jsonl


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_name(output_path.stem.replace("selfplay", "selfplay_summary") + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model = create_model(args)
    all_samples = []
    summaries = []
    for game_index in range(args.games):
        game_id = f"{args.gameIdPrefix}-{game_index + 1:06d}"
        config = SelfPlayConfig(
            game_id=game_id,
            max_plies=args.maxPlies,
            mcts_simulations=args.simulations,
            temperature=args.temperature,
            temperature_drop_ply=args.temperatureDropPly,
            seed=args.seed + game_index if args.seed is not None else None,
        )
        result = play_self_play_game(model, config)
        all_samples.extend(result.samples)
        summaries.append(result.to_summary())

    output_path.write_text(self_play_samples_to_jsonl(all_samples), encoding="utf-8")
    summary = {
        "games": args.games,
        "sample_count": len(all_samples),
        "output": str(output_path),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Self-play generation complete")
    print(f"games: {args.games}")
    print(f"samples: {len(all_samples)}")
    print(f"output: {output_path}")
    print(f"summary: {summary_path}")


def create_model(args: argparse.Namespace):
    if args.randomModel or not (args.policyModel and args.valueModel):
        return RandomPolicyValueModel(seed=args.seed)
    return TorchPolicyValueModel(args.policyModel, args.valueModel)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Oetongsu self-play JSONL samples.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--output", default="../data/selfplay/selfplay_000001.jsonl")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--policyModel", default=None)
    parser.add_argument("--valueModel", default=None)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--maxPlies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--randomModel", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatureDropPly", type=int, default=20)
    parser.add_argument("--gameIdPrefix", default="selfplay")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
