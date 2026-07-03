from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_arena import ModelArenaConfig, RandomModelPlayer, TorchModelPlayer, run_model_arena
from .model_registry import (
    get_latest_promoted,
    load_registry,
    promote_candidate,
    register_candidate,
    reject_candidate,
    save_registry,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry = load_registry(args.registry)
    candidate_version = args.version or (Path(args.candidate).stem if args.candidate else "random_candidate")
    champion_entry = get_latest_promoted(registry)
    champion_path = args.champion or (champion_entry["path"] if champion_entry else None)
    parent_version = champion_entry["version"] if champion_entry else None

    register_candidate(
        registry,
        version=candidate_version,
        path=args.candidate or "random",
        metadata_path=None,
        parent_version=parent_version,
    )

    candidate = RandomModelPlayer(name="candidate", seed=args.seed) if args.randomModel or not args.candidate else TorchModelPlayer(args.candidate, name="candidate")
    champion = RandomModelPlayer(name="champion", seed=args.seed + 1) if args.randomModel or not champion_path else TorchModelPlayer(champion_path, name="champion")

    arena_result = run_model_arena(
        candidate,
        champion,
        ModelArenaConfig(
            games=args.games,
            simulations=args.simulations,
            max_plies=args.maxPlies,
            temperature=args.temperature,
            seed=args.seed,
            promotion_threshold=args.threshold,
        ),
    )
    arena_json = arena_result.to_json()
    if arena_result.promoted:
        promote_candidate(registry, candidate_version, arena_json)
    else:
        reject_candidate(registry, candidate_version, arena_json)

    save_registry(args.registry, registry)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(arena_json, indent=2), encoding="utf-8")

    print("Model promotion check complete")
    print(f"candidate: {candidate_version}")
    print(f"promoted: {arena_result.promoted}")
    print(f"candidateScoreRate: {arena_result.candidateScoreRate:.4f}")
    print(f"registry: {args.registry}")
    print(f"result: {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model arena and update Oetongsu model registry.")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--champion", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--maxPlies", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--registry", default="../data/models/registry.json")
    parser.add_argument("--output", default="../data/models/arena/model_arena_latest.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--randomModel", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
