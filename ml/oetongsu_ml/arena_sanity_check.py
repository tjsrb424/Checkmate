from __future__ import annotations

import argparse
from pathlib import Path

from .arena_diagnostics import analyze_arena_payload, render_diagnostics
from .model_arena import ModelArenaConfig, RandomModelPlayer, TorchModelPlayer, run_model_arena


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="동일 모델/무작위 모델 arena side-bias sanity check를 실행합니다.")
    parser.add_argument("--quick", action="store_true", help="작은 설정으로 빠르게 실행합니다.")
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--maxPlies", type=int, default=None)
    parser.add_argument("--ruleset", default="kakao-like", choices=["oetongsu-basic", "kakao-like", "kja-like"])
    parser.add_argument("--checkpointDir", default="../data/models/checkpoints")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ModelArenaConfig:
    return ModelArenaConfig(
        games=args.games if args.games is not None else (4 if args.quick else 20),
        simulations=args.simulations if args.simulations is not None else (1 if args.quick else 16),
        max_plies=args.maxPlies if args.maxPlies is not None else (20 if args.quick else 100),
        temperature=0.0,
        ruleset_id=args.ruleset,
        promotion_threshold=0.55,
    )


def run_pair(name: str, candidate, champion, config: ModelArenaConfig) -> None:
    print(f"\n=== {name} ===")
    result = run_model_arena(candidate, champion, config)
    diagnostics = analyze_arena_payload(result.to_json(), path=f"{name}.json")
    print(render_diagnostics(diagnostics))


def maybe_checkpoint_pair(version: str, checkpoint_dir: Path, config: ModelArenaConfig) -> None:
    path = checkpoint_dir / f"{version}.pt"
    if not path.exists():
        print(f"\n=== {version} vs {version} ===")
        print(f"건너뜀: checkpoint 없음 ({path})")
        return
    run_pair(
        f"{version} vs {version}",
        TorchModelPlayer(path, name=f"{version}-candidate"),
        TorchModelPlayer(path, name=f"{version}-champion"),
        config,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    checkpoint_dir = Path(args.checkpointDir)

    run_pair(
        "random vs random",
        RandomModelPlayer(name="random-candidate", seed=1),
        RandomModelPlayer(name="random-champion", seed=1),
        config,
    )
    maybe_checkpoint_pair("supervised_v0001", checkpoint_dir, config)
    maybe_checkpoint_pair("az_iter_000002", checkpoint_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
