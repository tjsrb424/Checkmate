from __future__ import annotations

import argparse
from pathlib import Path

from .arena_diagnostics import analyze_arena_payload, percent
from .model_arena import ModelArenaConfig, RandomModelPlayer, TorchModelPlayer, run_model_arena


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="maxPlies / adjudication margin arena sweep를 실행합니다.")
    parser.add_argument("--quick", action="store_true", help="작은 설정으로 빠르게 실행합니다.")
    parser.add_argument("--versions", nargs="*", default=None, help="동일 모델 sanity check 버전 목록")
    parser.add_argument("--maxPlies", nargs="*", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=0.0)
    parser.add_argument("--checkpointDir", default="../data/models/checkpoints")
    parser.add_argument("--ruleset", default="kakao-like", choices=["oetongsu-basic", "kakao-like", "kja-like"])
    return parser.parse_args(argv)


def pair_specs(versions: list[str] | None, checkpoint_dir: Path):
    specs: list[tuple[str, object, object]] = [
        ("random vs random", RandomModelPlayer(name="random-candidate", seed=1), RandomModelPlayer(name="random-champion", seed=1))
    ]
    for version in versions or []:
        path = checkpoint_dir / f"{version}.pt"
        if path.exists():
            specs.append((f"{version} vs {version}", TorchModelPlayer(path, name=f"{version}-candidate"), TorchModelPlayer(path, name=f"{version}-champion")))
        else:
            specs.append((f"{version} vs {version} (missing)", None, None))
    return specs


def warning_label(diagnostics) -> str:
    if diagnostics.paired_summary and diagnostics.paired_summary.get("sideDominatedPairs") == diagnostics.paired_summary.get("pairs"):
        return "side dominated"
    if diagnostics.warnings:
        return diagnostics.warnings[0]
    return "-"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_plies_values = args.maxPlies or ([20, 40] if args.quick else [50, 100, 150, 200])
    games = args.games if args.games is not None else (4 if args.quick else 20)
    simulations = args.simulations if args.simulations is not None else (1 if args.quick else 16)
    versions = args.versions
    if versions is None:
        versions = ["supervised_v0001", "az_iter_000002"] if args.quick else []

    print("maxPlies | pair | score_adj | checkmate | draw | CHO win | HAN win | warning")
    for max_plies in max_plies_values:
        for label, candidate, champion in pair_specs(versions, Path(args.checkpointDir)):
            if candidate is None or champion is None:
                print(f"{max_plies:<8} | {label} | skip | skip | skip | skip | skip | checkpoint missing")
                continue
            result = run_model_arena(
                candidate,
                champion,
                ModelArenaConfig(
                    games=games,
                    simulations=simulations,
                    max_plies=max_plies,
                    temperature=0.0,
                    ruleset_id=args.ruleset,
                    adjudication_draw_margin=args.adjudicationDrawMargin,
                ),
            )
            diagnostics = analyze_arena_payload(result.to_json(), path=f"{label}.json")
            score_adj = diagnostics.outcome_counts.get("score_adjudication", 0)
            print(
                f"{max_plies:<8} | {label} | {percent(score_adj, diagnostics.games)} | "
                f"{percent(diagnostics.outcome_counts.get('checkmate', 0), diagnostics.games)} | "
                f"{percent(diagnostics.draws, diagnostics.games)} | "
                f"{percent(diagnostics.winner_counts.get('CHO', 0), diagnostics.games)} | "
                f"{percent(diagnostics.winner_counts.get('HAN', 0), diagnostics.games)} | "
                f"{warning_label(diagnostics)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
