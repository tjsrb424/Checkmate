from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cheap_validation_gate import run_gate
from .train_alphazero import train_alphazero


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe small seed-only AlphaZero training differences.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--outputDir", default="../data/training/seed_probe")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--runCheapArena", action="store_true")
    parser.add_argument("--noArena", action="store_true")
    parser.add_argument("--cheapArenaGames", type=int, default=4)
    parser.add_argument("--cheapArenaSimulations", type=int, default=8)
    parser.add_argument("--cheapArenaMaxPlies", type=int, default=80)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=1.5)
    return parser.parse_args(argv)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    data = require_file(args.data, "training data")
    resume = require_file(args.resume, "resume checkpoint") if args.resume else None
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        output = output_dir / f"seed_{seed}.pt"
        metrics = train_alphazero(
            data=data,
            output=output,
            epochs=args.epochs,
            batch_size=args.batchSize,
            lr=args.lr,
            limit=args.limit,
            seed=seed,
            channels=args.channels,
            resume=resume,
            training_metadata={
                "source": "seed_sensitivity_probe",
                "model_version": output.stem,
                "candidate_version": output.stem,
                "resume_version": resume.stem if resume else None,
            },
        )
        row = {
            "seed": seed,
            "checkpoint": str(output),
            "metricsPath": str(output.with_name(f"{output.stem}_metrics.json")),
            "last": metrics["history"][-1] if metrics.get("history") else {},
        }
        if args.runCheapArena and not args.noArena and resume is not None:
            arena_output = output_dir / f"seed_{seed}_cheap_arena.json"
            row["cheapArena"] = run_gate(
                argparse.Namespace(
                    candidate=str(output),
                    champion=str(resume),
                    games=min(args.cheapArenaGames, 4),
                    simulations=min(args.cheapArenaSimulations, 8),
                    maxPlies=min(args.cheapArenaMaxPlies, 80),
                    adjudicationDrawMargin=args.adjudicationDrawMargin,
                    minScoreRate=0.25,
                    ruleset="kakao-like",
                    output=str(arena_output),
                )
            )
            arena_output.write_text(json.dumps(row["cheapArena"], indent=2), encoding="utf-8")
        rows.append(row)
    summary = {"data": str(data), "resume": str(resume) if resume else None, "limit": args.limit, "runs": rows}
    (output_dir / "seed_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "seed_sensitivity_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Seed Sensitivity Probe",
        "",
        f"- data: `{summary['data']}`",
        f"- resume: `{summary.get('resume') or '-'}`",
        f"- limit: `{summary['limit']}`",
        "",
        "| seed | checkpoint | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 | cheap_gate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["runs"]:
        last = row.get("last", {})
        cheap = row.get("cheapArena", {})
        lines.append(
            f"| {row['seed']} | `{row['checkpoint']}` | {float(last.get('val_total_loss', 0.0)):.6f} | "
            f"{float(last.get('val_policy_loss', 0.0)):.6f} | {float(last.get('val_value_loss', 0.0)):.6f} | "
            f"{float(last.get('val_policy_top1_against_argmax', 0.0)):.6f} | {cheap.get('status', 'skipped')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_probe(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print(render_markdown(summary))
    print(f"output written: {Path(args.outputDir) / 'seed_sensitivity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
