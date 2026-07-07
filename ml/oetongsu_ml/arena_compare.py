from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arena_diagnostics import analyze_arena_payload, render_diagnostics
from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .model_arena import ModelArenaConfig, TorchModelPlayer, run_model_arena


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a registry-free pairwise arena comparison between two checkpoints.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--leftName", default="left")
    parser.add_argument("--rightName", default="right")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--maxPlies", type=int, default=150)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=1.5)
    parser.add_argument("--ruleset", choices=["oetongsu-basic", "kakao-like", "kja-like"], default="kakao-like")
    parser.add_argument("--output")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def require_checkpoint(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label} checkpoint: {resolved}")
    return resolved


def margin_summary(game_summaries: list[dict[str, Any]], draw_margin: float) -> dict[str, Any]:
    margins = []
    for summary in game_summaries:
        final_score = summary.get("finalScore")
        if isinstance(final_score, dict) and isinstance(final_score.get("margin"), (int, float)):
            margins.append(float(final_score["margin"]))
    if not margins:
        return {"count": 0, "min": None, "max": None, "avg": None, "median": None, "withinDrawMargin": 0, "outsideDrawMargin": 0}
    sorted_margins = sorted(margins)
    mid = len(sorted_margins) // 2
    median = sorted_margins[mid] if len(sorted_margins) % 2 else (sorted_margins[mid - 1] + sorted_margins[mid]) / 2
    return {
        "count": len(margins),
        "min": min(margins),
        "max": max(margins),
        "avg": sum(margins) / len(margins),
        "median": median,
        "withinDrawMargin": sum(1 for margin in margins if margin <= draw_margin),
        "outsideDrawMargin": sum(1 for margin in margins if margin > draw_margin),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    left = require_checkpoint(args.left, "left")
    right = require_checkpoint(args.right, "right")
    result = run_model_arena(
        TorchModelPlayer(left, name=args.leftName),
        TorchModelPlayer(right, name=args.rightName),
        ModelArenaConfig(
            games=args.games,
            simulations=args.simulations,
            max_plies=args.maxPlies,
            temperature=0.0,
            promotion_threshold=0.5,
            ruleset_id=args.ruleset,
            adjudication_draw_margin=args.adjudicationDrawMargin,
        ),
    ).to_json()
    payload = {
        "leftName": args.leftName,
        "rightName": args.rightName,
        "leftCheckpoint": str(left),
        "rightCheckpoint": str(right),
        "games": result["games"],
        "leftWins": result["candidateWins"],
        "rightWins": result["championWins"],
        "draws": result["draws"],
        "leftScoreRate": result["candidateScoreRate"],
        "rightScoreRate": result["championScoreRate"],
        "averagePlies": result["averagePlies"],
        "illegalMoves": result["illegalMoves"],
        "forfeits": result["forfeits"],
        "gameSummaries": result["gameSummaries"],
        "pairedSummary": result.get("pairedSummary"),
        "marginSummary": margin_summary(result["gameSummaries"], args.adjudicationDrawMargin),
        "rawArenaResult": result,
    }
    return payload


def render_summary(payload: dict[str, Any]) -> str:
    diagnostics = analyze_arena_payload(payload["rawArenaResult"], f"{payload['leftName']}_vs_{payload['rightName']}")
    lines = [
        "Pairwise arena compare",
        "",
        f"left: {payload['leftName']}",
        f"right: {payload['rightName']}",
        f"games: {payload['games']}",
        f"leftWins: {payload['leftWins']}",
        f"rightWins: {payload['rightWins']}",
        f"draws: {payload['draws']}",
        f"leftScoreRate: {payload['leftScoreRate'] * 100:.1f}%",
        f"rightScoreRate: {payload['rightScoreRate'] * 100:.1f}%",
        "",
        render_diagnostics(diagnostics),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_compare(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = render_summary(payload)
    print(summary)
    if args.output:
        print(f"\noutput written: {args.output}")
    if not args.no_report:
        upsert_section(args.report, "Pairwise Arena Compare", summary)
        print(f"report updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
