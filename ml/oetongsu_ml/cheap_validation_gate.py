from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arena_compare import margin_summary, require_checkpoint
from .model_arena import ModelArenaConfig, TorchModelPlayer, run_model_arena


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small registry-free validation gate before promotion arena.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--maxPlies", type=int, default=80)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=1.5)
    parser.add_argument("--minScoreRate", type=float, default=0.25)
    parser.add_argument("--ruleset", choices=["oetongsu-basic", "kakao-like", "kja-like"], default="kakao-like")
    parser.add_argument("--output", default="../data/training/cheap_validation_gate.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    repeats = max(1, int(getattr(args, "repeats", 1)))
    if repeats > 1:
        return run_repeated_gate(args, repeats)
    return run_single_gate(args, int(getattr(args, "seed", 1)))


def run_single_gate(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    candidate = require_checkpoint(args.candidate, "candidate")
    champion = require_checkpoint(args.champion, "champion")
    result = run_model_arena(
        TorchModelPlayer(candidate, name=candidate.stem),
        TorchModelPlayer(champion, name="champion"),
        ModelArenaConfig(
            games=args.games,
            simulations=args.simulations,
            max_plies=args.maxPlies,
            temperature=0.0,
            seed=seed,
            promotion_threshold=args.minScoreRate,
            ruleset_id=args.ruleset,
            adjudication_draw_margin=args.adjudicationDrawMargin,
        ),
    ).to_json()
    payload = summarize_gate_result(
        result,
        candidate=str(candidate),
        champion=str(champion),
        simulations=args.simulations,
        max_plies=args.maxPlies,
        adjudication_draw_margin=args.adjudicationDrawMargin,
        min_score_rate=args.minScoreRate,
    )
    payload["seed"] = seed
    return payload


def run_repeated_gate(args: argparse.Namespace, repeats: int) -> dict[str, Any]:
    runs = [run_single_gate(args, int(args.seed) + index) for index in range(repeats)]
    pass_count = sum(1 for row in runs if row["status"] == "pass")
    fail_count = sum(1 for row in runs if row["status"] == "fail")
    aggregate_score_rate = sum(float(row["candidateScoreRate"]) for row in runs) / len(runs)
    status = "pass" if pass_count == repeats else "fail" if fail_count == repeats else "warn"
    warnings: list[str] = []
    if status == "warn":
        warnings.append("repeated cheap validation produced mixed pass/fail results")
    for row in runs:
        for warning in row.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
    first = runs[0]
    return {
        "candidate": first["candidate"],
        "champion": first["champion"],
        "status": status,
        "repeats": repeats,
        "seed": int(args.seed),
        "runs": [
            {
                "seed": row["seed"],
                "candidateScoreRate": row["candidateScoreRate"],
                "status": row["status"],
                "warnings": row.get("warnings", []),
            }
            for row in runs
        ],
        "passCount": pass_count,
        "failCount": fail_count,
        "aggregateScoreRate": aggregate_score_rate,
        "warnings": warnings,
        "settings": {
            "games": args.games,
            "simulations": args.simulations,
            "maxPlies": args.maxPlies,
            "adjudicationDrawMargin": args.adjudicationDrawMargin,
            "minScoreRate": args.minScoreRate,
        },
    }


def summarize_gate_result(
    arena_result: dict[str, Any],
    *,
    candidate: str,
    champion: str,
    simulations: int,
    max_plies: int,
    adjudication_draw_margin: float,
    min_score_rate: float,
) -> dict[str, Any]:
    margins = margin_summary(arena_result.get("gameSummaries", []), adjudication_draw_margin)
    paired = arena_result.get("pairedSummary") if isinstance(arena_result.get("pairedSummary"), dict) else {}
    warnings = gate_warnings(arena_result, margins, paired, max_plies)
    status = gate_status(arena_result, paired, min_score_rate)
    return {
        "candidate": candidate,
        "champion": champion,
        "status": status,
        "candidateScoreRate": float(arena_result.get("candidateScoreRate") or 0.0),
        "games": int(arena_result.get("games") or 0),
        "simulations": simulations,
        "maxPlies": max_plies,
        "adjudicationDrawMargin": adjudication_draw_margin,
        "minScoreRate": min_score_rate,
        "candidateWins": int(arena_result.get("candidateWins") or 0),
        "championWins": int(arena_result.get("championWins") or 0),
        "draws": int(arena_result.get("draws") or 0),
        "averagePlies": float(arena_result.get("averagePlies") or 0.0),
        "illegalMoves": int(arena_result.get("illegalMoves") or 0),
        "forfeits": int(arena_result.get("forfeits") or 0),
        "marginSummary": margins,
        "pairedSummary": paired,
        "warnings": warnings,
        "rawArenaResult": arena_result,
    }


def gate_status(arena_result: dict[str, Any], paired: dict[str, Any], min_score_rate: float) -> str:
    score_rate = float(arena_result.get("candidateScoreRate") or 0.0)
    illegal_moves = int(arena_result.get("illegalMoves") or 0)
    forfeits = int(arena_result.get("forfeits") or 0)
    pairs = max(0, int(paired.get("pairs") or 0))
    side_dominated = int(paired.get("sideDominatedPairs") or 0)
    side_ratio = side_dominated / pairs if pairs else 0.0
    if score_rate == 0.0 or illegal_moves > 0 or forfeits > 0 or side_ratio >= 0.75:
        return "fail"
    if score_rate < min_score_rate:
        return "fail"
    return "pass"


def gate_warnings(arena_result: dict[str, Any], margins: dict[str, Any], paired: dict[str, Any], max_plies: int) -> list[str]:
    warnings: list[str] = []
    summaries = arena_result.get("gameSummaries", [])
    if summaries and all(int(summary.get("plies") or 0) >= max_plies for summary in summaries if isinstance(summary, dict)):
        warnings.append("all games reached maxPlies")
    score_adjudications = [
        summary
        for summary in summaries
        if isinstance(summary, dict) and str(summary.get("outcome", "")).endswith("score_adjudication")
    ]
    if summaries and len(score_adjudications) / len(summaries) >= 0.5:
        warnings.append("score adjudication dominated the gate")
    if int(margins.get("outsideDrawMargin") or 0) > int(margins.get("withinDrawMargin") or 0):
        warnings.append("most adjudicated margins were outside draw margin")
    for item in paired.get("warnings", []) if isinstance(paired.get("warnings"), list) else []:
        warnings.append(str(item))
    return warnings


def render_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Cheap validation gate",
            f"status: {payload['status']}",
            f"candidateScoreRate: {float(payload.get('candidateScoreRate', payload.get('aggregateScoreRate', 0.0))):.3f}",
            f"wins/losses/draws: {payload.get('candidateWins', '-')}/{payload.get('championWins', '-')}/{payload.get('draws', '-')}",
            f"illegalMoves: {payload.get('illegalMoves', '-')}",
            f"forfeits: {payload.get('forfeits', '-')}",
            f"warnings: {', '.join(payload['warnings']) if payload['warnings'] else '-'}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_gate(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(render_summary(payload))
    print(f"output written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
