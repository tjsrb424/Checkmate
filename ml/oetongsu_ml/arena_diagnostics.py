from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section


@dataclass
class MarginSummary:
    count: int = 0
    minimum: float = 0.0
    maximum: float = 0.0
    average: float = 0.0
    median: float = 0.0
    within_draw_margin: int = 0
    outside_draw_margin: int = 0


@dataclass
class ArenaDiagnostics:
    path: str
    games: int
    candidate_wins: int
    champion_wins: int
    draws: int
    candidate_score_rate: float
    champion_score_rate: float
    promoted: bool
    illegal_moves: int
    forfeits: int
    average_plies: float
    outcome_counts: Counter[str] = field(default_factory=Counter)
    max_plies_reached: int = 0
    inferred_max_plies: int = 0
    winner_counts: Counter[str] = field(default_factory=Counter)
    candidate_side_results: dict[str, Counter[str]] = field(default_factory=lambda: {"CHO": Counter(), "HAN": Counter()})
    candidate_cho_games: int = 0
    candidate_cho_wins: int = 0
    candidate_han_games: int = 0
    candidate_han_wins: int = 0
    margin_summary: MarginSummary = field(default_factory=MarginSummary)
    paired_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Oetongsu arena result bias, margins, and paired summaries.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="single arena JSON path")
    source.add_argument("--glob", help="arena JSON glob")
    parser.add_argument("--draw-margin", type=float, default=1.5)
    parser.add_argument("--summary", default="../data/training/arena_diagnostics_summary.md", help="Markdown summary output")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="A3 regression report output")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def load_arena_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"arena JSON object expected: {path}")
    return payload


def analyze_arena_payload(payload: dict[str, Any], path: str = "-", draw_margin: float = 1.5) -> ArenaDiagnostics:
    summaries = payload.get("gameSummaries") if isinstance(payload.get("gameSummaries"), list) else []
    games = int(payload.get("games") or len(summaries) or 0)
    diagnostics = ArenaDiagnostics(
        path=path,
        games=games,
        candidate_wins=int(payload.get("candidateWins") or 0),
        champion_wins=int(payload.get("championWins") or 0),
        draws=int(payload.get("draws") or 0),
        candidate_score_rate=float(payload.get("candidateScoreRate") or 0.0),
        champion_score_rate=float(payload.get("championScoreRate") or 0.0),
        promoted=bool(payload.get("promoted")),
        illegal_moves=int(payload.get("illegalMoves") or 0),
        forfeits=int(payload.get("forfeits") or 0),
        average_plies=float(payload.get("averagePlies") or 0.0),
    )
    paired_summary = payload.get("pairedSummary")
    if isinstance(paired_summary, dict):
        diagnostics.paired_summary = paired_summary

    plies_values: list[int] = []
    explicit_max_values: list[int] = []
    margins: list[float] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        outcome = str(summary.get("outcome") or "unknown")
        diagnostics.outcome_counts[outcome] += 1
        winner = summary.get("winner")
        if winner in ("CHO", "HAN"):
            diagnostics.winner_counts[str(winner)] += 1

        plies = int(summary.get("plies") or 0)
        plies_values.append(plies)
        if isinstance(summary.get("maxPlies"), int):
            explicit_max_values.append(int(summary["maxPlies"]))

        candidate_side = summary.get("candidateSide")
        if candidate_side in ("CHO", "HAN"):
            result = result_for_candidate(candidate_side, winner)
            diagnostics.candidate_side_results[candidate_side][result] += 1
            if candidate_side == "CHO":
                diagnostics.candidate_cho_games += 1
                if result == "win":
                    diagnostics.candidate_cho_wins += 1
            else:
                diagnostics.candidate_han_games += 1
                if result == "win":
                    diagnostics.candidate_han_wins += 1

        final_score = summary.get("finalScore")
        if isinstance(final_score, dict):
            margin = final_score.get("margin")
            if isinstance(margin, (int, float)):
                margins.append(float(margin))

    diagnostics.inferred_max_plies = infer_max_plies(explicit_max_values, plies_values, diagnostics.average_plies)
    if diagnostics.inferred_max_plies > 0:
        diagnostics.max_plies_reached = sum(1 for plies in plies_values if plies >= diagnostics.inferred_max_plies)
    diagnostics.margin_summary = summarize_margins(margins, draw_margin)
    diagnostics.warnings = build_warnings(diagnostics)
    return diagnostics


def result_for_candidate(candidate_side: str, winner: Any) -> str:
    if winner is None:
        return "draw"
    if winner == candidate_side:
        return "win"
    if winner in ("CHO", "HAN"):
        return "loss"
    return "unknown"


def summarize_margins(margins: list[float], draw_margin: float = 1.5) -> MarginSummary:
    if not margins:
        return MarginSummary()
    return MarginSummary(
        count=len(margins),
        minimum=min(margins),
        maximum=max(margins),
        average=sum(margins) / len(margins),
        median=float(median(margins)),
        within_draw_margin=sum(1 for value in margins if value <= draw_margin),
        outside_draw_margin=sum(1 for value in margins if value > draw_margin),
    )


def infer_max_plies(explicit_max_values: list[int], plies_values: list[int], average_plies: float) -> int:
    if explicit_max_values:
        return max(explicit_max_values)
    if plies_values:
        observed_max = max(plies_values)
        if average_plies and abs(average_plies - observed_max) < 0.001:
            return observed_max
        return observed_max
    return 0


def build_warnings(diagnostics: ArenaDiagnostics) -> list[str]:
    warnings: list[str] = []
    games = max(1, diagnostics.games)
    adjudication_rate = diagnostics.outcome_counts.get("score_adjudication", 0) / games
    max_plies_rate = diagnostics.max_plies_reached / games
    cho_win_rate = diagnostics.winner_counts.get("CHO", 0) / games
    han_win_rate = diagnostics.winner_counts.get("HAN", 0) / games

    if adjudication_rate >= 0.9:
        warnings.append("most games ended by score_adjudication")
    if max_plies_rate >= 0.9:
        warnings.append("most games reached maxPlies")
    if cho_win_rate >= 0.9:
        warnings.append("winner side is dominated by CHO")
    if han_win_rate >= 0.9:
        warnings.append("winner side is dominated by HAN")
    if adjudication_rate >= 0.9 and diagnostics.illegal_moves == 0 and diagnostics.forfeits == 0:
        warnings.append("result is adjudication-heavy without illegal moves or forfeits")
    if (cho_win_rate >= 0.9 or han_win_rate >= 0.9) and adjudication_rate >= 0.9:
        warnings.append("arena result may be dominated by side/scoring policy rather than model strength")
    pairs = int(diagnostics.paired_summary.get("pairs") or 0) if diagnostics.paired_summary else 0
    side_dominated = int(diagnostics.paired_summary.get("sideDominatedPairs") or 0) if diagnostics.paired_summary else 0
    if pairs > 0 and side_dominated / pairs >= 0.9:
        warnings.append("pairedSummary indicates side-dominated pairs")
    if diagnostics.margin_summary.count and diagnostics.margin_summary.outside_draw_margin == diagnostics.margin_summary.count:
        warnings.append("all final score margins are outside the draw margin")
    return warnings


def percent(part: int | float, total: int | float) -> str:
    if not total:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_diagnostics(diagnostics: ArenaDiagnostics) -> str:
    games = diagnostics.games
    margin = diagnostics.margin_summary
    lines = [
        "Oetongsu arena diagnostics",
        "",
        f"file: {Path(diagnostics.path).name}",
        f"games: {games}",
        f"candidateWins: {diagnostics.candidate_wins}",
        f"championWins: {diagnostics.champion_wins}",
        f"draws: {diagnostics.draws}",
        f"candidateScoreRate: {rate(diagnostics.candidate_score_rate)}",
        f"promoted: {diagnostics.promoted}",
        "",
        "outcomeCounts:",
    ]
    for outcome, count in sorted(diagnostics.outcome_counts.items()):
        lines.append(f"- {outcome}: {count} / {games}, {percent(count, games)}")

    lines.extend(
        [
            "",
            "plies:",
            f"- averagePlies: {diagnostics.average_plies:.1f}",
            f"- inferredMaxPlies: {diagnostics.inferred_max_plies}",
            f"- maxPliesReached: {diagnostics.max_plies_reached} / {games}, {percent(diagnostics.max_plies_reached, games)}",
            "",
            "winnerSideCounts:",
            f"- CHO: {diagnostics.winner_counts.get('CHO', 0)} / {games}, {percent(diagnostics.winner_counts.get('CHO', 0), games)}",
            f"- HAN: {diagnostics.winner_counts.get('HAN', 0)} / {games}, {percent(diagnostics.winner_counts.get('HAN', 0), games)}",
            "",
            "candidateSideResults:",
            f"- candidate as CHO: {dict(diagnostics.candidate_side_results['CHO'])}",
            f"- candidate as HAN: {dict(diagnostics.candidate_side_results['HAN'])}",
            "",
            "finalScoreMargins:",
            f"- count: {margin.count}",
            f"- min/max/avg/median: {margin.minimum:.3f} / {margin.maximum:.3f} / {margin.average:.3f} / {margin.median:.3f}",
            f"- margin <= draw margin: {margin.within_draw_margin}",
            f"- margin > draw margin: {margin.outside_draw_margin}",
            "",
            "pairedSummary:",
        ]
    )
    if diagnostics.paired_summary:
        for key in ("pairs", "sideDominatedPairs", "candidateDominatedPairs", "championDominatedPairs", "splitPairs", "drawPairs"):
            lines.append(f"- {key}: {diagnostics.paired_summary.get(key, '-')}")
    else:
        lines.append("- unavailable")
    lines.extend(["", "warnings:"])
    lines.extend([f"- {warning}" for warning in diagnostics.warnings] or ["- none"])
    return "\n".join(lines)


def render_markdown(diagnostics_list: list[ArenaDiagnostics]) -> str:
    lines = ["# Arena Diagnostics Summary", ""]
    for diagnostics in diagnostics_list:
        lines.extend([f"## {Path(diagnostics.path).name}", "", render_diagnostics(diagnostics), ""])
    return "\n".join(lines).rstrip() + "\n"


def render_report_section(diagnostics_list: list[ArenaDiagnostics]) -> str:
    if not diagnostics_list:
        return "_No arena diagnostics available._"
    return "\n\n".join(render_diagnostics(item) for item in diagnostics_list)


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.path:
        return [Path(args.path)]
    return [Path(path) for path in sorted(glob.glob(args.glob))]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    if not paths:
        print("ERROR: no arena JSON files matched")
        return 1

    diagnostics_list = []
    for path in paths:
        diagnostics = analyze_arena_payload(load_arena_payload(path), str(path), draw_margin=args.draw_margin)
        diagnostics_list.append(diagnostics)
        print(render_diagnostics(diagnostics))
        if path != paths[-1]:
            print("\n" + "-" * 60 + "\n")

    if not args.no_summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_markdown(diagnostics_list), encoding="utf-8")
        print(f"\nsummary written: {summary_path}")
    if not args.no_report:
        upsert_section(args.report, "Arena Result", render_report_section(diagnostics_list))
        print(f"report updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
