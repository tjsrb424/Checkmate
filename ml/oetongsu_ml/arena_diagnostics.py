from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    candidate_cho_games: int = 0
    candidate_cho_wins: int = 0
    candidate_han_games: int = 0
    candidate_han_wins: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="외통수 Arena 결과의 진영/점수 판정 편향을 진단합니다.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="단일 arena JSON 경로")
    source.add_argument("--glob", help="여러 arena JSON 경로 glob")
    parser.add_argument("--summary", default="../data/training/arena_diagnostics_summary.md", help="Markdown 요약 출력 경로")
    parser.add_argument("--no-summary", action="store_true", help="Markdown 요약 파일을 쓰지 않습니다.")
    return parser.parse_args(argv)


def load_arena_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"arena JSON object expected: {path}")
    return payload


def analyze_arena_payload(payload: dict[str, Any], path: str = "-") -> ArenaDiagnostics:
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

    plies_values: list[int] = []
    explicit_max_values: list[int] = []
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
        if candidate_side == "CHO":
            diagnostics.candidate_cho_games += 1
            if winner == "CHO":
                diagnostics.candidate_cho_wins += 1
        elif candidate_side == "HAN":
            diagnostics.candidate_han_games += 1
            if winner == "HAN":
                diagnostics.candidate_han_wins += 1

    diagnostics.inferred_max_plies = infer_max_plies(explicit_max_values, plies_values, diagnostics.average_plies)
    if diagnostics.inferred_max_plies > 0:
        diagnostics.max_plies_reached = sum(1 for plies in plies_values if plies >= diagnostics.inferred_max_plies)

    diagnostics.warnings = build_warnings(diagnostics)
    return diagnostics


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
        warnings.append("모든 또는 대부분의 대국이 maxPlies에서 점수 판정으로 끝났습니다.")
    if max_plies_rate >= 0.9:
        warnings.append("대부분의 대국이 최대 수순에 도달했습니다.")
    if cho_win_rate >= 0.9:
        warnings.append("모든 또는 대부분의 승자가 CHO입니다.")
    if han_win_rate >= 0.9:
        warnings.append("모든 또는 대부분의 승자가 HAN입니다.")
    if adjudication_rate >= 0.9 and diagnostics.illegal_moves == 0 and diagnostics.forfeits == 0:
        warnings.append("불법 수와 기권이 없는데 결과가 점수 판정에 집중되어 있습니다.")
    if (cho_win_rate >= 0.9 or han_win_rate >= 0.9) and adjudication_rate >= 0.9:
        warnings.append("현재 승격 대국은 모델 강도보다 진영/점수 판정 편향에 지배될 수 있습니다.")
    return warnings


def percent(part: int | float, total: int | float) -> str:
    if not total:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_diagnostics(diagnostics: ArenaDiagnostics) -> str:
    games = diagnostics.games
    lines = [
        "외통수 Arena 진단",
        "",
        f"파일: {Path(diagnostics.path).name}",
        f"게임 수: {games}",
        f"후보 승리: {diagnostics.candidate_wins}",
        f"챔피언 승리: {diagnostics.champion_wins}",
        f"무승부: {diagnostics.draws}",
        f"후보 점수율: {rate(diagnostics.candidate_score_rate)}",
        f"승격 여부: {'승격' if diagnostics.promoted else '미승격'}",
        "",
        "종료 유형",
    ]
    for outcome in ("score_adjudication", "checkmate", "illegal_move", "draw_max_plies", "loss_no_legal_moves", "draw_no_legal_moves"):
        count = diagnostics.outcome_counts.get(outcome, 0)
        lines.append(f"{outcome}: {count} / {games}, {percent(count, games)}")

    lines.extend(
        [
            "",
            "수순",
            f"averagePlies: {diagnostics.average_plies:.1f}",
            f"maxPlies 도달 비율: {diagnostics.max_plies_reached} / {games}, {percent(diagnostics.max_plies_reached, games)}",
            "",
            "진영 편향",
            f"CHO 승리: {diagnostics.winner_counts.get('CHO', 0)} / {games}, {percent(diagnostics.winner_counts.get('CHO', 0), games)}",
            f"HAN 승리: {diagnostics.winner_counts.get('HAN', 0)} / {games}, {percent(diagnostics.winner_counts.get('HAN', 0), games)}",
            (
                "후보가 CHO일 때 후보 승리: "
                f"{diagnostics.candidate_cho_wins} / {diagnostics.candidate_cho_games}, "
                f"{percent(diagnostics.candidate_cho_wins, diagnostics.candidate_cho_games)}"
            ),
            (
                "후보가 HAN일 때 후보 승리: "
                f"{diagnostics.candidate_han_wins} / {diagnostics.candidate_han_games}, "
                f"{percent(diagnostics.candidate_han_wins, diagnostics.candidate_han_games)}"
            ),
            "",
            "경고",
        ]
    )
    lines.extend([f"- {warning}" for warning in diagnostics.warnings] or ["- 특이 경고 없음"])
    return "\n".join(lines)


def render_markdown(diagnostics_list: list[ArenaDiagnostics]) -> str:
    lines = [
        "# Arena Diagnostics Summary",
        "",
        "## A3 전 권장 검토",
        "",
        "- `maxPlies`를 100에서 150 또는 200으로 늘려 점수 판정 집중도를 낮출지 검토",
        "- `promotionGames`는 최소 40 유지, 편향이 계속 보이면 증가 검토",
        "- `score_adjudication` 판정식과 HAN 덤/보정값이 의도대로 작동하는지 점검",
        "- maxPlies 도달 시 무승부 처리 또는 점수 차 margin 기준 도입 검토",
        "- 후보 점수율이 선후/초한 advantage를 충분히 상쇄하는지 동일 모델 sanity check로 확인",
        "",
    ]
    for diagnostics in diagnostics_list:
        lines.extend(
            [
                f"## {Path(diagnostics.path).name}",
                "",
                f"- 게임 수: {diagnostics.games}",
                f"- 후보 점수율: {rate(diagnostics.candidate_score_rate)}",
                f"- score_adjudication: {diagnostics.outcome_counts.get('score_adjudication', 0)} / {diagnostics.games}",
                f"- maxPlies 도달: {diagnostics.max_plies_reached} / {diagnostics.games}",
                f"- CHO 승리: {diagnostics.winner_counts.get('CHO', 0)} / {diagnostics.games}",
                f"- HAN 승리: {diagnostics.winner_counts.get('HAN', 0)} / {diagnostics.games}",
                "",
                "### 경고",
                "",
            ]
        )
        lines.extend([f"- {warning}" for warning in diagnostics.warnings] or ["- 특이 경고 없음"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.path:
        return [Path(args.path)]
    return [Path(path) for path in sorted(glob.glob(args.glob))]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    if not paths:
        print("진단할 arena JSON 파일을 찾지 못했습니다.")
        return 1

    diagnostics_list = []
    for path in paths:
        diagnostics = analyze_arena_payload(load_arena_payload(path), str(path))
        diagnostics_list.append(diagnostics)
        print(render_diagnostics(diagnostics))
        if path != paths[-1]:
            print("\n" + "━" * 40 + "\n")

    if not args.no_summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_markdown(diagnostics_list), encoding="utf-8")
        print(f"\nMarkdown 요약 저장: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
