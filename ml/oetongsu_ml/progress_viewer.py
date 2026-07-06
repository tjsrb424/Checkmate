from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
DEFAULT_PROGRESS_PATH = Path("../data/training/progress.json")
BAR_WIDTH = 28


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="외통수 AlphaZero 학습 진행률을 한글/KST 기준으로 표시합니다.")
    parser.add_argument("--path", default=str(DEFAULT_PROGRESS_PATH), help="progress.json 경로")
    parser.add_argument("--watch", action="store_true", help="지정한 간격으로 진행률을 반복 표시합니다.")
    parser.add_argument("--interval", type=int, default=30, help="watch 갱신 간격(초)")
    parser.add_argument("--no-clear", action="store_true", help="watch 모드에서 화면을 지우지 않습니다.")
    return parser.parse_args(argv)


def load_progress(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"진행률 파일을 찾을 수 없습니다: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return None, f"진행률 JSON을 읽을 수 없습니다: {error}"
    if not isinstance(payload, dict):
        return None, "진행률 JSON 형식이 올바르지 않습니다."
    return payload, None


def format_kst(iso_text: Any) -> str:
    if not isinstance(iso_text, str) or not iso_text:
        return "-"
    normalized = iso_text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return "-"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def clamp_percent(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(100.0, float(value)))


def progress_bar(value: Any, width: int = BAR_WIDTH) -> str:
    percent = clamp_percent(value)
    filled = round((percent / 100) * width)
    return "█" * filled + "░" * (width - filled)


def value_text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.2f}" if not value.is_integer() else str(int(value))
    return str(value)


def ratio_text(current: Any, total: Any) -> str:
    left = value_text(current)
    right = value_text(total)
    if left == "-" and right == "-":
        return "-"
    return f"{left} / {right}"


def percent_text(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def threshold_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "55.0%"
    return percent_text(value)


def promotion_need_text(arena: dict[str, Any], threshold: float = 0.55) -> str:
    current_games = int(arena.get("currentGames") or 0)
    total_games = int(arena.get("totalGames") or 0)
    if total_games <= 0 or current_games >= total_games:
        return "-"

    remaining = max(0, total_games - current_games)
    candidate_wins = float(arena.get("candidateWins") or 0)
    draws = float(arena.get("draws") or 0)
    current_score = candidate_wins + draws * 0.5
    required_total = threshold * total_games
    required_remaining = max(0.0, required_total - current_score)
    return f"남은 {remaining}판에서 {required_remaining:.1f}점 이상"


def promotion_judgement(arena: dict[str, Any], threshold: float = 0.55) -> str:
    score_rate = arena.get("candidateScoreRate")
    if not isinstance(score_rate, (int, float)):
        return "승격 대국 결과를 기다리고 있습니다."
    if score_rate >= threshold:
        return "현재 기준 승격 기준 이상입니다. 단, 최종 결과는 전체 승격 대국 종료 후 확정됩니다."
    return "현재 기준 근소하게 승격 기준 아래입니다."


def line(label: str, value: str) -> str:
    return f"{label:<10}: {value}"


def render_progress(payload: dict[str, Any]) -> str:
    iteration = payload.get("iteration") if isinstance(payload.get("iteration"), dict) else {}
    self_play = payload.get("selfPlay") if isinstance(payload.get("selfPlay"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    arena = payload.get("arena") if isinstance(payload.get("arena"), dict) else {}
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    threshold = models.get("promotionThreshold") if isinstance(models.get("promotionThreshold"), (int, float)) else 0.55

    overall = clamp_percent(payload.get("overallPercent"))
    phase = clamp_percent(payload.get("phasePercent"))
    divider = "━" * 46
    rows = [
        divider,
        "외통수 AlphaZero 학습 진행 상황",
        divider,
        "",
        line("상태", value_text(payload.get("statusLabelKo") or status_label(payload.get("status")))),
        line("현재 단계", value_text(payload.get("phaseLabelKo") or phase_label(payload.get("phaseKey")))),
        line("메시지", value_text(payload.get("messageKo") or payload.get("message"))),
        "",
        line(
            "회차",
            f"{value_text(iteration.get('current'))} / {value_text(iteration.get('total'))}  완료 {value_text(iteration.get('completed'))}",
        ),
        line("전체 진행률", f"{overall:.1f}%  {progress_bar(overall)}"),
        line("단계 진행률", f"{phase:.1f}%  {progress_bar(phase)}"),
        "",
        "시간",
        line("시작", format_kst(payload.get("startedAt"))),
        line("최근 갱신", format_kst(payload.get("updatedAt"))),
        line("경과", value_text(payload.get("elapsedText"))),
        line("예상 남은", value_text(payload.get("etaText"))),
        "",
        "모델",
        line("현재 챔피언", value_text(models.get("championVersion") or models.get("latestPromotedVersion"))),
        line("후보 AI", value_text(models.get("candidateVersion"))),
        line("최근 승격", value_text(models.get("latestPromotedVersion"))),
        line("승격 기준", threshold_text(threshold)),
        "",
        "단계별",
        line("자기대국", ratio_text(self_play.get("currentGames"), self_play.get("totalGames"))),
        line("후보 학습", ratio_text(training.get("currentEpoch"), training.get("totalEpochs"))),
        line("승격 대국", ratio_text(arena.get("currentGames"), arena.get("totalGames"))),
        "",
        "승격 대국 중간 결과",
        line("후보 승리", value_text(arena.get("candidateWins"))),
        line("챔피언 승리", value_text(arena.get("championWins"))),
        line("무승부", value_text(arena.get("draws"))),
        line("후보 점수율", percent_text(arena.get("candidateScoreRate"))),
        line("불법 수", value_text(arena.get("illegalMoves"))),
        line("기권", value_text(arena.get("forfeits"))),
        line("남은 대국", value_text(remaining_games(arena))),
        line("필요 점수", promotion_need_text(arena, float(threshold))),
        line("판정", promotion_judgement(arena, float(threshold))),
        divider,
    ]
    return "\n".join(rows)


def remaining_games(arena: dict[str, Any]) -> int | str:
    current = arena.get("currentGames")
    total = arena.get("totalGames")
    if not isinstance(current, (int, float)) or not isinstance(total, (int, float)):
        return "-"
    return max(0, int(total) - int(current))


def status_label(status: Any) -> str:
    return {"running": "진행 중", "completed": "완료", "failed": "실패", "idle": "대기 중"}.get(status, "대기 중")


def phase_label(phase: Any) -> str:
    labels = {
        "selfplay": "자기대국 생성",
        "train": "후보 학습",
        "arena": "승격 대국",
        "package": "결과 정리",
        "completed": "완료",
        "failed": "실패",
    }
    return labels.get(phase, "대기 중")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def run_once(path: Path) -> int:
    payload, error = load_progress(path)
    if error:
        print(error)
        return 1
    assert payload is not None
    print(render_progress(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_output_encoding()
    args = parse_args(argv)
    path = Path(args.path)
    if not args.watch:
        return run_once(path)

    interval = max(1, int(args.interval))
    while True:
        if not args.no_clear:
            clear_screen()
        run_once(path)
        time.sleep(interval)


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
