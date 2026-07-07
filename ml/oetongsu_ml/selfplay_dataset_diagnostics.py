from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from .dataset import read_jsonl
from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .move_index import is_valid_policy_index
from .schema import TrainingPosition


@dataclass
class SelfPlayDiagnostics:
    path: str
    sample_count: int
    game_count: int
    value_counts: Counter[str] = field(default_factory=Counter)
    side_to_move_counts: Counter[str] = field(default_factory=Counter)
    winner_counts: Counter[str] = field(default_factory=Counter)
    policy_index_counts: Counter[int] = field(default_factory=Counter)
    invalid_policy_targets: int = 0
    empty_policy_targets: int = 0
    average_plies: float = 0.0
    max_plies_reached_rate: float | None = None
    summary_game_count: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Oetongsu self-play JSONL quality.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--compare")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def require_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing self-play JSONL: {resolved}")
    return resolved


def value_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "zero"


def policy_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = row.get("policy_target", [])
    return items if isinstance(items, list) else []


def analyze(path: str | Path, summary: str | Path | None = None) -> SelfPlayDiagnostics:
    resolved = require_path(path)
    rows = read_jsonl(resolved)
    game_ids = set()
    plies = []
    diagnostics = SelfPlayDiagnostics(path=str(resolved), sample_count=len(rows), game_count=0)

    for row in rows:
        if row.get("game_id") is not None:
            game_ids.add(str(row.get("game_id")))
        if isinstance(row.get("ply"), int):
            plies.append(int(row["ply"]))
        diagnostics.value_counts[value_bucket(row.get("value_target"))] += 1
        try:
            position = TrainingPosition.from_raw(row["position"])
            diagnostics.side_to_move_counts[position.turn] += 1
        except Exception:
            diagnostics.side_to_move_counts["invalid"] += 1
        winner = row.get("final_winner")
        diagnostics.winner_counts[str(winner) if winner is not None else "draw"] += 1

        items = policy_items(row)
        if not items:
            diagnostics.empty_policy_targets += 1
        for item in items:
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                diagnostics.invalid_policy_targets += 1
                continue
            if not is_valid_policy_index(index):
                diagnostics.invalid_policy_targets += 1
            else:
                diagnostics.policy_index_counts[index] += 1

    diagnostics.game_count = len(game_ids)
    diagnostics.average_plies = float(mean(plies)) if plies else 0.0

    if summary:
        summary_path = Path(summary)
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            diagnostics.summary_game_count = int(payload.get("games") or 0)
            summaries = payload.get("summaries") if isinstance(payload.get("summaries"), list) else []
            maxed = 0
            total = 0
            for item in summaries:
                if not isinstance(item, dict):
                    continue
                total += 1
                plies_value = int(item.get("plies") or 0)
                max_plies = int(item.get("maxPlies") or item.get("max_plies") or 0)
                outcome = str(item.get("outcome") or "")
                if (max_plies and plies_value >= max_plies) or outcome in {"draw_max_plies", "score_adjudication", "draw_score_adjudication"}:
                    maxed += 1
            diagnostics.max_plies_reached_rate = maxed / total if total else None
    return diagnostics


def render_markdown(primary: SelfPlayDiagnostics, comparison: SelfPlayDiagnostics | None = None) -> str:
    lines = [
        f"Path: `{primary.path}`",
        "",
        f"- sample_count: {primary.sample_count}",
        f"- game_count: {primary.game_count}",
        f"- summary_game_count: {primary.summary_game_count if primary.summary_game_count is not None else '-'}",
        f"- average_ply_index: {primary.average_plies:.2f}",
        f"- maxPlies_reached_rate: {primary.max_plies_reached_rate:.3f}" if primary.max_plies_reached_rate is not None else "- maxPlies_reached_rate: -",
        f"- invalid_policy_targets: {primary.invalid_policy_targets}",
        f"- empty_policy_targets: {primary.empty_policy_targets}",
        f"- value_distribution: {dict(primary.value_counts)}",
        f"- side_to_move_distribution: {dict(primary.side_to_move_counts)}",
        f"- winner_distribution_by_sample: {dict(primary.winner_counts)}",
        f"- unique_policy_target_indices: {len(primary.policy_index_counts)}",
    ]
    if comparison is not None:
        lines.extend(
            [
                "",
                "Comparison:",
                f"- compare_path: `{comparison.path}`",
                f"- sample_count_delta: {primary.sample_count - comparison.sample_count}",
                f"- game_count_delta: {primary.game_count - comparison.game_count}",
                f"- invalid_policy_target_delta: {primary.invalid_policy_targets - comparison.invalid_policy_targets}",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        primary = analyze(args.path, args.summary)
        comparison = analyze(args.compare) if args.compare else None
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    body = render_markdown(primary, comparison)
    print(body)
    if not args.no_report:
        upsert_section(args.report, "Self-play Dataset Quality", body)
        print(f"\nreport updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
