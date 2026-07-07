from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .arena_compare import margin_summary, require_checkpoint
from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .model_arena import ModelArenaConfig, TorchModelPlayer, run_model_arena


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate registry-free ablation checkpoints against a champion.")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--maxPlies", type=int, default=150)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=1.5)
    parser.add_argument("--ruleset", choices=["oetongsu-basic", "kakao-like", "kja-like"], default="kakao-like")
    parser.add_argument("--output", default="../data/training/ablation_a3/evaluation_summary.json")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def resolve_candidates(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def evaluate(args: argparse.Namespace) -> dict:
    champion = require_checkpoint(args.champion, "champion")
    candidates = [require_checkpoint(path, "candidate") for path in resolve_candidates(args.candidates)]
    rows = []
    for candidate in candidates:
        result = run_model_arena(
            TorchModelPlayer(candidate, name=candidate.stem),
            TorchModelPlayer(champion, name="champion"),
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
        rows.append(
            {
                "candidate": str(candidate),
                "candidateName": candidate.stem,
                "candidateWins": result["candidateWins"],
                "championWins": result["championWins"],
                "draws": result["draws"],
                "candidateScoreRate": result["candidateScoreRate"],
                "averagePlies": result["averagePlies"],
                "illegalMoves": result["illegalMoves"],
                "forfeits": result["forfeits"],
                "marginSummary": margin_summary(result["gameSummaries"], args.adjudicationDrawMargin),
                "pairedSummary": result.get("pairedSummary"),
            }
        )
    best = max(rows, key=lambda row: row["candidateScoreRate"], default=None)
    return {"champion": str(champion), "runs": rows, "bestCandidate": best}


def render_markdown(summary: dict) -> str:
    lines = [
        "# A3 Ablation Evaluation",
        "",
        "| candidate | score_rate | wins | losses | draws | avg_plies | margin_avg | margin_median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        margin = row["marginSummary"]
        lines.append(
            f"| {row['candidateName']} | {row['candidateScoreRate'] * 100:.1f}% | {row['candidateWins']} | "
            f"{row['championWins']} | {row['draws']} | {row['averagePlies']:.1f} | "
            f"{float(margin['avg'] or 0.0):.3f} | {float(margin['median'] or 0.0):.3f} |"
        )
    best = summary.get("bestCandidate")
    if best:
        lines.extend(["", f"Best candidate by quick score rate: `{best['candidateName']}` ({best['candidateScoreRate'] * 100:.1f}%)."])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    body = render_markdown(summary)
    (output.parent / "evaluation_summary.md").write_text(body, encoding="utf-8")
    print(body)
    print(f"output written: {output}")
    if not args.no_report:
        upsert_section(args.report, "Ablation Evaluation", body)
        print(f"report updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
