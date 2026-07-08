from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("..")


@dataclass
class CandidateDecision:
    name: str
    lr: float | None
    score_rate: float
    wins: int
    losses: int
    draws: int
    average_plies: float
    margin_avg: float | None
    margin_median: float | None
    within_draw_margin: int | None
    outside_draw_margin: int | None
    val_total_loss: float | None
    val_policy_loss: float | None
    val_value_loss: float | None
    val_policy_top1: float | None
    warnings: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a cost-gated A3 ablation decision report.")
    parser.add_argument("--root", help="Artifact or repo root containing data/training and data/models.")
    parser.add_argument("--summary", help="Ablation evaluation_summary.json path.")
    parser.add_argument("--retrain", help="Ablation ablation_retrain_summary.json path.")
    parser.add_argument("--a3Arena", help="Original A3 arena JSON path.")
    parser.add_argument("--output", default="../docs/ml/a3_ablation_decision_report.md")
    return parser.parse_args(argv)


def paths_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.root) if args.root else DEFAULT_ROOT
    summary = Path(args.summary) if args.summary else root / "data/training/ablation_a3/evaluation_summary.json"
    retrain = Path(args.retrain) if args.retrain else root / "data/training/ablation_a3/ablation_retrain_summary.json"
    a3_arena = Path(args.a3Arena) if args.a3Arena else root / "data/models/arena/az_iter_000003_arena.json"
    return summary, retrain, a3_arena


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object: {path}")
    return payload


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def lr_from_name(name: str) -> float | None:
    match = re.search(r"lr_([0-9]+(?:_[0-9]+)*)", name)
    if not match:
        return None
    return as_float(match.group(1).replace("_", "."))


def candidate_label(path_or_name: str) -> str:
    return Path(path_or_name).stem


def retrain_by_lr(retrain_payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    rows: dict[float, dict[str, Any]] = {}
    for row in retrain_payload.get("runs", []):
        if not isinstance(row, dict):
            continue
        lr = as_float(row.get("lr"))
        if lr is not None:
            rows[lr] = row
    return rows


def build_candidates(evaluation_payload: dict[str, Any], retrain_payload: dict[str, Any]) -> list[CandidateDecision]:
    retrain_rows = retrain_by_lr(retrain_payload)
    candidates: list[CandidateDecision] = []
    for row in evaluation_payload.get("runs", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("candidateName") or candidate_label(str(row.get("candidate", "candidate"))))
        lr = lr_from_name(name)
        retrain = retrain_rows.get(lr) if lr is not None else None
        last = retrain.get("last", {}) if isinstance(retrain, dict) else {}
        margin = row.get("marginSummary", {}) if isinstance(row.get("marginSummary"), dict) else {}
        paired = row.get("pairedSummary", {}) if isinstance(row.get("pairedSummary"), dict) else {}
        candidates.append(
            CandidateDecision(
                name=name,
                lr=lr,
                score_rate=as_float(row.get("candidateScoreRate")) or 0.0,
                wins=as_int(row.get("candidateWins")),
                losses=as_int(row.get("championWins")),
                draws=as_int(row.get("draws")),
                average_plies=as_float(row.get("averagePlies")) or 0.0,
                margin_avg=as_float(margin.get("avg")),
                margin_median=as_float(margin.get("median")),
                within_draw_margin=as_int(margin.get("withinDrawMargin")) if "withinDrawMargin" in margin else None,
                outside_draw_margin=as_int(margin.get("outsideDrawMargin")) if "outsideDrawMargin" in margin else None,
                val_total_loss=as_float(last.get("val_total_loss")),
                val_policy_loss=as_float(last.get("val_policy_loss")),
                val_value_loss=as_float(last.get("val_value_loss")),
                val_policy_top1=as_float(last.get("val_policy_top1_against_argmax")),
                warnings=[str(item) for item in paired.get("warnings", [])] if isinstance(paired.get("warnings"), list) else [],
            )
        )
    return candidates


def best_candidate(candidates: list[CandidateDecision]) -> CandidateDecision | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -item.score_rate,
            item.margin_avg if item.margin_avg is not None else 999.0,
            item.val_total_loss if item.val_total_loss is not None else 999.0,
            item.name,
        ),
    )[0]


def baseline_summary(a3_payload: dict[str, Any]) -> dict[str, Any]:
    margins = [
        as_float((game.get("finalScore") or {}).get("margin"))
        for game in a3_payload.get("gameSummaries", [])
        if isinstance(game, dict)
    ]
    valid_margins = [item for item in margins if item is not None]
    return {
        "games": as_int(a3_payload.get("games")),
        "candidateWins": as_int(a3_payload.get("candidateWins")),
        "championWins": as_int(a3_payload.get("championWins")),
        "draws": as_int(a3_payload.get("draws")),
        "candidateScoreRate": as_float(a3_payload.get("candidateScoreRate")) or 0.0,
        "averagePlies": as_float(a3_payload.get("averagePlies")) or 0.0,
        "marginAvg": sum(valid_margins) / len(valid_margins) if valid_margins else None,
        "marginMedian": sorted(valid_margins)[len(valid_margins) // 2] if valid_margins else None,
        "pairedSummary": a3_payload.get("pairedSummary") if isinstance(a3_payload.get("pairedSummary"), dict) else {},
    }


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(
    candidates: list[CandidateDecision],
    best: CandidateDecision | None,
    baseline: dict[str, Any],
    inputs: dict[str, Path],
) -> str:
    raw_tops = [item for item in candidates if best and item.score_rate == max(candidate.score_rate for candidate in candidates)]
    raw_top_names = ", ".join(f"`{item.name}`" for item in raw_tops) if raw_tops else "-"
    baseline_rate = as_float(baseline.get("candidateScoreRate")) or 0.0
    improved = bool(best and best.score_rate > baseline_rate)
    full_run_decision = "STOP for full RunPod A4 now; GO only for local/cheap confirmation gates."

    lines = [
        "# A3 Ablation Decision Report",
        "",
        "## 1. Executive Summary",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Best actionable candidate: `{best.name}` (LR `{number(best.lr, 4)}`), selected by score rate, then lower margin, then lower validation loss.",
                f"- Raw top score-rate candidates: {raw_top_names}; this matters because the source JSON's first tied candidate is not necessarily the strongest operational choice.",
                f"- Original A3 baseline was `{percent(baseline_rate)}` ({baseline.get('candidateWins', 0)}-{baseline.get('championWins', 0)}-{baseline.get('draws', 0)}). Best ablation is `{percent(best.score_rate)}` ({best.wins}-{best.losses}-{best.draws}).",
                f"- A3 baseline improvement: {'yes' if improved else 'no'}, but every ablation game still reached `maxPlies=150`, so adjudication/draw handling remains a major uncertainty.",
                f"- Stop/go decision: **{full_run_decision}**",
            ]
        )
    else:
        lines.append("- No candidate rows were available; stop all expensive follow-up runs.")

    lines.extend(
        [
            "",
            "## 2. Inputs",
            "",
            f"- Evaluation summary: `{inputs['summary']}`",
            f"- Retrain summary: `{inputs['retrain']}`",
            f"- A3 arena baseline: `{inputs['a3Arena']}`",
            "- Artifact archive: `D:/OetongsuArtifacts/oetongsu_runpod_a3_ablation_artifacts.tgz`",
            "- Extracted artifact root: `D:/OetongsuArtifacts/a3_ablation_extract`",
            "",
            "## 3. A3 Failure Recap",
            "",
            f"- A3 arena result: candidate `{baseline.get('candidateWins', 0)}` wins, champion `{baseline.get('championWins', 0)}` wins, draws `{baseline.get('draws', 0)}`, score rate `{percent(baseline_rate)}`.",
            f"- Average plies: `{number(baseline.get('averagePlies'), 1)}`. The run was dominated by score adjudication at the maximum ply limit.",
            f"- Average margin: `{number(baseline.get('marginAvg'))}`. This is much larger than the `1.5` draw margin and indicates a real practical collapse under that evaluation setup.",
            "",
            "## 4. Ablation Retrain Metrics",
            "",
            "| LR | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(candidates, key=lambda candidate: candidate.lr or 0.0, reverse=True):
        lines.append(
            f"| {number(item.lr, 4)} | {number(item.val_total_loss, 6)} | {number(item.val_policy_loss, 6)} | "
            f"{number(item.val_value_loss, 6)} | {number(item.val_policy_top1, 6)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Ablation Evaluation Result",
            "",
            "| candidate | LR | score_rate | wins | losses | draws | avg_plies | margin_avg | margin_median | draw_margin_hits | warnings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in sorted(candidates, key=lambda candidate: (-(candidate.score_rate), candidate.name)):
        draw_hits = (
            "-"
            if item.within_draw_margin is None or item.outside_draw_margin is None
            else f"{item.within_draw_margin} within / {item.outside_draw_margin} outside"
        )
        warnings = "; ".join(item.warnings) if item.warnings else "-"
        lines.append(
            f"| `{item.name}` | {number(item.lr, 4)} | {percent(item.score_rate)} | {item.wins} | {item.losses} | "
            f"{item.draws} | {number(item.average_plies, 1)} | {number(item.margin_avg)} | {number(item.margin_median)} | {draw_hits} | {warnings} |"
        )

    lines.extend(
        [
            "",
            "## 6. Best Candidate",
            "",
        ]
    )
    if best:
        lines.extend(
            [
                f"- Choose LR `{number(best.lr, 4)}` / `{best.name}` as the best actionable candidate.",
                "- Reason: it ties the top score rate at `75.0%`, has the lower average/median margin among the tied candidates, and has the strongest validation losses from retraining.",
                "- Treat LR `0.0001` as a raw-score tie, not the operational winner; its validation losses are much worse despite the same short-match score rate.",
                "- Treat LR `0.0003` as rejected for now because it scored only `50.0%` and its paired summary warned that side advantage dominated the result.",
            ]
        )
    else:
        lines.append("- No best candidate.")

    lines.extend(
        [
            "",
            "## 7. Updated Root Cause Hypothesis",
            "",
            "- The failure is probably not a simple learning-rate-only issue. LR changes can recover short-evaluation score rate, but the original A3 full iteration still collapsed to `0.0%`.",
            "- The stronger hypothesis is a self-play target/data loop problem: max-plies-heavy games, score adjudication targets, replay composition, and champion/candidate policy drift can create candidates that look trained but fail promotion.",
            "- Arena/adjudication calibration remains a confounder because every ablation game ended at `150` plies and many wins/draws are margin-threshold decisions, not checkmates.",
            "",
            "## 8. Cost Review",
            "",
            "- This RunPod artifact reached packaging after about `04:59:05` wall time and already exceeded the intended exploratory budget.",
            "- Continuing with full A4-style RunPod runs before local confirmation risks paying again for another non-promotable candidate.",
            "- The cost-effective next step is analysis plus small deterministic validation: no new self-play, no champion registry mutation, and no long GPU run until gates pass.",
            "",
            "## 9. Stop/Go Decision",
            "",
            f"- Full RunPod A4: **STOP**.",
            f"- Local report and expanded evaluation: **GO**.",
            f"- Limited RunPod micro-run: **conditional GO** only if local gates confirm LR `{number(best.lr, 4) if best else '-'}` remains above `60%` score rate with stable margins.",
            "- Registry champion change: **STOP** until promotion criteria pass on a fresh, documented evaluation.",
            "",
            "Required gates before any full RunPod run:",
            "",
            "- Re-run the best candidate locally or in a capped job with at least `40` paired games.",
            "- Confirm score rate is above `60%`, not just a first-order tie from a 20-game sample.",
            "- Confirm paired summary is not side-dominated and illegal moves/forfeits remain zero.",
            "- Confirm average margin is below the failed A3 baseline and draw-margin hits are explainable.",
            "- Document expected wall time, expected cost, timeout, and autostop path before launch.",
            "",
            "## 10. Recommended Next Sprint",
            "",
            "- Recommended path: **Option B, self-play target/data redesign**, with LR `0.001` retained as the cheap validation probe.",
            "- Do not run A4 full training as the first action. First inspect value target construction, score-adjudication value labels, maxPlies-heavy samples, replay-buffer composition, and supervised/self-play data mixing.",
            "- If the local gates pass after those checks, run a deliberately capped A4 candidate with `learningRate=0.001`, `gamesPerIteration=100`, `simulations=48`, `arenaSimulations=48`, `maxPlies=150`, `adjudicationDrawMargin=1.5`, `trainEpochs=1`, and `batchSize=64`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary_path, retrain_path, a3_arena_path = paths_from_args(args)
        evaluation_payload = load_json(summary_path, "evaluation summary")
        retrain_payload = load_json(retrain_path, "retrain summary")
        a3_payload = load_json(a3_arena_path, "A3 arena")
        candidates = build_candidates(evaluation_payload, retrain_payload)
        best = best_candidate(candidates)
        baseline = baseline_summary(a3_payload)
        markdown = render_markdown(
            candidates,
            best,
            baseline,
            {"summary": summary_path, "retrain": retrain_path, "a3Arena": a3_arena_path},
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"output written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
