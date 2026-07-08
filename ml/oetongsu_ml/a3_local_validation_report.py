from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A3 local seed and cheap validation results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="../docs/ml/a3_local_validation_report.md")
    return parser.parse_args(argv)


def paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "seed": base / "data/training/seed_probe/seed_sensitivity_summary.json",
        "cheap_a3": base / "data/training/cheap_validation_az_iter_000003.json",
        "cheap_ablation": base / "data/training/cheap_validation_ablation_lr_0_001.json",
        "full_a3": base / "data/models/arena/az_iter_000003_arena.json",
        "full_ablation": base / "data/training/ablation_a3/evaluation_summary.json",
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def ablation_lr_row(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload.get("runs", []):
        if isinstance(row, dict) and row.get("candidateName") == "ablation_a3_lr_0_001":
            return row
    return {}


def last_metrics(row: dict[str, Any]) -> dict[str, float]:
    last = row.get("last", {}) if isinstance(row, dict) else {}
    return {key: float(last.get(key) or 0.0) for key in ("val_total_loss", "val_policy_loss", "val_value_loss", "val_policy_top1_against_argmax")}


def classify_seed(seed_summary: dict[str, Any]) -> tuple[str, str]:
    rows = [row for row in seed_summary.get("runs", []) if isinstance(row, dict)]
    if len(rows) < 2:
        return "insufficient", "Seed sensitivity cannot be judged because fewer than two seed runs are available."
    ranked = sorted(rows, key=lambda row: last_metrics(row)["val_total_loss"])
    best = ranked[0]
    worst = ranked[-1]
    delta = last_metrics(worst)["val_total_loss"] - last_metrics(best)["val_total_loss"]
    if delta >= 0.25:
        verdict = "high"
        text = f"Seed sensitivity is meaningful: best seed `{best.get('seed')}` beats worst seed `{worst.get('seed')}` by val_total_loss `{delta:.6f}`."
    elif delta >= 0.05:
        verdict = "moderate"
        text = f"Seed sensitivity is present but moderate: best seed `{best.get('seed')}` beats worst seed `{worst.get('seed')}` by val_total_loss `{delta:.6f}`."
    else:
        verdict = "low"
        text = f"Seed sensitivity is small in this probe: val_total_loss delta is `{delta:.6f}`."
    return verdict, text


def cheap_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status", "unknown"))


def full_summary_a3(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scoreRate": float(payload.get("candidateScoreRate") or 0.0),
        "wins": int(payload.get("candidateWins") or 0),
        "losses": int(payload.get("championWins") or 0),
        "draws": int(payload.get("draws") or 0),
        "averagePlies": float(payload.get("averagePlies") or 0.0),
    }


def cheap_full_consistency(cheap_a3: dict[str, Any], cheap_ablation: dict[str, Any], full_a3: dict[str, Any], full_ablation_row: dict[str, Any]) -> tuple[str, str]:
    a3_full_rate = float(full_a3.get("candidateScoreRate") or 0.0)
    ablation_full_rate = float(full_ablation_row.get("candidateScoreRate") or 0.0)
    a3_fail = cheap_status(cheap_a3) == "fail"
    ablation_pass = cheap_status(cheap_ablation) == "pass"
    if a3_fail and ablation_pass and a3_full_rate == 0.0 and ablation_full_rate >= 0.5:
        return "consistent", "Cheap gate matches the full-arena direction: A3 fails while ablation LR 0.001 passes."
    if cheap_status(cheap_a3) == cheap_status(cheap_ablation):
        return "inconsistent", "Cheap gate does not separate A3 and ablation, so games=4/sims=8/maxPlies=80 is too weak or noisy as a standalone gate."
    return "partial", "Cheap gate separates the candidates, but not cleanly enough to adopt without more repeats or a stronger confidence rule."


def stop_go(seed_verdict: str, consistency: str) -> tuple[str, str]:
    if consistency == "consistent" and seed_verdict in {"low", "moderate"}:
        return "STOP full A4; use cheap gate as required preflight", "Next sprint should harden cheap validation fail-fast and reporting before any micro-run."
    if consistency == "consistent":
        return "STOP full A4; add multi-seed validation before any micro-run", "Next sprint should require seed ensemble or multi-seed validation."
    return "STOP full A4; redesign cheap gate before relying on it", "Next sprint should increase paired games/repeats or add margin confidence gates."


def render_report(root: str | Path) -> str:
    resolved = paths(root)
    seed = load_json(resolved["seed"], "seed sensitivity summary")
    cheap_a3 = load_json(resolved["cheap_a3"], "cheap validation az_iter_000003")
    cheap_ablation = load_json(resolved["cheap_ablation"], "cheap validation ablation LR 0.001")
    full_a3 = load_json(resolved["full_a3"], "full A3 arena")
    full_ablation = load_json(resolved["full_ablation"], "full ablation evaluation")
    full_ablation_row = ablation_lr_row(full_ablation)
    seed_verdict, seed_text = classify_seed(seed)
    consistency, consistency_text = cheap_full_consistency(cheap_a3, cheap_ablation, full_a3, full_ablation_row)
    decision, next_sprint = stop_go(seed_verdict, consistency)
    seed_rows = [row for row in seed.get("runs", []) if isinstance(row, dict)]
    lines = [
        "# A3 Local Validation Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Seed verdict: `{seed_verdict}`. {seed_text}",
        f"- Cheap/full consistency: `{consistency}`. {consistency_text}",
        f"- Stop/Go decision: **{decision}**.",
        "",
        "## 2. Inputs",
        "",
    ]
    for label, path in resolved.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## 3. Seed Sensitivity Probe", "", "| seed | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 | checkpoint |", "| ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in seed_rows:
        metrics = last_metrics(row)
        lines.append(
            f"| {row.get('seed')} | {metrics['val_total_loss']:.6f} | {metrics['val_policy_loss']:.6f} | "
            f"{metrics['val_value_loss']:.6f} | {metrics['val_policy_top1_against_argmax']:.6f} | `{row.get('checkpoint')}` |"
        )
    lines.extend(
        [
            "",
            seed_text,
            "",
            "## 4. Cheap Validation: az_iter_000003",
            "",
            cheap_table(cheap_a3),
            "",
            "## 5. Cheap Validation: ablation LR 0.001",
            "",
            cheap_table(cheap_ablation),
            "",
            "## 6. Comparison With Full Arena Results",
            "",
            "| candidate | cheap_status | cheap_score_rate | full_score_rate | full_w/l/d | avg_plies |",
            "| --- | --- | ---: | ---: | --- | ---: |",
            f"| az_iter_000003 | {cheap_status(cheap_a3)} | {score_rate(cheap_a3):.3f} | {float(full_a3.get('candidateScoreRate') or 0.0):.3f} | {full_a3.get('candidateWins', 0)}/{full_a3.get('championWins', 0)}/{full_a3.get('draws', 0)} | {float(full_a3.get('averagePlies') or 0.0):.1f} |",
            f"| ablation_a3_lr_0_001 | {cheap_status(cheap_ablation)} | {score_rate(cheap_ablation):.3f} | {float(full_ablation_row.get('candidateScoreRate') or 0.0):.3f} | {full_ablation_row.get('candidateWins', 0)}/{full_ablation_row.get('championWins', 0)}/{full_ablation_row.get('draws', 0)} | {float(full_ablation_row.get('averagePlies') or 0.0):.1f} |",
            "",
            consistency_text,
            "",
            "## 7. Stop/Go Decision",
            "",
            f"- **{decision}**",
            "- RunPod full A4 remains blocked.",
            "- Champion registry mutation remains blocked.",
            "- Any micro-run must be explicitly bounded and preceded by cheap validation plus documented stop conditions.",
            "",
            "## 8. Recommended Next Sprint",
            "",
            f"- {next_sprint}",
            "- Add repeat-based cheap validation thresholds if single-run gate results are noisy.",
            "- Keep all generated checkpoints, JSONL, and local validation outputs out of Git.",
        ]
    )
    return "\n".join(lines) + "\n"


def score_rate(payload: dict[str, Any]) -> float:
    if "candidateScoreRate" in payload:
        return float(payload.get("candidateScoreRate") or 0.0)
    return float(payload.get("aggregateScoreRate") or 0.0)


def cheap_table(payload: dict[str, Any]) -> str:
    warnings = ", ".join(str(item) for item in payload.get("warnings", [])) or "-"
    return "\n".join(
        [
            "| status | score_rate | games | simulations | maxPlies | wins/losses/draws | warnings |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            f"| {cheap_status(payload)} | {score_rate(payload):.3f} | {payload.get('games', payload.get('settings', {}).get('games', '-'))} | "
            f"{payload.get('simulations', payload.get('settings', {}).get('simulations', '-'))} | {payload.get('maxPlies', payload.get('settings', {}).get('maxPlies', '-'))} | "
            f"{payload.get('candidateWins', '-')}/{payload.get('championWins', '-')}/{payload.get('draws', '-')} | {warnings} |",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        body = render_report(args.root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(body)
    print(f"output written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
