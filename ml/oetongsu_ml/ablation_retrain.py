from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .train_alphazero import train_alphazero


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registry-free ablation retraining on fixed A3 self-play data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--outputDir", default="../data/training/ablation_a3")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lrs", nargs="+", type=float, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def safe_lr_label(lr: float) -> str:
    return str(lr).replace(".", "_")


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def run_retrain(args: argparse.Namespace) -> list[dict]:
    data = require_file(args.data, "training data")
    resume = require_file(args.resume, "resume checkpoint") if args.resume else None
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lr in args.lrs:
        output = output_dir / f"ablation_a3_lr_{safe_lr_label(lr)}.pt"
        metrics = train_alphazero(
            data=data,
            output=output,
            epochs=args.epochs,
            batch_size=args.batchSize,
            lr=lr,
            limit=args.limit,
            seed=args.seed,
            channels=args.channels,
            resume=resume,
            training_metadata={
                "source": "ablation_retrain",
                "model_version": output.stem,
                "candidate_version": output.stem,
                "champion_version": "supervised_v0001" if resume and resume.name == "supervised_v0001.pt" else None,
                "resume_version": resume.stem if resume else None,
            },
        )
        last = metrics["history"][-1] if metrics.get("history") else {}
        rows.append({"lr": lr, "checkpoint": str(output), "metricsPath": str(output.with_name(f"{output.stem}_metrics.json")), "last": last})
    summary_path = output_dir / "ablation_retrain_summary.json"
    summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    markdown = render_markdown(rows)
    (output_dir / "ablation_retrain_summary.md").write_text(markdown, encoding="utf-8")
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# A3 Ablation Retrain",
        "",
        "| lr | checkpoint | total_loss | val_total_loss | policy_loss | val_policy_loss | value_loss | val_value_loss |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        last = row.get("last") or {}
        lines.append(
            f"| {row['lr']} | `{row['checkpoint']}` | {float(last.get('total_loss', 0.0)):.6f} | "
            f"{float(last.get('val_total_loss', 0.0)):.6f} | {float(last.get('policy_loss', 0.0)):.6f} | "
            f"{float(last.get('val_policy_loss', 0.0)):.6f} | {float(last.get('value_loss', 0.0)):.6f} | "
            f"{float(last.get('val_value_loss', 0.0)):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = run_retrain(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    body = render_markdown(rows)
    print(body)
    if not args.no_report:
        upsert_section(args.report, "Ablation Retrain", body)
        print(f"report updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
