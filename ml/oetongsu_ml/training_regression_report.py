from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section


METRIC_KEYS = [
    "policy_loss",
    "value_loss",
    "total_loss",
    "value_mae",
    "policy_top1_against_argmax",
    "val_policy_loss",
    "val_value_loss",
    "val_total_loss",
    "val_value_mae",
    "val_policy_top1_against_argmax",
]


@dataclass
class MetricsSnapshot:
    path: str
    exists: bool
    label: str
    sample_count: int | None = None
    train_count: int | None = None
    val_count: int | None = None
    lr: float | None = None
    epochs: int | None = None
    resume: str | None = None
    last: dict[str, float] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Oetongsu training metric JSON files.")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def label_for(path: Path) -> str:
    name = path.name
    suffix = "_metrics.json"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def load_metrics(path: str | Path) -> MetricsSnapshot:
    resolved = Path(path)
    if not resolved.exists():
        return MetricsSnapshot(path=str(resolved), exists=False, label=label_for(resolved))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    return MetricsSnapshot(
        path=str(resolved),
        exists=True,
        label=label_for(resolved),
        sample_count=as_int(payload.get("sample_count")),
        train_count=as_int(payload.get("train_count")),
        val_count=as_int(payload.get("val_count")),
        lr=as_float(payload.get("lr")),
        epochs=as_int(payload.get("epochs")),
        resume=str(payload.get("resume")) if payload.get("resume") else None,
        last={key: as_float(last.get(key)) or 0.0 for key in METRIC_KEYS},
    )


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_delta(current: MetricsSnapshot, previous: MetricsSnapshot, key: str) -> float | None:
    if current.last is None or previous.last is None:
        return None
    return current.last.get(key, 0.0) - previous.last.get(key, 0.0)


def suspected_causes(snapshots: list[MetricsSnapshot]) -> list[str]:
    existing = [item for item in snapshots if item.exists and item.last is not None]
    if len(existing) < 2:
        return ["insufficient metric files to classify regression causes"]
    previous = existing[-2]
    current = existing[-1]
    causes: list[str] = []
    for key in ("val_total_loss", "val_value_loss", "val_policy_loss"):
        delta = metric_delta(current, previous, key)
        if delta is not None and delta > 0.05:
            causes.append(f"{key} worsened by {delta:.4f}; training regression or data shift is plausible")
    if current.sample_count is not None and previous.sample_count is not None and current.sample_count < previous.sample_count:
        causes.append("sample_count dropped versus previous candidate; data shortage is plausible")
    if current.resume and "supervised_v0001" not in current.resume and "az_iter_000002" not in current.resume:
        causes.append(f"resume checkpoint is unexpected: {current.resume}")
    if not causes:
        causes.append("training metrics alone do not show an obvious regression; inspect model outputs, self-play distribution, and arena adjudication")
    return causes


def recommended_actions(causes: list[str]) -> list[str]:
    actions = [
        "Do not start A4 until az_iter_000003 checkpoint, self-play, and arena diagnostics are reviewed together.",
        "If model output entropy or legal policy mass collapses, rerun A3 with a lower learning rate and checkpoint validation before arena.",
        "If self-play values are one-sided or maxPlies-heavy, tune self-play/adjudication before adding more iterations.",
        "If arena margins are all above 1.5 at maxPlies 150, rerun a calibrated sweep before changing training hyperparameters.",
    ]
    if any("sample_count dropped" in cause for cause in causes):
        actions.insert(1, "Regenerate self-play with a fixed sample target before retraining.")
    return actions


def load_summary(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def render_markdown(snapshots: list[MetricsSnapshot], summary: dict[str, Any] | None = None) -> tuple[str, list[str], list[str]]:
    lines = [
        "| model | exists | sample_count | train_count | val_count | lr | epochs | resume | policy_loss | value_loss | total_loss | val_total_loss | value_mae | policy_top1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in snapshots:
        last = item.last or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    item.label,
                    str(item.exists),
                    value_text(item.sample_count),
                    value_text(item.train_count),
                    value_text(item.val_count),
                    value_text(item.lr),
                    value_text(item.epochs),
                    item.resume or "-",
                    value_text(last.get("policy_loss")),
                    value_text(last.get("value_loss")),
                    value_text(last.get("total_loss")),
                    value_text(last.get("val_total_loss")),
                    value_text(last.get("value_mae")),
                    value_text(last.get("policy_top1_against_argmax")),
                ]
            )
            + " |"
        )
    if summary:
        lines.extend(["", "AutoTrain summary:", f"- completedIterations: {summary.get('completedIterations', '-')}", f"- latestChampionVersion: {summary.get('latestChampionVersion', '-')}"])
    causes = suspected_causes(snapshots)
    actions = recommended_actions(causes)
    return "\n".join(lines), causes, actions


def value_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshots = [load_metrics(path) for path in args.metrics]
        summary = load_summary(args.summary)
    except (json.JSONDecodeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    body, causes, actions = render_markdown(snapshots, summary)
    print(body)
    print("\nSuspected causes:")
    for cause in causes:
        print(f"- {cause}")
    print("\nRecommended next action:")
    for action in actions:
        print(f"- {action}")
    if not args.no_report:
        upsert_section(args.report, "Training Metrics", body)
        upsert_section(args.report, "Suspected Causes", "\n".join(f"- {cause}" for cause in causes))
        upsert_section(args.report, "Recommended Next Action", "\n".join(f"- {action}" for action in actions))
        print(f"\nreport updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
