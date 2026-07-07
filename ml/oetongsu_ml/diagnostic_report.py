from __future__ import annotations

from pathlib import Path


DEFAULT_REPORT_PATH = Path("../data/training/a3_regression_diagnostics_report.md")
SECTION_ORDER = [
    "Summary",
    "Arena Result",
    "Model Output Comparison",
    "Self-play Dataset Quality",
    "Training Metrics",
    "Pairwise Arena Compare",
    "Policy/Value Sample Review",
    "Ablation Retrain",
    "Ablation Evaluation",
    "Suspected Causes",
    "Recommended Next Action",
]


def base_report() -> str:
    lines = ["# A3 Regression Diagnostics", ""]
    for section in SECTION_ORDER:
        lines.extend([f"## {section}", "", "_Pending._", ""])
    return "\n".join(lines).rstrip() + "\n"


def upsert_section(path: str | Path, section: str, body: str) -> None:
    if section not in SECTION_ORDER:
        raise ValueError(f"unknown report section: {section}")
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else base_report()
    lines = text.splitlines()
    header = f"## {section}"
    try:
        start = lines.index(header)
    except ValueError:
        text = text.rstrip() + f"\n\n{header}\n\n{body.strip()}\n"
        report_path.write_text(text, encoding="utf-8")
        return

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    replacement = [header, "", *body.strip().splitlines(), ""]
    new_lines = [*lines[:start], *replacement, *lines[end:]]
    report_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def append_bullets(title: str, rows: list[str]) -> str:
    lines = [title, ""]
    lines.extend(f"- {row}" for row in rows)
    return "\n".join(lines)
