from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import read_jsonl
from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .inference import TorchAlphaZeroModel
from .move_index import index_to_move, move_to_index
from .python_rules import generate_legal_moves
from .schema import Move, TrainingPosition


@dataclass
class ModelOutput:
    name: str
    policy: np.ndarray
    value: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export readable policy/value sample review for champion, previous, and candidate.")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--topK", type=int, default=5)
    parser.add_argument("--output", default="../data/training/a3_policy_value_sample_review.md")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def move_text(move: Move | None) -> str:
    if move is None:
        return "-"
    return f"({move.from_.x},{move.from_.y})->({move.to.x},{move.to.y})"


def target_indices(row: dict[str, Any]) -> set[int]:
    return {int(item["index"]) for item in row.get("policy_target", []) if isinstance(item, dict) and "index" in item}


def top_moves(policy: np.ndarray, top_k: int) -> list[tuple[int, float, Move | None]]:
    indices = np.argsort(policy)[::-1][:top_k]
    rows = []
    for index in indices:
        move = None
        try:
            move = index_to_move(int(index))
        except ValueError:
            move = None
        rows.append((int(index), float(policy[index]), move))
    return rows


def review_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_path = require_file(args.samples, "samples JSONL")
    models = {
        "champion": TorchAlphaZeroModel(require_file(args.champion, "champion checkpoint"), device="cpu"),
        "previous": TorchAlphaZeroModel(require_file(args.previous, "previous checkpoint"), device="cpu"),
        "candidate": TorchAlphaZeroModel(require_file(args.candidate, "candidate checkpoint"), device="cpu"),
    }
    rows = read_jsonl(sample_path)[: max(1, args.limit)]
    output_rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        position = TrainingPosition.from_raw(row["position"])
        legal_indices = {move_to_index(move) for move in generate_legal_moves(position)}
        outputs = []
        for name, model in models.items():
            policy, value = model.predict(position)
            outputs.append(ModelOutput(name=name, policy=policy, value=value))
        champion_top1 = int(outputs[0].policy.argmax())
        previous_top1 = int(outputs[1].policy.argmax())
        candidate_top1 = int(outputs[2].policy.argmax())
        policy_targets = target_indices(row)
        output_rows.append(
            {
                "sampleIndex": sample_index,
                "sideToMove": position.turn,
                "valueTarget": row.get("value_target"),
                "finalWinner": row.get("final_winner"),
                "legalMoveCount": len(legal_indices),
                "championValue": outputs[0].value,
                "previousValue": outputs[1].value,
                "candidateValue": outputs[2].value,
                "candidateValueDeltaVsChampion": outputs[2].value - outputs[0].value,
                "candidateValueDeltaVsPrevious": outputs[2].value - outputs[1].value,
                "top1AgreementChampionCandidate": champion_top1 == candidate_top1,
                "top1AgreementPreviousCandidate": previous_top1 == candidate_top1,
                "candidateTop1Legal": candidate_top1 in legal_indices,
                "policyTargetInCandidateTopK": bool(policy_targets & {index for index, _, _ in top_moves(outputs[2].policy, args.topK)}),
                "policyTargetMoves": [move_text(index_to_move(index)) for index in sorted(policy_targets)[: args.topK]],
                "topMoves": {output.name: top_moves(output.policy, args.topK) for output in outputs},
            }
        )
    output_rows.sort(key=lambda item: (item["top1AgreementChampionCandidate"], -abs(float(item["candidateValueDeltaVsChampion"]))))
    return output_rows


def render_markdown(rows: list[dict[str, Any]], top_k: int) -> str:
    lines = ["# A3 Policy/Value Sample Review", ""]
    for item in rows:
        lines.extend(
            [
                f"## Sample {item['sampleIndex']}",
                "",
                f"- side_to_move: {item['sideToMove']}",
                f"- value_target: {item['valueTarget']}",
                f"- final_winner: {item['finalWinner']}",
                f"- legal_move_count: {item['legalMoveCount']}",
                f"- values: champion={item['championValue']:.4f}, previous={item['previousValue']:.4f}, candidate={item['candidateValue']:.4f}",
                f"- candidate_value_delta_vs_champion: {item['candidateValueDeltaVsChampion']:.4f}",
                f"- candidate_value_delta_vs_previous: {item['candidateValueDeltaVsPrevious']:.4f}",
                f"- champion_candidate_top1_agree: {item['top1AgreementChampionCandidate']}",
                f"- previous_candidate_top1_agree: {item['top1AgreementPreviousCandidate']}",
                f"- candidate_top1_legal: {item['candidateTop1Legal']}",
                f"- policy_target_in_candidate_top{top_k}: {item['policyTargetInCandidateTopK']}",
                f"- policy_target_moves: {', '.join(item['policyTargetMoves']) or '-'}",
                "",
                "| model | rank | policy_index | prob | move |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for model_name, moves in item["topMoves"].items():
            for rank, (index, probability, move) in enumerate(moves, start=1):
                lines.append(f"| {model_name} | {rank} | {index} | {probability:.6f} | {move_text(move)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = review_rows(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    markdown = render_markdown(rows, args.topK)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"output written: {output}")
    if not args.no_report:
        upsert_section(args.report, "Policy/Value Sample Review", f"Sample review written to `{output}`.\n\n" + "\n".join(markdown.splitlines()[:80]))
        print(f"report updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
