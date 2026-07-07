from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import POLICY_SIZE
from .dataset import read_jsonl
from .diagnostic_report import DEFAULT_REPORT_PATH, upsert_section
from .inference import TorchAlphaZeroModel
from .move_index import move_to_index
from .python_rules import generate_legal_moves
from .schema import TrainingPosition


EPSILON = 1e-12


@dataclass
class ModelStats:
    name: str
    path: str
    exists: bool
    parameter_count: int | None = None
    policy_entropy_mean: float | None = None
    legal_policy_mass_mean: float | None = None
    top1_legal_rate: float | None = None
    value_mean: float | None = None
    value_std: float | None = None
    value_min: float | None = None
    value_max: float | None = None


@dataclass
class PairStats:
    left: str
    right: str
    policy_kl_mean: float
    top1_agreement_rate: float
    value_delta_mean: float
    value_abs_delta_mean: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Oetongsu model outputs on shared self-play positions.")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def load_positions(path: str | Path, limit: int) -> list[TrainingPosition]:
    require_file(path, "samples JSONL")
    rows = read_jsonl(path)
    positions = [TrainingPosition.from_raw(row["position"]) for row in rows[: max(1, limit)]]
    if not positions:
        raise ValueError(f"no positions found in samples: {path}")
    return positions


def checkpoint_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        metric_count = checkpoint.get("parameter_count")
        if isinstance(metric_count, int):
            return metric_count
        state = checkpoint.get("model_state")
        if isinstance(state, dict):
            return int(sum(tensor.numel() for tensor in state.values() if hasattr(tensor, "numel")))
    return 0


def legal_mask(position: TrainingPosition) -> np.ndarray:
    mask = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for move in generate_legal_moves(position):
        mask[move_to_index(move)] = 1.0
    return mask


def policy_entropy(policy: np.ndarray) -> float:
    clipped = np.clip(policy, EPSILON, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def model_stats(name: str, path: Path, positions: list[TrainingPosition], legal_masks: np.ndarray, device: str) -> tuple[ModelStats, np.ndarray, np.ndarray]:
    model = TorchAlphaZeroModel(path, device=device)
    policies, values = model.predict_batch(positions)
    top1 = policies.argmax(axis=1)
    legal_mass = (policies * legal_masks).sum(axis=1)
    legal_top1 = legal_masks[np.arange(len(top1)), top1] > 0
    stats = ModelStats(
        name=name,
        path=str(path),
        exists=path.exists(),
        parameter_count=checkpoint_parameter_count(path),
        policy_entropy_mean=float(np.mean([policy_entropy(row) for row in policies])),
        legal_policy_mass_mean=float(np.mean(legal_mass)),
        top1_legal_rate=float(np.mean(legal_top1)),
        value_mean=float(np.mean(values)),
        value_std=float(np.std(values)),
        value_min=float(np.min(values)),
        value_max=float(np.max(values)),
    )
    return stats, policies, values


def pair_stats(left_name: str, right_name: str, left_policy: np.ndarray, right_policy: np.ndarray, left_values: np.ndarray, right_values: np.ndarray) -> PairStats:
    left = np.clip(left_policy, EPSILON, 1.0)
    right = np.clip(right_policy, EPSILON, 1.0)
    kl = (left * (np.log(left) - np.log(right))).sum(axis=1)
    left_top1 = left_policy.argmax(axis=1)
    right_top1 = right_policy.argmax(axis=1)
    value_delta = right_values - left_values
    return PairStats(
        left=left_name,
        right=right_name,
        policy_kl_mean=float(np.mean(kl)),
        top1_agreement_rate=float(np.mean(left_top1 == right_top1)),
        value_delta_mean=float(np.mean(value_delta)),
        value_abs_delta_mean=float(np.mean(np.abs(value_delta))),
    )


def run_diagnostics(args: argparse.Namespace) -> tuple[list[ModelStats], list[PairStats], int]:
    champion = require_file(args.champion, "champion checkpoint")
    previous = require_file(args.previous, "previous checkpoint")
    candidate = require_file(args.candidate, "candidate checkpoint")
    positions = load_positions(args.samples, args.limit)
    masks = np.stack([legal_mask(position) for position in positions])
    models = [
        ("champion", champion),
        ("previous", previous),
        ("candidate", candidate),
    ]
    stats: list[ModelStats] = []
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in models:
        item, policies, values = model_stats(name, path, positions, masks, args.device)
        stats.append(item)
        outputs[name] = (policies, values)
    champion_policy, champion_values = outputs["champion"]
    previous_policy, previous_values = outputs["previous"]
    candidate_policy, candidate_values = outputs["candidate"]
    pairs = [
        pair_stats("champion", "previous", champion_policy, previous_policy, champion_values, previous_values),
        pair_stats("champion", "candidate", champion_policy, candidate_policy, champion_values, candidate_values),
        pair_stats("previous", "candidate", previous_policy, candidate_policy, previous_values, candidate_values),
    ]
    return stats, pairs, len(positions)


def format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def render_markdown(stats: list[ModelStats], pairs: list[PairStats], sample_count: int) -> str:
    lines = [
        f"Compared positions: {sample_count}",
        "",
        "| model | exists | parameters | policy_entropy | legal_policy_mass | top1_legal_rate | value_mean | value_std | value_min | value_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in stats:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    str(item.exists),
                    str(item.parameter_count or 0),
                    format_float(item.policy_entropy_mean),
                    format_float(item.legal_policy_mass_mean),
                    format_float(item.top1_legal_rate),
                    format_float(item.value_mean),
                    format_float(item.value_std),
                    format_float(item.value_min),
                    format_float(item.value_max),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| pair | policy_kl_mean | top1_agreement_rate | value_delta_mean | value_abs_delta_mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in pairs:
        lines.append(
            f"| {pair.left} vs {pair.right} | {pair.policy_kl_mean:.6f} | {pair.top1_agreement_rate:.6f} | "
            f"{pair.value_delta_mean:.6f} | {pair.value_abs_delta_mean:.6f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats, pairs, sample_count = run_diagnostics(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    body = render_markdown(stats, pairs, sample_count)
    print(body)
    if not args.no_report:
        upsert_section(args.report, "Model Output Comparison", body)
        print(f"\nreport updated: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
