from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .encoder import encode_position
from .move_index import move_to_index
from .schema import PolicyTrainingSample, ValueTrainingSample, to_jsonable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_policy_samples(path: str | Path) -> list[PolicyTrainingSample]:
    return [PolicyTrainingSample.from_raw(row) for row in read_jsonl(path)]


def load_value_samples(path: str | Path) -> list[ValueTrainingSample]:
    return [ValueTrainingSample.from_raw(row) for row in read_jsonl(path)]


def policy_sample_to_arrays(sample: PolicyTrainingSample | dict) -> tuple[np.ndarray, np.ndarray]:
    parsed = PolicyTrainingSample.from_raw(sample)
    tensor = encode_position(parsed.position)
    target = np.array(parsed.move_index if parsed.move_index is not None else move_to_index(parsed.move), dtype=np.int64)
    return tensor, target

