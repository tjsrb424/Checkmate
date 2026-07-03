from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

from .constants import POLICY_SIZE
from .dataset import read_jsonl
from .encoder import encode_position
from .schema import TrainingPosition


class AlphaZeroJsonlDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        rows = read_jsonl(path)
        self.rows = rows[:limit] if limit is not None else rows
        if len(self.rows) == 0:
            raise ValueError(f"AlphaZero dataset is empty: {path}")

        for index, row in enumerate(self.rows):
            value = float(row["value_target"])
            if value < -1.0 or value > 1.0:
                raise ValueError(f"value_target out of range at row {index}: {value}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        features = torch.from_numpy(encode_position(TrainingPosition.from_raw(row["position"])))
        policy = torch.from_numpy(dense_policy_target(row.get("policy_target", [])))
        value = torch.tensor([float(row["value_target"])], dtype=torch.float32)
        return features, policy, value


def dense_policy_target(sparse_policy: list[dict]) -> np.ndarray:
    target = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for item in sparse_policy:
        index = int(item["index"])
        if index < 0 or index >= POLICY_SIZE:
            raise ValueError(f"policy index out of range: {index}")
        target[index] = float(item["prob"])

    total = float(target.sum())
    if total > 0:
        target /= total
    return target
