from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .dataset import load_policy_samples
from .encoder import encode_position
from .move_index import is_valid_policy_index
from .schema import PolicyTrainingSample


class PolicyJsonlDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        samples = load_policy_samples(path)
        self.samples: Sequence[PolicyTrainingSample] = samples[:limit] if limit is not None else samples
        if len(self.samples) == 0:
            raise ValueError(f"policy dataset is empty: {path}")

        for index, sample in enumerate(self.samples):
            if not is_valid_policy_index(sample.move_index):
                raise ValueError(f"invalid move_index at row {index}: {sample.move_index}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        features = torch.from_numpy(encode_position(sample.position))
        target = torch.tensor(sample.move_index, dtype=torch.long)
        return features, target
