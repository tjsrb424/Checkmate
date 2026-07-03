from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .dataset import load_value_samples
from .encoder import encode_position
from .schema import ValueTrainingSample


class ValueJsonlDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        samples = load_value_samples(path)
        self.samples: Sequence[ValueTrainingSample] = samples[:limit] if limit is not None else samples
        if len(self.samples) == 0:
            raise ValueError(f"value dataset is empty: {path}")

        for index, sample in enumerate(self.samples):
            if sample.value < -1.0 or sample.value > 1.0:
                raise ValueError(f"value target out of range at row {index}: {sample.value}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        features = torch.from_numpy(encode_position(sample.position))
        target = torch.tensor([sample.value], dtype=torch.float32)
        return features, target
