from __future__ import annotations

import torch
from torch import nn

from .constants import BOARD_HEIGHT, BOARD_WIDTH, ENCODER_CHANNELS, POLICY_SIZE


class AlphaZeroNet(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.backbone = nn.Sequential(
            nn.Conv2d(ENCODER_CHANNELS, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * BOARD_HEIGHT * BOARD_WIDTH, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * BOARD_HEIGHT * BOARD_WIDTH, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x.float())
        return self.policy_head(features), self.value_head(features)
