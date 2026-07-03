from __future__ import annotations

import torch
from torch import nn

from .constants import BOARD_HEIGHT, BOARD_WIDTH, ENCODER_CHANNELS, POLICY_SIZE


class PolicyNet(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(ENCODER_CHANNELS, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * BOARD_HEIGHT * BOARD_WIDTH, POLICY_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
