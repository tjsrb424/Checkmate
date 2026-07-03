from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .constants import POLICY_SIZE
from .encoder import encode_position
from .alphazero_model import AlphaZeroNet
from .model import PolicyNet, ValueNet
from .schema import TrainingPosition


class PolicyValueModel(ABC):
    @abstractmethod
    def predict(self, position: TrainingPosition | dict) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class RandomPolicyValueModel(PolicyValueModel):
    def __init__(self, value: float = 0.0, seed: int | None = 1) -> None:
        self.value = float(np.clip(value, -1.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def predict(self, position: TrainingPosition | dict) -> tuple[np.ndarray, float]:
        _ = TrainingPosition.from_raw(position)
        policy = self.rng.random(POLICY_SIZE, dtype=np.float32)
        total = float(policy.sum())
        if total <= 0:
            policy.fill(1.0 / POLICY_SIZE)
        else:
            policy /= total
        return policy, self.value


class TorchPolicyValueModel(PolicyValueModel):
    def __init__(
        self,
        policy_checkpoint: str | Path,
        value_checkpoint: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_state = torch.load(policy_checkpoint, map_location=self.device)
        value_state = torch.load(value_checkpoint, map_location=self.device)
        self.policy_model = PolicyNet(channels=int(policy_state.get("channels", 64))).to(self.device)
        self.value_model = ValueNet(channels=int(value_state.get("channels", 64))).to(self.device)
        self.policy_model.load_state_dict(policy_state["model_state"])
        self.value_model.load_state_dict(value_state["model_state"])
        self.policy_model.eval()
        self.value_model.eval()

    def predict(self, position: TrainingPosition | dict) -> tuple[np.ndarray, float]:
        parsed = TrainingPosition.from_raw(position)
        features = torch.from_numpy(encode_position(parsed)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_model(features)
            policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            value = float(self.value_model(features).squeeze().cpu())
        return policy, float(np.clip(value, -1.0, 1.0))


class TorchAlphaZeroModel(PolicyValueModel):
    def __init__(self, checkpoint: str | Path, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=self.device)
        self.model = AlphaZeroNet(channels=int(state.get("channels", 64))).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

    def predict(self, position: TrainingPosition | dict) -> tuple[np.ndarray, float]:
        parsed = TrainingPosition.from_raw(position)
        features = torch.from_numpy(encode_position(parsed)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(features)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            value_float = float(value.squeeze().cpu())
        return policy, float(np.clip(value_float, -1.0, 1.0))
