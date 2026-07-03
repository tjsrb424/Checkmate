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

    def predict_batch(self, positions: list[TrainingPosition | dict]) -> tuple[np.ndarray, np.ndarray]:
        predictions = [self.predict(position) for position in positions]
        if not predictions:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        policies = np.stack([policy for policy, _ in predictions]).astype(np.float32)
        values = np.array([value for _, value in predictions], dtype=np.float32)
        return policies, values


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

    def predict_batch(self, positions: list[TrainingPosition | dict]) -> tuple[np.ndarray, np.ndarray]:
        parsed_count = len([TrainingPosition.from_raw(position) for position in positions])
        if parsed_count == 0:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        policies = self.rng.random((parsed_count, POLICY_SIZE), dtype=np.float32)
        totals = policies.sum(axis=1, keepdims=True)
        policies = np.divide(policies, totals, out=np.full_like(policies, 1.0 / POLICY_SIZE), where=totals > 0)
        values = np.full((parsed_count,), self.value, dtype=np.float32)
        return policies.astype(np.float32), values


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

    def predict_batch(self, positions: list[TrainingPosition | dict]) -> tuple[np.ndarray, np.ndarray]:
        if not positions:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        parsed = [TrainingPosition.from_raw(position) for position in positions]
        features = torch.from_numpy(np.stack([encode_position(position) for position in parsed])).to(self.device)
        with torch.no_grad():
            logits = self.policy_model(features)
            values = self.value_model(features).view(-1)
            policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            value_array = values.cpu().numpy().astype(np.float32)
        return policies, np.clip(value_array, -1.0, 1.0).astype(np.float32)


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

    def predict_batch(self, positions: list[TrainingPosition | dict]) -> tuple[np.ndarray, np.ndarray]:
        if not positions:
            return np.zeros((0, POLICY_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        parsed = [TrainingPosition.from_raw(position) for position in positions]
        features = torch.from_numpy(np.stack([encode_position(position) for position in parsed])).to(self.device)
        with torch.no_grad():
            policy_logits, values = self.model(features)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy().astype(np.float32)
            value_array = values.view(-1).cpu().numpy().astype(np.float32)
        return policies, np.clip(value_array, -1.0, 1.0).astype(np.float32)
