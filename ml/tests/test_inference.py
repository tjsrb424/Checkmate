import numpy as np
import torch

from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.inference import RandomPolicyValueModel, TorchPolicyValueModel
from oetongsu_ml.model import PolicyNet, ValueNet
from oetongsu_ml.schema import Piece, TrainingPosition


def sample_position():
    board = [[None for _ in range(9)] for _ in range(10)]
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    board[7][4] = Piece(side="CHO", kind="GUARD")
    board[1][4] = Piece(side="HAN", kind="GENERAL")
    board[6][0] = Piece(side="CHO", kind="SOLDIER")
    return TrainingPosition(board=board, turn="CHO")


def test_random_policy_value_model_output_shape():
    model = RandomPolicyValueModel(value=0.25, seed=3)
    policy, value = model.predict(sample_position())

    assert policy.shape == (POLICY_SIZE,)
    assert np.isclose(float(policy.sum()), 1.0)
    assert value == 0.25


def test_torch_policy_value_model_smoke(tmp_path):
    policy_path = tmp_path / "policy.pt"
    value_path = tmp_path / "value.pt"
    policy = PolicyNet(channels=4)
    value = ValueNet(channels=4)
    torch.save({"model_state": policy.state_dict(), "channels": 4}, policy_path)
    torch.save({"model_state": value.state_dict(), "channels": 4}, value_path)

    model = TorchPolicyValueModel(policy_path, value_path, device="cpu")
    policy_probs, value_prediction = model.predict(sample_position())

    assert policy_probs.shape == (POLICY_SIZE,)
    assert np.isclose(float(policy_probs.sum()), 1.0, atol=1e-5)
    assert -1.0 <= value_prediction <= 1.0
