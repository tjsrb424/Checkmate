import numpy as np
import torch

from oetongsu_ml.alphazero_model import AlphaZeroNet
from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.inference import RandomPolicyValueModel, TorchAlphaZeroModel
from oetongsu_ml.python_rules import create_initial_position


def test_random_policy_value_model_predict_batch_shape():
    model = RandomPolicyValueModel(seed=17)
    positions = [create_initial_position(), create_initial_position()]

    policies, values = model.predict_batch(positions)

    assert policies.shape == (2, POLICY_SIZE)
    assert values.shape == (2,)
    assert np.allclose(policies.sum(axis=1), np.ones((2,), dtype=np.float32))


def test_torch_alphazero_model_predict_batch_shape(tmp_path):
    checkpoint = tmp_path / "az.pt"
    network = AlphaZeroNet(channels=4)
    torch.save({"model_state": network.state_dict(), "channels": 4}, checkpoint)
    model = TorchAlphaZeroModel(checkpoint, device="cpu")
    positions = [create_initial_position(), create_initial_position(), create_initial_position()]

    policies, values = model.predict_batch(positions)

    assert policies.shape == (3, POLICY_SIZE)
    assert values.shape == (3,)
    assert np.allclose(policies.sum(axis=1), np.ones((3,), dtype=np.float32), atol=1e-5)
    assert np.all(values >= -1.0)
    assert np.all(values <= 1.0)
