import torch

from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.model import PolicyNet, ValueNet, count_parameters


def test_policy_net_outputs_policy_logits():
    model = PolicyNet(channels=8)
    logits = model(torch.zeros((2, 16, 10, 9)))

    assert tuple(logits.shape) == (2, POLICY_SIZE)
    assert count_parameters(model) > POLICY_SIZE


def test_value_net_outputs_bounded_scalar():
    model = ValueNet(channels=8)
    values = model(torch.zeros((2, 16, 10, 9)))

    assert tuple(values.shape) == (2, 1)
    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)
