import torch

from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.model import PolicyNet, count_parameters


def test_policy_net_outputs_policy_logits():
    model = PolicyNet(channels=8)
    logits = model(torch.zeros((2, 16, 10, 9)))

    assert tuple(logits.shape) == (2, POLICY_SIZE)
    assert count_parameters(model) > POLICY_SIZE
