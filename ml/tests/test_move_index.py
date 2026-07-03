import numpy as np
import pytest

from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.move_index import index_to_move, is_valid_policy_index, legal_moves_to_mask, move_to_index
from oetongsu_ml.schema import Move, Position


def test_move_to_index_round_trip():
    move = Move(from_=Position(x=2, y=6), to=Position(x=4, y=5))

    assert index_to_move(move_to_index(move)) == move


def test_policy_index_bounds():
    assert is_valid_policy_index(0)
    assert is_valid_policy_index(POLICY_SIZE - 1)
    assert not is_valid_policy_index(-1)
    assert not is_valid_policy_index(POLICY_SIZE)


def test_index_to_move_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        index_to_move(POLICY_SIZE)


def test_legal_moves_to_mask_marks_only_legal_moves():
    moves = [
        {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}},
        {"from": {"x": 2, "y": 6}, "to": {"x": 3, "y": 6}},
    ]

    mask = legal_moves_to_mask(moves)

    assert mask.shape == (POLICY_SIZE,)
    assert mask.dtype == np.float32
    assert np.sum(mask) == 2
    for move in moves:
        assert mask[move_to_index(move)] == 1

