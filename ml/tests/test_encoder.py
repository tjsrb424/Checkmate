import numpy as np

from oetongsu_ml.constants import ENCODER_CHANNELS, PIECE_TO_CHANNEL
from oetongsu_ml.encoder import decode_piece_planes, encode_position, side_to_move_planes
from oetongsu_ml.schema import Piece, TrainingPosition


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def test_empty_board_piece_planes_sum_to_zero():
    tensor = encode_position(TrainingPosition(board=empty_board(), turn="CHO"))

    assert tensor.shape == (16, 10, 9)
    assert np.sum(tensor[:14]) == 0


def test_single_piece_sets_expected_channel_and_coordinate():
    board = empty_board()
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    tensor = encode_position(TrainingPosition(board=board, turn="HAN"))

    channel = PIECE_TO_CHANNEL[("CHO", "GENERAL")]
    assert tensor[channel, 8, 4] == 1
    assert np.sum(tensor[:14]) == 1


def test_side_to_move_planes_differ_for_cho_and_han():
    cho = side_to_move_planes("CHO")
    han = side_to_move_planes("HAN")

    assert cho.shape == (2, 10, 9)
    assert np.all(cho[0] == 1)
    assert np.all(cho[1] == 0)
    assert np.all(han[0] == 0)
    assert np.all(han[1] == 1)


def test_decode_piece_planes_round_trip():
    board = empty_board()
    board[0][5] = Piece(side="HAN", kind="CHARIOT")
    tensor = encode_position(TrainingPosition(board=board, turn="CHO"))

    decoded = decode_piece_planes(tensor)
    assert decoded[0][5] == Piece(side="HAN", kind="CHARIOT")
    assert decoded[1][1] is None


def test_encoder_shape_constant_matches_tensor_shape():
    tensor = encode_position({"board": empty_board(), "turn": "CHO"})

    assert tensor.shape[0] == ENCODER_CHANNELS

