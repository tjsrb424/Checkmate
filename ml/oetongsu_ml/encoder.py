from __future__ import annotations

import numpy as np

from .constants import BOARD_HEIGHT, BOARD_WIDTH, ENCODER_CHANNELS, PIECE_CHANNELS, PIECE_TO_CHANNEL, SIDES
from .schema import Piece, TrainingPosition


def encode_position(position: TrainingPosition | dict) -> np.ndarray:
    parsed = TrainingPosition.from_raw(position)
    validate_board(parsed.board)

    tensor = np.zeros((ENCODER_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for y, row in enumerate(parsed.board):
        for x, piece in enumerate(row):
            if piece is None:
                continue
            tensor[PIECE_TO_CHANNEL[(piece.side, piece.kind)], y, x] = 1.0

    tensor[PIECE_CHANNELS : PIECE_CHANNELS + len(SIDES), :, :] = side_to_move_planes(parsed.turn)
    return tensor


def decode_piece_planes(tensor: np.ndarray) -> list[list[Piece | None]]:
    if tensor.shape[0] < PIECE_CHANNELS or tensor.shape[1:] != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(f"expected tensor with at least {PIECE_CHANNELS} channels and board shape 10x9")

    board: list[list[Piece | None]] = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
    for (side, kind), channel in PIECE_TO_CHANNEL.items():
        ys, xs = np.where(tensor[channel] > 0.5)
        for y, x in zip(ys.tolist(), xs.tolist()):
            board[y][x] = Piece(side=side, kind=kind)
    return board


def side_to_move_planes(turn: str) -> np.ndarray:
    if turn not in SIDES:
        raise ValueError(f"invalid side to move: {turn}")
    planes = np.zeros((len(SIDES), BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    planes[SIDES.index(turn), :, :] = 1.0
    return planes


def validate_board(board: list[list[Piece | None]]) -> None:
    if len(board) != BOARD_HEIGHT:
        raise ValueError(f"board must have {BOARD_HEIGHT} rows")
    for row in board:
        if len(row) != BOARD_WIDTH:
            raise ValueError(f"board rows must have {BOARD_WIDTH} columns")

