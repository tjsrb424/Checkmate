from __future__ import annotations

import numpy as np

from .constants import BOARD_HEIGHT, BOARD_WIDTH, POLICY_SIZE
from .schema import Move


def move_to_index(move: Move | dict) -> int:
    parsed = Move.from_raw(move)
    validate_coordinate(parsed.from_.x, parsed.from_.y)
    validate_coordinate(parsed.to.x, parsed.to.y)
    return (((parsed.from_.x * BOARD_HEIGHT) + parsed.from_.y) * BOARD_WIDTH + parsed.to.x) * BOARD_HEIGHT + parsed.to.y


def index_to_move(index: int) -> Move:
    if not is_valid_policy_index(index):
        raise ValueError(f"policy index out of range: {index}")
    remaining = index
    to_y = remaining % BOARD_HEIGHT
    remaining //= BOARD_HEIGHT
    to_x = remaining % BOARD_WIDTH
    remaining //= BOARD_WIDTH
    from_y = remaining % BOARD_HEIGHT
    remaining //= BOARD_HEIGHT
    from_x = remaining
    return Move.from_raw({"from": {"x": from_x, "y": from_y}, "to": {"x": to_x, "y": to_y}})


def is_valid_policy_index(index: int) -> bool:
    return 0 <= int(index) < POLICY_SIZE


def legal_moves_to_mask(moves: list[Move | dict]) -> np.ndarray:
    mask = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for move in moves:
        mask[move_to_index(move)] = 1.0
    return mask


def validate_coordinate(x: int, y: int) -> None:
    if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
        raise ValueError(f"coordinate out of range: ({x}, {y})")

