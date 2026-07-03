from __future__ import annotations

from .move_index import move_to_index
from .python_rules import apply_move, generate_legal_moves
from .schema import TrainingPosition


def perft(position: TrainingPosition | dict, depth: int) -> int:
    parsed = TrainingPosition.from_raw(position)
    if depth == 0:
        return 1
    total = 0
    for move in generate_legal_moves(parsed):
        total += perft(apply_move(parsed, move, append_history=False), depth - 1)
    return total


def perft_divide(position: TrainingPosition | dict, depth: int) -> list[dict]:
    parsed = TrainingPosition.from_raw(position)
    rows = []
    for move in generate_legal_moves(parsed):
        count = perft(apply_move(parsed, move, append_history=False), depth - 1)
        rows.append({"move": move.to_json(), "move_index": move_to_index(move), "nodes": count})
    return rows
