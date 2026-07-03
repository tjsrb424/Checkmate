from __future__ import annotations

from .schema import PieceKind, Side

JANGGI_MATERIAL_VALUES: dict[PieceKind, float] = {
    "GENERAL": 0,
    "GUARD": 3,
    "ELEPHANT": 3,
    "HORSE": 5,
    "CHARIOT": 13,
    "CANNON": 7,
    "SOLDIER": 2,
}


def score_side_material(board, side: Side) -> float:
    score = 1.5 if side == "HAN" else 0.0
    for row in board:
        for piece in row:
            if piece is not None and piece.side == side:
                score += JANGGI_MATERIAL_VALUES[piece.kind]
    return score


def score_board_material(board) -> dict:
    cho = score_side_material(board, "CHO")
    han = score_side_material(board, "HAN")
    winner: Side | str = "CHO" if cho > han else "HAN" if han > cho else "DRAW"
    return {"cho": cho, "han": han, "winner": winner}
