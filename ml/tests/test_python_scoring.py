from oetongsu_ml.python_rules import create_initial_position
from oetongsu_ml.schema import Piece
from oetongsu_ml.scoring import score_board_material, score_side_material


def test_initial_board_scores_with_han_deom():
    position = create_initial_position("inner-elephant", "inner-elephant")

    assert score_side_material(position.board, "CHO") == 72
    assert score_side_material(position.board, "HAN") == 73.5
    assert score_board_material(position.board) == {"cho": 72.0, "han": 73.5, "winner": "HAN", "margin": 1.5}


def test_removed_chariot_changes_winner():
    position = create_initial_position("inner-elephant", "inner-elephant")
    position.board[0][0] = None

    assert score_board_material(position.board) == {"cho": 72.0, "han": 60.5, "winner": "CHO", "margin": 11.5}


def test_han_deom_applies_on_bare_generals():
    board = [[None for _ in range(9)] for _ in range(10)]
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    board[1][4] = Piece(side="HAN", kind="GENERAL")

    assert score_board_material(board) == {"cho": 0.0, "han": 1.5, "winner": "HAN", "margin": 1.5}
