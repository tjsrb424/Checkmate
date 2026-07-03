from oetongsu_ml.python_rules import (
    apply_move,
    board_position_key,
    count_position_occurrences,
    create_initial_position,
    generate_legal_moves,
    position_key,
    would_repeat_position,
)
from oetongsu_ml.schema import Move, Piece, Position, TrainingPosition


def tiny_repeat_position():
    board = [[None for _ in range(9)] for _ in range(10)]
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    board[1][4] = Piece(side="HAN", kind="GENERAL")
    board[7][4] = Piece(side="CHO", kind="GUARD")
    board[6][0] = Piece(side="CHO", kind="SOLDIER")
    position = TrainingPosition(board=board, turn="CHO")
    return TrainingPosition(board=board, turn="CHO", position_history=[position_key(position)])


def test_create_initial_position_has_position_history():
    position = create_initial_position()

    assert len(position.position_history) == 1
    assert position.position_history[0] == position_key(position)


def test_training_position_reads_camel_and_snake_position_history():
    raw = create_initial_position().to_json()
    raw["positionHistory"] = ["camel"]
    raw.pop("position_history")

    parsed = TrainingPosition.from_raw(raw)
    assert parsed.position_history == ["camel"]

    raw["position_history"] = ["snake"]
    parsed = TrainingPosition.from_raw(raw)
    assert parsed.position_history == ["snake"]


def test_apply_move_appends_history_and_position_history_when_append_history_true():
    position = tiny_repeat_position()
    move = Move(Position(0, 6), Position(0, 5))

    next_position = apply_move(position, move, append_history=True)

    assert len(next_position.history) == 1
    assert len(next_position.position_history) == 2
    assert next_position.position_history[-1] == position_key(next_position)


def test_apply_move_appends_position_history_when_append_history_false():
    position = tiny_repeat_position()
    move = Move(Position(0, 6), Position(0, 5))

    next_position = apply_move(position, move, append_history=False)

    assert len(next_position.history) == 0
    assert len(next_position.position_history) == 2
    assert next_position.position_history[-1] == position_key(next_position)


def test_second_occurrence_allowed_and_third_occurrence_banned():
    position = tiny_repeat_position()
    target_board = [[cell for cell in row] for row in position.board]
    target_board[6][0] = None
    target_board[5][0] = Piece(side="CHO", kind="SOLDIER")
    target_key = board_position_key(target_board, "HAN")
    move = Move(Position(0, 6), Position(0, 5))

    position.position_history = [position_key(position), target_key]
    assert count_position_occurrences(position, target_key) == 1
    assert would_repeat_position(position, move) is False

    position.position_history = [position_key(position), target_key, "other", target_key]
    assert count_position_occurrences(position, target_key) == 2
    assert would_repeat_position(position, move) is True


def test_banned_repetition_move_is_not_legal():
    position = tiny_repeat_position()
    target_board = [[cell for cell in row] for row in position.board]
    target_board[6][0] = None
    target_board[5][0] = Piece(side="CHO", kind="SOLDIER")
    target_key = board_position_key(target_board, "HAN")
    position.position_history = [position_key(position), target_key, "other", target_key]
    forbidden = Move(Position(0, 6), Position(0, 5))

    legal_moves = generate_legal_moves(position)

    assert forbidden not in legal_moves
