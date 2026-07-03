from oetongsu_ml.dataset import load_policy_samples, policy_sample_to_arrays, read_jsonl, write_jsonl
from oetongsu_ml.move_index import move_to_index
from oetongsu_ml.schema import Move, Piece, PolicyTrainingSample, Position, TrainingPosition, ValueTrainingSample


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def sample_position():
    board = empty_board()
    board[9][4] = Piece(side="CHO", kind="GENERAL")
    return TrainingPosition(board=board, turn="CHO", metadata={"id": "sample"})


def test_training_position_from_raw_parses_board_and_history():
    raw = {
        "board": empty_board(),
        "turn": "HAN",
        "history": [{"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}}],
        "metadata": {"source": "unit"},
    }
    raw["board"][0][4] = {"side": "HAN", "kind": "GENERAL"}

    parsed = TrainingPosition.from_raw(raw)

    assert parsed.board[0][4] == Piece(side="HAN", kind="GENERAL")
    assert parsed.history[0] == Move(from_=Position(x=0, y=6), to=Position(x=0, y=5))
    assert parsed.metadata["source"] == "unit"


def test_policy_sample_jsonl_round_trip(tmp_path):
    move = Move(from_=Position(x=0, y=6), to=Position(x=0, y=5))
    sample = PolicyTrainingSample(
        position=sample_position(),
        move=move,
        move_index=move_to_index(move),
        result=1,
        source="unit",
    )
    path = tmp_path / "policy.jsonl"

    write_jsonl(path, [sample])
    loaded = load_policy_samples(path)

    assert len(read_jsonl(path)) == 1
    assert loaded[0].move == move
    assert loaded[0].move_index == move_to_index(move)


def test_policy_sample_to_arrays_returns_tensor_and_target():
    move = Move(from_=Position(x=0, y=6), to=Position(x=0, y=5))
    sample = PolicyTrainingSample(position=sample_position(), move=move, move_index=move_to_index(move))

    tensor, target = policy_sample_to_arrays(sample)

    assert tensor.shape == (16, 10, 9)
    assert int(target) == move_to_index(move)


def test_value_sample_from_raw():
    raw = {"position": sample_position().to_json(), "value": -1, "result": "loss", "source": "unit"}

    parsed = ValueTrainingSample.from_raw(raw)

    assert parsed.value == -1.0
    assert parsed.result == "loss"

