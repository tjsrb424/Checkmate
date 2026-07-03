import pytest

from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.value_dataset import ValueJsonlDataset


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def sample_row(value):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {}},
        "value": value,
        "result": "cho",
        "source": "unit",
    }


def test_value_jsonl_dataset_loads_features_and_targets(tmp_path):
    rows = [sample_row(1), sample_row(-1), sample_row(0)]
    path = tmp_path / "value.jsonl"
    write_jsonl(path, rows)

    dataset = ValueJsonlDataset(path)
    features, target = dataset[0]

    assert len(dataset) == 3
    assert tuple(features.shape) == (16, 10, 9)
    assert tuple(target.shape) == (1,)
    assert float(target) == 1.0


def test_value_jsonl_dataset_rejects_out_of_range_targets(tmp_path):
    path = tmp_path / "bad_value.jsonl"
    write_jsonl(path, [sample_row(2)])

    with pytest.raises(ValueError, match="out of range"):
        ValueJsonlDataset(path)
