from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.move_index import move_to_index
from oetongsu_ml.policy_dataset import PolicyJsonlDataset


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def sample_row(move):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {}},
        "move": move,
        "move_index": move_to_index(move),
        "result": "cho",
        "source": "unit",
    }


def test_policy_jsonl_dataset_loads_features_and_targets(tmp_path):
    rows = [
        sample_row({"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}}),
        sample_row({"from": {"x": 2, "y": 6}, "to": {"x": 2, "y": 5}}),
    ]
    path = tmp_path / "policy.jsonl"
    write_jsonl(path, rows)

    dataset = PolicyJsonlDataset(path)
    features, target = dataset[0]

    assert len(dataset) == 2
    assert tuple(features.shape) == (16, 10, 9)
    assert int(target) == rows[0]["move_index"]
