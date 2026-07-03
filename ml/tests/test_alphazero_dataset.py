import numpy as np

from oetongsu_ml.alphazero_dataset import AlphaZeroJsonlDataset, dense_policy_target
from oetongsu_ml.dataset import write_jsonl


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def sample_row():
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {}},
        "policy_target": [{"index": 10, "prob": 2.0}, {"index": 20, "prob": 2.0}],
        "value_target": 1.0,
        "move": {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}},
        "ply": 0,
        "game_id": "unit",
        "to_play": "CHO",
        "final_winner": "CHO",
    }


def test_sparse_policy_becomes_dense_distribution():
    dense = dense_policy_target(sample_row()["policy_target"])

    assert dense.shape == (8100,)
    assert np.isclose(float(dense.sum()), 1.0)
    assert dense[10] == 0.5
    assert dense[20] == 0.5


def test_alphazero_jsonl_dataset_loads_targets(tmp_path):
    path = tmp_path / "selfplay.jsonl"
    write_jsonl(path, [sample_row()])

    dataset = AlphaZeroJsonlDataset(path)
    features, policy, value = dataset[0]

    assert tuple(features.shape) == (16, 10, 9)
    assert tuple(policy.shape) == (8100,)
    assert tuple(value.shape) == (1,)
    assert float(value) == 1.0


def test_alphazero_dataset_reads_supervised_export_shape(tmp_path):
    path = tmp_path / "az_supervised.jsonl"
    row = sample_row()
    row["policy_target"] = [{"index": 123, "prob": 1.0}]
    row["position"]["positionHistory"] = ["initial-key"]
    write_jsonl(path, [row])

    dataset = AlphaZeroJsonlDataset(path)
    _, policy, value = dataset[0]

    assert np.isclose(float(policy.sum()), 1.0)
    assert float(policy[123]) == 1.0
    assert float(value) == 1.0
