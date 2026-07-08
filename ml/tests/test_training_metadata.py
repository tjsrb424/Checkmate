import json

import torch

from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.train_alphazero import train_alphazero


def sample_row(index, value):
    board = [[None for _ in range(9)] for _ in range(10)]
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {}},
        "policy_target": [{"index": index, "prob": 1.0}],
        "value_target": value,
        "move": {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}},
        "ply": 0,
        "game_id": "metadata",
        "to_play": "CHO",
        "final_winner": "CHO",
    }


def test_train_alphazero_writes_training_metadata_to_metrics_and_checkpoint(tmp_path):
    data = tmp_path / "selfplay.jsonl"
    output = tmp_path / "candidate.pt"
    write_jsonl(data, [sample_row(1, 1.0), sample_row(2, 0.0), sample_row(3, -1.0), sample_row(4, 1.0)])

    train_alphazero(
        data,
        output,
        epochs=1,
        batch_size=2,
        channels=4,
        seed=4,
        training_metadata={"source": "test", "candidate_version": "az_iter_000003", "champion_version": "supervised_v0001"},
    )

    metrics = json.loads(output.with_name("candidate_metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output, map_location="cpu")
    metadata = metrics["training_metadata"]

    assert metadata["source"] == "test"
    assert metadata["candidate_version"] == "az_iter_000003"
    assert metadata["champion_version"] == "supervised_v0001"
    assert metadata["seed"] == 4
    assert metadata["split_seed"] == 4
    assert metadata["optimizer"] == "Adam"
    assert metadata["weight_decay"] == 0.0
    assert metadata["shuffle"] is True
    assert metadata["train_count"] == 3
    assert metadata["val_count"] == 1
    assert checkpoint["training_metadata"]["candidate_version"] == "az_iter_000003"
