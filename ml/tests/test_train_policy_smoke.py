import json

import torch

from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.evaluate_policy import evaluate_policy
from oetongsu_ml.move_index import move_to_index
from oetongsu_ml.train_policy import train_policy


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def row(move):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {"source": "smoke"}},
        "move": move,
        "move_index": move_to_index(move),
        "result": "cho",
        "source": "smoke",
    }


def test_train_policy_smoke_creates_checkpoint_and_metrics(tmp_path):
    data_path = tmp_path / "tiny_policy.jsonl"
    output_path = tmp_path / "policy_net.pt"
    rows = [
        row({"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}}),
        row({"from": {"x": 2, "y": 6}, "to": {"x": 2, "y": 5}}),
        row({"from": {"x": 4, "y": 6}, "to": {"x": 4, "y": 5}}),
        row({"from": {"x": 6, "y": 6}, "to": {"x": 6, "y": 5}}),
    ]
    write_jsonl(data_path, rows)

    metrics = train_policy(data_path, output_path, epochs=1, batch_size=2, lr=0.001, seed=7, channels=8)
    checkpoint = torch.load(output_path, map_location="cpu")
    eval_metrics = evaluate_policy(output_path, data_path)

    assert output_path.exists()
    assert output_path.with_name("policy_net_metrics.json").exists()
    assert "model_state" in checkpoint
    assert checkpoint["channels"] == 8
    assert tuple(next(iter(checkpoint["model_state"].values())).shape)
    assert metrics["history"][0]["train_loss"] == metrics["history"][0]["train_loss"]
    assert eval_metrics["top5"] >= 0

    saved_metrics = json.loads(output_path.with_name("policy_net_metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["sample_count"] == 4
