import json

import torch

from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.evaluate_value import evaluate_value
from oetongsu_ml.train_value import train_value


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def row(value, turn="CHO"):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": turn, "history": [], "winner": None, "metadata": {"source": "smoke"}},
        "value": value,
        "result": "cho" if value >= 0 else "han",
        "source": "smoke",
    }


def test_train_value_smoke_creates_checkpoint_and_metrics(tmp_path):
    data_path = tmp_path / "tiny_value.jsonl"
    output_path = tmp_path / "value_net.pt"
    rows = [row(1), row(-1, "HAN"), row(0), row(1)]
    write_jsonl(data_path, rows)

    metrics = train_value(data_path, output_path, epochs=1, batch_size=2, lr=0.001, seed=7, channels=8)
    checkpoint = torch.load(output_path, map_location="cpu")
    eval_metrics = evaluate_value(output_path, data_path)

    assert output_path.exists()
    assert output_path.with_name("value_net_metrics.json").exists()
    assert "model_state" in checkpoint
    assert checkpoint["channels"] == 8
    assert metrics["history"][0]["train_mse"] == metrics["history"][0]["train_mse"]
    assert eval_metrics["mae"] >= 0

    saved_metrics = json.loads(output_path.with_name("value_net_metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["sample_count"] == 4
