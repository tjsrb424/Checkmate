import json

import torch

from oetongsu_ml.alphazero_model import AlphaZeroNet
from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.train_alphazero import soft_cross_entropy, train_alphazero
from oetongsu_ml.training_loop import TrainingIterationConfig, run_training_iteration


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def row(index, value):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {"source": "az-smoke"}},
        "policy_target": [{"index": index, "prob": 1.0}],
        "value_target": value,
        "move": {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}},
        "ply": 0,
        "game_id": "az-smoke",
        "to_play": "CHO",
        "final_winner": "CHO" if value > 0 else None,
    }


def test_alphazero_model_forward_and_loss_shape():
    model = AlphaZeroNet(channels=4)
    policy_logits, value = model(torch.zeros((2, 16, 10, 9)))
    target = torch.zeros((2, 8100))
    target[:, 10] = 1.0
    loss = soft_cross_entropy(policy_logits, target)

    assert tuple(policy_logits.shape) == (2, 8100)
    assert tuple(value.shape) == (2, 1)
    assert loss == loss


def test_train_alphazero_smoke_creates_checkpoint(tmp_path):
    data = tmp_path / "selfplay.jsonl"
    output = tmp_path / "az_model.pt"
    write_jsonl(data, [row(10, 1.0), row(20, 0.0), row(30, -1.0), row(40, 1.0)])

    metrics = train_alphazero(data, output, epochs=1, batch_size=2, channels=4)

    assert output.exists()
    assert output.with_name("az_model_metrics.json").exists()
    assert metrics["history"][0]["total_loss"] == metrics["history"][0]["total_loss"]
    saved = json.loads(output.with_name("az_model_metrics.json").read_text(encoding="utf-8"))
    assert saved["sample_count"] == 4


def test_quick_training_iteration_smoke(tmp_path):
    result = run_training_iteration(
        TrainingIterationConfig(
            games=1,
            max_plies=2,
            simulations=2,
            epochs=1,
            batch_size=2,
            channels=4,
            output_dir=str(tmp_path / "selfplay"),
            model_output=str(tmp_path / "az_model_latest.pt"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert result["sample_count"] > 0
    assert (tmp_path / "az_model_latest.pt").exists()
    assert result["checkpoint"].endswith(".pt")
