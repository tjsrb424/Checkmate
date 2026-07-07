import numpy as np

from oetongsu_ml import policy_value_sample_review as review
from oetongsu_ml.dataset import write_jsonl


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def sample_row():
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": "CHO", "history": [], "winner": None, "metadata": {}},
        "policy_target": [{"index": 10, "prob": 1.0}],
        "value_target": 1.0,
        "game_id": "g1",
        "final_winner": "CHO",
    }


class DummyModel:
    def __init__(self, path, device="cpu"):
        self.path = path

    def predict(self, position):
        policy = np.zeros((8100,), dtype=np.float32)
        policy[10] = 0.9
        policy[20] = 0.1
        return policy, 0.25


def test_policy_value_sample_review_writes_markdown(tmp_path, monkeypatch):
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "review.md"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("x")
    write_jsonl(samples, [sample_row()])
    monkeypatch.setattr(review, "TorchAlphaZeroModel", DummyModel)
    monkeypatch.setattr(review, "generate_legal_moves", lambda position: [])

    status = review.main(
        [
            "--champion",
            str(checkpoint),
            "--previous",
            str(checkpoint),
            "--candidate",
            str(checkpoint),
            "--samples",
            str(samples),
            "--output",
            str(output),
            "--no-report",
        ]
    )

    assert status == 0
    text = output.read_text(encoding="utf-8")
    assert "# A3 Policy/Value Sample Review" in text
    assert "policy_target_in_candidate_top5: True" in text
