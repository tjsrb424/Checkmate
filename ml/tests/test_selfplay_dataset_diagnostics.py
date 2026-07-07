from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.selfplay_dataset_diagnostics import analyze, render_markdown


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def row(game_id, turn, index, value, winner):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": turn, "history": [], "winner": None, "metadata": {}},
        "policy_target": [{"index": index, "prob": 1.0}],
        "value_target": value,
        "ply": index,
        "game_id": game_id,
        "final_winner": winner,
    }


def test_fixture_selfplay_jsonl_analysis(tmp_path):
    path = tmp_path / "selfplay.jsonl"
    write_jsonl(path, [row("g1", "CHO", 10, 1.0, "CHO"), row("g1", "HAN", 20, -1.0, "CHO")])

    diagnostics = analyze(path)

    assert diagnostics.sample_count == 2
    assert diagnostics.game_count == 1
    assert diagnostics.value_counts["positive"] == 1
    assert diagnostics.value_counts["negative"] == 1
    assert diagnostics.side_to_move_counts["CHO"] == 1
    assert diagnostics.side_to_move_counts["HAN"] == 1
    assert diagnostics.invalid_policy_targets == 0
    assert "sample_count: 2" in render_markdown(diagnostics)
