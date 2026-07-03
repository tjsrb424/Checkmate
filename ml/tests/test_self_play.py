import json

from oetongsu_ml.inference import RandomPolicyValueModel
from oetongsu_ml.self_play import (
    SelfPlayConfig,
    play_self_play_game,
    self_play_sample_from_raw,
    self_play_samples_to_jsonl,
)


def test_self_play_game_generates_samples():
    result = play_self_play_game(
        RandomPolicyValueModel(seed=11),
        SelfPlayConfig(game_id="unit", max_plies=4, mcts_simulations=2, temperature=0, seed=5),
    )

    assert result.game_id == "unit"
    assert 0 < len(result.samples) <= 4
    assert result.plies <= 4
    assert result.outcome in {"checkmate", "draw_max_plies", "draw_no_legal_moves", "loss_no_legal_moves"}
    assert all(sample.policy_target for sample in result.samples)
    assert all(sample.value_target in (-1.0, 0.0, 1.0) for sample in result.samples)


def test_self_play_respects_max_plies_draw():
    result = play_self_play_game(
        RandomPolicyValueModel(seed=3),
        SelfPlayConfig(game_id="short", max_plies=2, mcts_simulations=2, temperature=0, seed=9),
    )

    assert result.plies <= 2
    assert result.outcome == "draw_max_plies"
    assert result.winner is None
    assert all(sample.value_target == 0.0 for sample in result.samples)


def test_self_play_jsonl_round_trip():
    result = play_self_play_game(
        RandomPolicyValueModel(seed=17),
        SelfPlayConfig(game_id="jsonl", max_plies=2, mcts_simulations=2, temperature=0, seed=2),
    )
    jsonl = self_play_samples_to_jsonl(result.samples)
    rows = [json.loads(line) for line in jsonl.splitlines()]
    restored = [self_play_sample_from_raw(row) for row in rows]

    assert len(restored) == len(result.samples)
    assert restored[0].game_id == "jsonl"
    assert restored[0].policy_target
    assert restored[0].move == result.samples[0].move
