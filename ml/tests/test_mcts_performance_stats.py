from oetongsu_ml.inference import RandomPolicyValueModel
from oetongsu_ml.mcts import MCTSConfig, run_mcts
from oetongsu_ml.python_rules import create_initial_position


def test_mcts_collects_performance_stats_when_enabled():
    result = run_mcts(
        create_initial_position(),
        RandomPolicyValueModel(seed=23),
        MCTSConfig(simulations=3, temperature=0, collect_stats=True),
    )

    assert result.performance is not None
    assert result.performance.simulations == 3
    assert result.performance.inference_calls > 0
    assert result.performance.expanded_nodes > 0
    assert result.performance.legal_move_generations > 0
    assert result.performance.total_ms >= 0
    assert result.performance.inference_ms >= 0


def test_mcts_skips_performance_stats_by_default():
    result = run_mcts(create_initial_position(), RandomPolicyValueModel(seed=29), MCTSConfig(simulations=1))

    assert result.performance is None
