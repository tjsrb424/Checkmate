import numpy as np

from oetongsu_ml.constants import POLICY_SIZE
from oetongsu_ml.inference import PolicyValueModel, RandomPolicyValueModel
from oetongsu_ml.mcts import MCTSConfig, run_mcts
from oetongsu_ml.move_index import move_to_index
from oetongsu_ml.python_rules import generate_legal_moves
from oetongsu_ml.schema import Move, Piece, TrainingPosition


class UniformModel(PolicyValueModel):
    def __init__(self, value=0.0):
        self.value = value

    def predict(self, position):
        return np.ones((POLICY_SIZE,), dtype=np.float32) / POLICY_SIZE, self.value


class PriorModel(PolicyValueModel):
    def __init__(self, preferred_move: Move):
        self.preferred_index = move_to_index(preferred_move)

    def predict(self, position):
        policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
        policy[self.preferred_index] = 1.0
        return policy, 0.0


def sample_position():
    board = [[None for _ in range(9)] for _ in range(10)]
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    board[7][4] = Piece(side="CHO", kind="GUARD")
    board[1][4] = Piece(side="HAN", kind="GENERAL")
    board[6][0] = Piece(side="CHO", kind="SOLDIER")
    board[3][0] = Piece(side="HAN", kind="SOLDIER")
    return TrainingPosition(board=board, turn="CHO")


def terminal_position():
    board = [[None for _ in range(9)] for _ in range(10)]
    board[1][4] = Piece(side="HAN", kind="GENERAL")
    return TrainingPosition(board=board, turn="CHO")


def test_mcts_returns_a_legal_move():
    position = sample_position()
    result = run_mcts(position, UniformModel(), MCTSConfig(simulations=8, temperature=0))
    legal = {move_to_index(move) for move in generate_legal_moves(position)}

    assert result.move is not None
    assert move_to_index(result.move) in legal
    assert result.policy_target.shape == (POLICY_SIZE,)
    assert np.isclose(float(result.policy_target.sum()), 1.0)


def test_mcts_root_visit_count_matches_simulations():
    result = run_mcts(sample_position(), RandomPolicyValueModel(seed=5), MCTSConfig(simulations=6, temperature=0))

    assert sum(result.visit_counts.values()) == 6
    assert len(result.children_summary) > 0


def test_mcts_handles_terminal_state_safely():
    result = run_mcts(terminal_position(), UniformModel(), MCTSConfig(simulations=4))

    assert result.move is None
    assert result.visit_counts == {}
    assert float(result.policy_target.sum()) == 0.0
    assert result.root_value == -1.0


def test_uniform_model_returns_move():
    result = run_mcts(sample_position(), UniformModel(), MCTSConfig(simulations=4, temperature=0))

    assert result.move is not None


def test_policy_prior_biases_visit_counts():
    position = sample_position()
    preferred = generate_legal_moves(position)[0]
    preferred_index = move_to_index(preferred)
    result = run_mcts(position, PriorModel(preferred), MCTSConfig(simulations=12, temperature=0, cpuct=2.0))

    assert result.move == preferred
    assert result.visit_counts[preferred_index] == max(result.visit_counts.values())
