from __future__ import annotations

from .inference import RandomPolicyValueModel
from .mcts import MCTSConfig, run_mcts
from .schema import Piece, TrainingPosition


def demo_position() -> TrainingPosition:
    board = [[None for _ in range(9)] for _ in range(10)]
    board[8][4] = Piece(side="CHO", kind="GENERAL")
    board[7][4] = Piece(side="CHO", kind="GUARD")
    board[1][4] = Piece(side="HAN", kind="GENERAL")
    board[6][0] = Piece(side="CHO", kind="SOLDIER")
    board[3][0] = Piece(side="HAN", kind="SOLDIER")
    return TrainingPosition(board=board, turn="CHO")


def main() -> None:
    result = run_mcts(demo_position(), RandomPolicyValueModel(seed=7), MCTSConfig(simulations=8, temperature=0))
    print("MCTS demo complete")
    print(f"move: {result.move.to_json() if result.move else None}")
    print(f"root_value: {result.root_value:.4f}")
    print(f"children: {result.children_summary[:5]}")


if __name__ == "__main__":
    main()
