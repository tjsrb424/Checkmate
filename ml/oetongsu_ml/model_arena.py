from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .inference import PolicyValueModel, RandomPolicyValueModel, TorchAlphaZeroModel
from .mcts import MCTSConfig, run_mcts
from .python_rules import apply_move, create_initial_position, generate_legal_moves, is_in_check, other_side
from .ruleset import RulesetId, resolve_ruleset
from .scoring import score_board_material
from .schema import Move, Side, TrainingPosition


@dataclass
class ModelArenaConfig:
    games: int = 2
    simulations: int = 2
    max_plies: int = 4
    swap_sides: bool = True
    temperature: float = 0.0
    seed: int = 1
    promotion_threshold: float = 0.55
    ruleset_id: RulesetId = "kakao-like"


@dataclass
class ModelArenaResult:
    games: int
    candidateWins: int
    championWins: int
    draws: int
    candidateScoreRate: float
    championScoreRate: float
    averagePlies: float
    promoted: bool
    illegalMoves: int
    forfeits: int
    gameSummaries: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ModelPlayer:
    name: str

    def select_move(self, position: TrainingPosition, config: ModelArenaConfig) -> Move | None:
        raise NotImplementedError


class MCTSModelPlayer(ModelPlayer):
    def __init__(self, model: PolicyValueModel, name: str = "model") -> None:
        self.model = model
        self.name = name

    def select_move(self, position: TrainingPosition, config: ModelArenaConfig) -> Move | None:
        result = run_mcts(
            position,
            self.model,
            MCTSConfig(simulations=config.simulations, temperature=config.temperature, ruleset_id=config.ruleset_id),
        )
        return result.move


class RandomModelPlayer(MCTSModelPlayer):
    def __init__(self, name: str = "random", seed: int | None = 1) -> None:
        super().__init__(RandomPolicyValueModel(seed=seed), name=name)


class TorchModelPlayer(MCTSModelPlayer):
    def __init__(self, checkpoint: str | Path, name: str = "torch") -> None:
        super().__init__(TorchAlphaZeroModel(checkpoint), name=name)


def run_model_arena(
    candidate: ModelPlayer,
    champion: ModelPlayer,
    config: ModelArenaConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ModelArenaResult:
    cfg = config or ModelArenaConfig()
    candidate_wins = 0
    champion_wins = 0
    draws = 0
    illegal_moves = 0
    forfeits = 0
    plies: list[int] = []
    summaries: list[dict[str, Any]] = []

    for game_index in range(cfg.games):
        candidate_side: Side = "CHO" if (not cfg.swap_sides or game_index % 2 == 0) else "HAN"
        champion_side = other_side(candidate_side)
        result = play_arena_game(candidate, champion, candidate_side, champion_side, cfg, game_index)
        plies.append(result["plies"])
        summaries.append(result)
        illegal_moves += int(result["illegalMoves"])
        forfeits += int(result["forfeits"])
        winner = result["winner"]
        if winner is None:
            draws += 1
        elif winner == candidate_side:
            candidate_wins += 1
        else:
            champion_wins += 1
        if progress_callback:
            completed_games = game_index + 1
            candidate_score = candidate_wins + draws * 0.5
            champion_score = champion_wins + draws * 0.5
            progress_callback(
                {
                    "currentGames": completed_games,
                    "totalGames": cfg.games,
                    "candidateWins": candidate_wins,
                    "championWins": champion_wins,
                    "draws": draws,
                    "candidateScoreRate": candidate_score / completed_games if completed_games > 0 else 0.0,
                    "championScoreRate": champion_score / completed_games if completed_games > 0 else 0.0,
                    "illegalMoves": illegal_moves,
                    "forfeits": forfeits,
                }
            )

    candidate_score = candidate_wins + draws * 0.5
    champion_score = champion_wins + draws * 0.5
    candidate_rate = candidate_score / cfg.games if cfg.games > 0 else 0.0
    champion_rate = champion_score / cfg.games if cfg.games > 0 else 0.0
    promoted = candidate_rate >= cfg.promotion_threshold and illegal_moves == 0 and forfeits == 0
    return ModelArenaResult(
        games=cfg.games,
        candidateWins=candidate_wins,
        championWins=champion_wins,
        draws=draws,
        candidateScoreRate=candidate_rate,
        championScoreRate=champion_rate,
        averagePlies=sum(plies) / len(plies) if plies else 0.0,
        promoted=promoted,
        illegalMoves=illegal_moves,
        forfeits=forfeits,
        gameSummaries=summaries,
    )


def play_arena_game(
    candidate: ModelPlayer,
    champion: ModelPlayer,
    candidate_side: Side,
    champion_side: Side,
    config: ModelArenaConfig,
    game_index: int,
) -> dict[str, Any]:
    ruleset = resolve_ruleset(config.ruleset_id)
    position = create_initial_position()
    illegal_moves = 0
    forfeits = 0
    winner: Side | None = None
    outcome = "draw_max_plies"

    for ply in range(config.max_plies):
        legal_moves = generate_legal_moves(position, ruleset=ruleset)
        if not legal_moves:
            if is_in_check(position, position.turn):
                winner = other_side(position.turn)
                outcome = "loss_no_legal_moves"
                forfeits += 1
            else:
                outcome = "draw_no_legal_moves"
            break

        player = candidate if position.turn == candidate_side else champion
        move = player.select_move(position, config)
        if move is None or move not in legal_moves:
            illegal_moves += 1
            forfeits += 1
            winner = other_side(position.turn)
            outcome = "illegal_move"
            break

        position = apply_move(position, move, append_history=True)
        if position.winner is not None:
            winner = position.winner
            outcome = "checkmate"
            break
    else:
        if ruleset.max_ply_policy == "score-adjudication":
            score = score_board_material(position.board)
            winner = score["winner"] if score["winner"] in ("CHO", "HAN") else None
            outcome = "score_adjudication" if winner is not None else "draw_max_plies"
        else:
            outcome = "draw_max_plies"
        ply = config.max_plies - 1

    return {
        "game": game_index + 1,
        "candidateSide": candidate_side,
        "championSide": champion_side,
        "winner": winner,
        "outcome": outcome,
        "plies": max(0, ply + 1 if config.max_plies > 0 else 0),
        "illegalMoves": illegal_moves,
        "forfeits": forfeits,
    }
