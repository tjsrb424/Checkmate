from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .constants import POLICY_SIZE
from .inference import PolicyValueModel
from .mcts import MCTSConfig, MCTSResult, run_mcts
from .move_index import index_to_move, is_valid_policy_index, move_to_index
from .python_rules import apply_move, create_initial_position, generate_legal_moves, is_in_check, other_side
from .ruleset import RulesetId, resolve_ruleset
from .scoring import score_board_material
from .schema import Move, Side, TrainingPosition

Outcome = Literal["checkmate", "draw_max_plies", "score_adjudication", "draw_no_legal_moves", "loss_no_legal_moves"]


@dataclass
class SelfPlayConfig:
    game_id: str = "selfplay-000001"
    max_plies: int = 120
    mcts_simulations: int = 64
    temperature: float = 1.0
    temperature_drop_ply: int = 20
    seed: int | None = 1
    record_history: bool = True
    cpuct: float = 1.5
    max_depth: int = 200
    ruleset_id: RulesetId = "kakao-like"


@dataclass
class SelfPlaySampleRecord:
    position: TrainingPosition
    policy_target: list[dict[str, float | int]]
    value_target: float
    move: Move
    ply: int
    game_id: str
    to_play: Side
    final_winner: Side | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_json(),
            "policy_target": self.policy_target,
            "value_target": self.value_target,
            "move": self.move.to_json(),
            "ply": self.ply,
            "game_id": self.game_id,
            "to_play": self.to_play,
            "final_winner": self.final_winner,
        }


@dataclass
class PendingSelfPlaySample:
    position: TrainingPosition
    policy_target: list[dict[str, float | int]]
    move: Move
    ply: int
    game_id: str
    to_play: Side


@dataclass
class SelfPlayGameResult:
    game_id: str
    winner: Side | None
    outcome: Outcome
    plies: int
    history: list[Move] = field(default_factory=list)
    samples: list[SelfPlaySampleRecord] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "outcome": self.outcome,
            "plies": self.plies,
            "sample_count": len(self.samples),
            "history": [move.to_json() for move in self.history],
        }


def play_self_play_game(model: PolicyValueModel, config: SelfPlayConfig | None = None) -> SelfPlayGameResult:
    cfg = config or SelfPlayConfig()
    ruleset = resolve_ruleset(cfg.ruleset_id)
    rng = np.random.default_rng(cfg.seed)
    position = create_initial_position()
    pending_samples: list[PendingSelfPlaySample] = []
    history: list[Move] = []
    winner: Side | None = None
    outcome: Outcome = "draw_max_plies"

    for ply in range(cfg.max_plies):
        legal_moves = generate_legal_moves(position, ruleset=ruleset)
        if len(legal_moves) == 0:
            if is_in_check(position, position.turn):
                winner = other_side(position.turn)
                outcome = "loss_no_legal_moves"
            else:
                outcome = "draw_no_legal_moves"
            break

        temperature = 0.0 if ply >= cfg.temperature_drop_ply else cfg.temperature
        mcts_config = MCTSConfig(
            simulations=cfg.mcts_simulations,
            cpuct=cfg.cpuct,
            temperature=temperature,
            max_depth=cfg.max_depth,
            ruleset_id=ruleset.id,
        )
        mcts_result = run_mcts(position, model, mcts_config)
        move = select_move(mcts_result, temperature, rng)
        if move is None:
            outcome = "draw_no_legal_moves"
            break

        pending_samples.append(
            PendingSelfPlaySample(
                position=TrainingPosition.from_raw(position.to_json()),
                policy_target=sparse_policy_target(mcts_result.policy_target),
                move=move,
                ply=ply,
                game_id=cfg.game_id,
                to_play=position.turn,
            )
        )
        history.append(move)
        position = apply_move(position, move, append_history=cfg.record_history)
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

    samples = finalize_samples(pending_samples, winner)
    return SelfPlayGameResult(
        game_id=cfg.game_id,
        winner=winner,
        outcome=outcome,
        plies=len(history),
        history=history if cfg.record_history else [],
        samples=samples,
    )


def self_play_samples_to_jsonl(samples: list[SelfPlaySampleRecord]) -> str:
    return "".join(json.dumps(sample.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n" for sample in samples)


def self_play_sample_from_raw(raw: dict[str, Any]) -> SelfPlaySampleRecord:
    move = Move.from_raw(raw["move"])
    policy_target = [
        {"index": int(item["index"]), "prob": float(item["prob"])}
        for item in raw.get("policy_target", [])
        if is_valid_policy_index(int(item["index"]))
    ]
    return SelfPlaySampleRecord(
        position=TrainingPosition.from_raw(raw["position"]),
        policy_target=policy_target,
        value_target=float(raw["value_target"]),
        move=move,
        ply=int(raw["ply"]),
        game_id=str(raw["game_id"]),
        to_play=raw["to_play"],
        final_winner=raw.get("final_winner"),
    )


def select_move(result: MCTSResult, temperature: float, rng: np.random.Generator) -> Move | None:
    if not result.visit_counts:
        return result.move
    if temperature <= 0:
        index = max(result.visit_counts.items(), key=lambda item: (item[1], item[0]))[0]
        return index_to_move(index)

    indexes = np.array(list(result.visit_counts.keys()), dtype=np.int64)
    counts = np.array([result.visit_counts[int(index)] for index in indexes], dtype=np.float64)
    weights = counts ** (1.0 / max(temperature, 1e-6))
    total = float(weights.sum())
    if total <= 0:
        probabilities = np.ones_like(weights) / len(weights)
    else:
        probabilities = weights / total
    return index_to_move(int(rng.choice(indexes, p=probabilities)))


def sparse_policy_target(policy_target: np.ndarray) -> list[dict[str, float | int]]:
    entries: list[dict[str, float | int]] = []
    for index in np.flatnonzero(policy_target > 0):
        entries.append({"index": int(index), "prob": float(policy_target[index])})
    return entries


def finalize_samples(pending_samples: list[PendingSelfPlaySample], winner: Side | None) -> list[SelfPlaySampleRecord]:
    samples: list[SelfPlaySampleRecord] = []
    for sample in pending_samples:
        samples.append(
            SelfPlaySampleRecord(
                position=sample.position,
                policy_target=sample.policy_target,
                value_target=value_target_for(sample.to_play, winner),
                move=sample.move,
                ply=sample.ply,
                game_id=sample.game_id,
                to_play=sample.to_play,
                final_winner=winner,
            )
        )
    return samples


def value_target_for(to_play: Side, winner: Side | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if to_play == winner else -1.0
