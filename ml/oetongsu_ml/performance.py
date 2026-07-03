from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any


@dataclass
class InferenceStats:
    calls: int = 0
    positions: int = 0
    total_ms: float = 0.0
    avg_ms: float = 0.0
    positions_per_sec: float = 0.0

    @classmethod
    def from_totals(cls, calls: int, positions: int, total_ms: float) -> "InferenceStats":
        return cls(
            calls=calls,
            positions=positions,
            total_ms=total_ms,
            avg_ms=(total_ms / calls) if calls > 0 else 0.0,
            positions_per_sec=rate_per_sec(positions, total_ms),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MCTSPerformanceStats:
    simulations: int = 0
    expanded_nodes: int = 0
    inference_calls: int = 0
    inference_ms: float = 0.0
    legal_move_generations: int = 0
    total_ms: float = 0.0
    nodes_per_sec: float = 0.0

    @classmethod
    def from_totals(
        cls,
        simulations: int,
        expanded_nodes: int,
        inference_calls: int,
        inference_ms: float,
        legal_move_generations: int,
        total_ms: float,
    ) -> "MCTSPerformanceStats":
        return cls(
            simulations=simulations,
            expanded_nodes=expanded_nodes,
            inference_calls=inference_calls,
            inference_ms=inference_ms,
            legal_move_generations=legal_move_generations,
            total_ms=total_ms,
            nodes_per_sec=rate_per_sec(expanded_nodes, total_ms),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfPlayPerformanceStats:
    games: int = 0
    plies: int = 0
    samples: int = 0
    total_ms: float = 0.0
    games_per_sec: float = 0.0
    plies_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    mcts_total_ms: float = 0.0
    inference_total_ms: float = 0.0

    @classmethod
    def from_totals(
        cls,
        games: int,
        plies: int,
        samples: int,
        total_ms: float,
        mcts_total_ms: float = 0.0,
        inference_total_ms: float = 0.0,
    ) -> "SelfPlayPerformanceStats":
        return cls(
            games=games,
            plies=plies,
            samples=samples,
            total_ms=total_ms,
            games_per_sec=rate_per_sec(games, total_ms),
            plies_per_sec=rate_per_sec(plies, total_ms),
            samples_per_sec=rate_per_sec(samples, total_ms),
            mcts_total_ms=mcts_total_ms,
            inference_total_ms=inference_total_ms,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def now_ms() -> float:
    return perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return max(0.0, now_ms() - start_ms)


def rate_per_sec(count: int | float, total_ms: float) -> float:
    return float(count) / (total_ms / 1000.0) if total_ms > 0 else 0.0


def aggregate_self_play_performance(items: list[SelfPlayPerformanceStats]) -> SelfPlayPerformanceStats:
    return SelfPlayPerformanceStats.from_totals(
        games=sum(item.games for item in items),
        plies=sum(item.plies for item in items),
        samples=sum(item.samples for item in items),
        total_ms=sum(item.total_ms for item in items),
        mcts_total_ms=sum(item.mcts_total_ms for item in items),
        inference_total_ms=sum(item.inference_total_ms for item in items),
    )
