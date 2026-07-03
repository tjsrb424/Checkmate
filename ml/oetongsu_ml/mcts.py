from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import POLICY_SIZE
from .inference import PolicyValueModel
from .move_index import move_to_index
from .performance import MCTSPerformanceStats, elapsed_ms, now_ms
from .python_rules import apply_move, generate_legal_moves, is_in_check
from .schema import Move, TrainingPosition
from .ruleset import RulesetId


@dataclass
class MCTSConfig:
    simulations: int = 64
    cpuct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    max_depth: int = 200
    ruleset_id: RulesetId = "oetongsu-basic"
    collect_stats: bool = False


@dataclass
class MCTSNode:
    position: TrainingPosition
    prior: float = 0.0
    move: Move | None = None
    parent: "MCTSNode | None" = None
    to_play: str = "CHO"
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    def ucb_score(self, child: "MCTSNode", cpuct: float) -> float:
        prior_score = cpuct * child.prior * math.sqrt(max(1, self.visit_count)) / (1 + child.visit_count)
        return -child.q_value + prior_score


@dataclass
class MCTSResult:
    move: Move | None
    visit_counts: dict[int, int]
    policy_target: np.ndarray
    root_value: float
    children_summary: list[dict]
    performance: MCTSPerformanceStats | None = None


def run_mcts(position: TrainingPosition | dict, model: PolicyValueModel, config: MCTSConfig | None = None) -> MCTSResult:
    start_ms = now_ms()
    parsed = TrainingPosition.from_raw(position)
    cfg = config or MCTSConfig()
    counters = new_counters() if cfg.collect_stats else None
    root = MCTSNode(position=parsed, to_play=parsed.turn)
    legal_moves = counted_legal_moves(parsed, cfg.ruleset_id, counters)
    if len(legal_moves) == 0:
        value = terminal_value(parsed)
        return build_result(root, cfg, value, build_performance(cfg, counters, elapsed_ms(start_ms)))

    expand(root, model, add_dirichlet_noise=cfg.dirichlet_epsilon > 0, config=cfg, counters=counters)
    root_value = root.q_value

    for _ in range(max(0, cfg.simulations)):
        node = root
        search_path = [node]
        depth = 0
        while node.expanded and depth < cfg.max_depth:
            node = select_child(node, cfg)
            search_path.append(node)
            depth += 1

        legal = counted_legal_moves(node.position, cfg.ruleset_id, counters)
        if len(legal) == 0:
            value = terminal_value(node.position)
        elif depth >= cfg.max_depth:
            _, value = counted_predict(model, node.position, counters)
        else:
            _, value = expand(node, model, config=cfg, counters=counters)
        backup(search_path, value)
        root_value = root.q_value

    return build_result(root, cfg, root_value, build_performance(cfg, counters, elapsed_ms(start_ms)))


def select_child(node: MCTSNode, config: MCTSConfig) -> MCTSNode:
    return max(node.children.values(), key=lambda child: (node.ucb_score(child, config.cpuct), child.prior))


def expand(
    node: MCTSNode,
    model: PolicyValueModel,
    add_dirichlet_noise: bool = False,
    config: MCTSConfig | None = None,
    counters: dict[str, float] | None = None,
) -> tuple[np.ndarray, float]:
    legal_moves = counted_legal_moves(node.position, config.ruleset_id if config else None, counters)
    policy_probs, value = counted_predict(model, node.position, counters)
    if counters is not None:
        counters["expanded_nodes"] += 1
    priors = normalize_legal_priors(policy_probs, legal_moves)

    if add_dirichlet_noise and legal_moves and config is not None:
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(legal_moves)).astype(np.float32)
        epsilon = config.dirichlet_epsilon
        for index, move in enumerate(legal_moves):
            move_index = move_to_index(move)
            priors[move_index] = (1 - epsilon) * priors[move_index] + epsilon * float(noise[index])

    for move in legal_moves:
        move_index = move_to_index(move)
        child_position = apply_move(node.position, move)
        node.children[move_index] = MCTSNode(
            position=child_position,
            prior=float(priors[move_index]),
            move=move,
            parent=node,
            to_play=child_position.turn,
        )
    return priors, float(np.clip(value, -1.0, 1.0))


def normalize_legal_priors(policy_probs: np.ndarray, legal_moves: list[Move]) -> np.ndarray:
    priors = np.zeros((POLICY_SIZE,), dtype=np.float32)
    if len(legal_moves) == 0:
        return priors

    legal_indexes = [move_to_index(move) for move in legal_moves]
    raw = np.array([float(policy_probs[index]) if 0 <= index < len(policy_probs) else 0.0 for index in legal_indexes], dtype=np.float32)
    raw = np.maximum(raw, 0.0)
    total = float(raw.sum())
    if total <= 0:
        raw.fill(1.0 / len(legal_indexes))
    else:
        raw /= total
    for index, move_index in enumerate(legal_indexes):
        priors[move_index] = raw[index]
    return priors


def backup(search_path: list[MCTSNode], value: float) -> None:
    current_value = float(np.clip(value, -1.0, 1.0))
    for node in reversed(search_path):
        node.value_sum += current_value
        node.visit_count += 1
        current_value = -current_value


def terminal_value(position: TrainingPosition) -> float:
    return -1.0 if is_in_check(position, position.turn) else 0.0


def build_result(
    root: MCTSNode,
    config: MCTSConfig,
    root_value: float,
    performance: MCTSPerformanceStats | None = None,
) -> MCTSResult:
    visit_counts = {move_index: child.visit_count for move_index, child in root.children.items()}
    policy_target = np.zeros((POLICY_SIZE,), dtype=np.float32)
    total_visits = sum(visit_counts.values())
    if total_visits > 0:
        for move_index, count in visit_counts.items():
            policy_target[move_index] = count / total_visits

    selected = select_move_from_visits(root, config)
    return MCTSResult(
        move=selected,
        visit_counts=visit_counts,
        policy_target=policy_target,
        root_value=float(np.clip(root_value, -1.0, 1.0)),
        children_summary=children_summary(root),
        performance=performance if config.collect_stats else None,
    )


def select_move_from_visits(root: MCTSNode, config: MCTSConfig) -> Move | None:
    if not root.children:
        return None
    if config.temperature == 0:
        return max(root.children.values(), key=lambda child: (child.visit_count, child.prior)).move

    exponent = 1.0 / max(config.temperature, 1e-6)
    return max(root.children.values(), key=lambda child: ((child.visit_count ** exponent), child.prior)).move


def children_summary(root: MCTSNode) -> list[dict]:
    return [
        {
            "move": child.move.to_json() if child.move else None,
            "move_index": move_index,
            "prior": child.prior,
            "visit_count": child.visit_count,
            "q_value": child.q_value,
        }
        for move_index, child in sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
    ]


def new_counters() -> dict[str, float]:
    return {
        "expanded_nodes": 0.0,
        "inference_calls": 0.0,
        "inference_ms": 0.0,
        "legal_move_generations": 0.0,
    }


def counted_legal_moves(position: TrainingPosition, ruleset, counters: dict[str, float] | None) -> list[Move]:
    if counters is not None:
        counters["legal_move_generations"] += 1
    return generate_legal_moves(position, ruleset=ruleset)


def counted_predict(
    model: PolicyValueModel,
    position: TrainingPosition,
    counters: dict[str, float] | None,
) -> tuple[np.ndarray, float]:
    start_ms = now_ms()
    policy, value = model.predict(position)
    if counters is not None:
        counters["inference_calls"] += 1
        counters["inference_ms"] += elapsed_ms(start_ms)
    return policy, value


def build_performance(config: MCTSConfig, counters: dict[str, float] | None, total_ms: float) -> MCTSPerformanceStats | None:
    if not config.collect_stats or counters is None:
        return None
    return MCTSPerformanceStats.from_totals(
        simulations=config.simulations,
        expanded_nodes=int(counters["expanded_nodes"]),
        inference_calls=int(counters["inference_calls"]),
        inference_ms=float(counters["inference_ms"]),
        legal_move_generations=int(counters["legal_move_generations"]),
        total_ms=total_ms,
    )
