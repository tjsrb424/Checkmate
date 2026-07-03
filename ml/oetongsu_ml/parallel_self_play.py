from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .inference import RandomPolicyValueModel, TorchAlphaZeroModel
from .performance import SelfPlayPerformanceStats, aggregate_self_play_performance, elapsed_ms, now_ms, rate_per_sec
from .ruleset import RulesetId
from .self_play import SelfPlayConfig, play_self_play_game, self_play_samples_to_jsonl

WorkerStatus = Literal["completed", "failed"]
RunStatus = Literal["completed", "failed"]


@dataclass
class ParallelSelfPlayConfig:
    games: int = 10
    workers: int = 2
    simulations: int = 64
    max_plies: int = 120
    temperature: float = 1.0
    temperature_drop_ply: int = 20
    seed: int = 1
    ruleset_id: RulesetId = "kakao-like"
    output: str = "../data/selfplay/parallel_selfplay_latest.jsonl"
    summary: str = "../data/selfplay/parallel_selfplay_latest_summary.json"
    shard_dir: str = "../data/selfplay/shards"
    model_checkpoint: str | None = None
    random_model: bool = False


@dataclass
class WorkerSelfPlayResult:
    worker_id: int
    games: int
    sample_count: int
    shard_path: str
    summary_path: str
    startedAt: str
    endedAt: str
    status: WorkerStatus
    game_summaries: list[dict[str, Any]] = field(default_factory=list)
    performance: SelfPlayPerformanceStats | None = None
    error: str | None = None


@dataclass
class ParallelSelfPlayResult:
    runId: str
    games: int
    workers: int
    sample_count: int
    output: str
    summary: str
    shard_dir: str
    shards: list[str]
    worker_summaries: list[WorkerSelfPlayResult]
    startedAt: str
    endedAt: str
    status: RunStatus
    total_ms: float = 0.0
    games_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    mcts_total_ms: float = 0.0
    inference_total_ms: float = 0.0
    slowest_worker: dict[str, Any] | None = None
    fastest_worker: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    partial: bool = False


def split_games(total_games: int, workers: int) -> list[int]:
    if total_games <= 0:
        raise ValueError("total_games must be greater than zero")
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    effective_workers = min(total_games, workers)
    base = total_games // effective_workers
    remainder = total_games % effective_workers
    return [base + (1 if index < remainder else 0) for index in range(effective_workers)]


def run_parallel_self_play(config: ParallelSelfPlayConfig | None = None) -> ParallelSelfPlayResult:
    cfg = config or ParallelSelfPlayConfig()
    run_start_ms = now_ms()
    game_counts = split_games(cfg.games, cfg.workers)
    run_id = uuid4().hex[:12]
    started_at = utc_now()
    shard_dir = Path(cfg.shard_dir)
    output_path = Path(cfg.output)
    summary_path = Path(cfg.summary)
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[WorkerSelfPlayResult] = []
    errors: list[str] = []
    if len(game_counts) == 1:
        results.append(_run_worker(0, game_counts[0], run_id, cfg))
    else:
        with ProcessPoolExecutor(max_workers=len(game_counts)) as executor:
            futures = {
                executor.submit(_run_worker, worker_id, games, run_id, cfg): worker_id
                for worker_id, games in enumerate(game_counts)
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    results.append(future.result())
                except BaseException as error:
                    errors.append(f"worker {worker_id}: {error}")

    results.sort(key=lambda item: item.worker_id)
    errors.extend(result.error for result in results if result.error)
    failed = bool(errors) or any(result.status == "failed" for result in results)
    total_ms = elapsed_ms(run_start_ms)
    performance_summary = summarize_worker_performance(results, total_ms)

    if failed:
        run_result = ParallelSelfPlayResult(
            runId=run_id,
            games=cfg.games,
            workers=len(game_counts),
            sample_count=sum(result.sample_count for result in results),
            output=str(output_path),
            summary=str(summary_path),
            shard_dir=str(shard_dir),
            shards=[result.shard_path for result in results],
            worker_summaries=results,
            startedAt=started_at,
            endedAt=utc_now(),
            status="failed",
            **performance_summary,
            errors=errors,
            partial=True,
        )
        write_parallel_summary(summary_path, run_result)
        raise RuntimeError("; ".join(errors) or "parallel self-play failed")

    merge_shards([Path(result.shard_path) for result in results], output_path)
    run_result = ParallelSelfPlayResult(
        runId=run_id,
        games=cfg.games,
        workers=len(game_counts),
        sample_count=sum(result.sample_count for result in results),
        output=str(output_path),
        summary=str(summary_path),
        shard_dir=str(shard_dir),
        shards=[result.shard_path for result in results],
        worker_summaries=results,
        startedAt=started_at,
        endedAt=utc_now(),
        status="completed",
        **performance_summary,
        errors=[],
        partial=False,
    )
    write_parallel_summary(summary_path, run_result)
    return run_result


def _run_worker(worker_id: int, games: int, run_id: str, cfg: ParallelSelfPlayConfig) -> WorkerSelfPlayResult:
    worker_start_ms = now_ms()
    started_at = utc_now()
    shard_dir = Path(cfg.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"parallel_{run_id}_worker_{worker_id:03d}.jsonl"
    summary_path = shard_dir / f"parallel_{run_id}_worker_{worker_id:03d}_summary.json"
    worker_seed = cfg.seed + worker_id * 100000
    all_samples = []
    summaries: list[dict[str, Any]] = []
    game_performances: list[SelfPlayPerformanceStats] = []
    try:
        model = create_worker_model(cfg, worker_seed)
        for game_index in range(games):
            game_seed = worker_seed + game_index
            result = play_self_play_game(
                model,
                SelfPlayConfig(
                    game_id=f"parallel-{run_id}-w{worker_id:03d}-{game_index + 1:06d}",
                    max_plies=cfg.max_plies,
                    mcts_simulations=cfg.simulations,
                    temperature=cfg.temperature,
                    temperature_drop_ply=cfg.temperature_drop_ply,
                    seed=game_seed,
                    ruleset_id=cfg.ruleset_id,
                ),
            )
            all_samples.extend(result.samples)
            summaries.append(result.to_summary())
            if result.performance is not None:
                game_performances.append(result.performance)

        shard_path.write_text(self_play_samples_to_jsonl(all_samples), encoding="utf-8")
        performance = aggregate_worker_performance(games, len(all_samples), game_performances, elapsed_ms(worker_start_ms))
        worker_result = WorkerSelfPlayResult(
            worker_id=worker_id,
            games=games,
            sample_count=len(all_samples),
            shard_path=str(shard_path),
            summary_path=str(summary_path),
            startedAt=started_at,
            endedAt=utc_now(),
            status="completed",
            game_summaries=summaries,
            performance=performance,
        )
    except BaseException as error:
        performance = aggregate_worker_performance(len(summaries), len(all_samples), game_performances, elapsed_ms(worker_start_ms))
        worker_result = WorkerSelfPlayResult(
            worker_id=worker_id,
            games=games,
            sample_count=len(all_samples),
            shard_path=str(shard_path),
            summary_path=str(summary_path),
            startedAt=started_at,
            endedAt=utc_now(),
            status="failed",
            game_summaries=summaries,
            performance=performance,
            error=str(error),
        )
    summary_path.write_text(json.dumps(asdict(worker_result), indent=2), encoding="utf-8")
    return worker_result


def create_worker_model(cfg: ParallelSelfPlayConfig, seed: int):
    if cfg.random_model or not cfg.model_checkpoint:
        return RandomPolicyValueModel(seed=seed)
    return TorchAlphaZeroModel(cfg.model_checkpoint)


def merge_shards(shards: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for shard in shards:
            if not shard.exists():
                raise FileNotFoundError(f"missing shard: {shard}")
            destination.write(shard.read_text(encoding="utf-8"))


def write_parallel_summary(path: str | Path, result: ParallelSelfPlayResult) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def aggregate_worker_performance(
    games: int,
    sample_count: int,
    game_performances: list[SelfPlayPerformanceStats],
    total_ms: float,
) -> SelfPlayPerformanceStats:
    aggregate = aggregate_self_play_performance(game_performances)
    return SelfPlayPerformanceStats.from_totals(
        games=games,
        plies=aggregate.plies,
        samples=sample_count,
        total_ms=total_ms,
        mcts_total_ms=aggregate.mcts_total_ms,
        inference_total_ms=aggregate.inference_total_ms,
    )


def summarize_worker_performance(results: list[WorkerSelfPlayResult], total_ms: float) -> dict[str, Any]:
    performances = [result.performance for result in results if result.performance is not None]
    aggregate = aggregate_self_play_performance(performances)
    completed = [result for result in results if result.performance is not None]
    slowest = max(completed, key=lambda item: item.performance.total_ms if item.performance else 0.0, default=None)
    fastest = min(completed, key=lambda item: item.performance.total_ms if item.performance else 0.0, default=None)
    return {
        "total_ms": total_ms,
        "games_per_sec": rate_per_sec(sum(result.games for result in results), total_ms),
        "samples_per_sec": rate_per_sec(sum(result.sample_count for result in results), total_ms),
        "mcts_total_ms": aggregate.mcts_total_ms,
        "inference_total_ms": aggregate.inference_total_ms,
        "slowest_worker": worker_brief(slowest),
        "fastest_worker": worker_brief(fastest),
    }


def worker_brief(result: WorkerSelfPlayResult | None) -> dict[str, Any] | None:
    if result is None or result.performance is None:
        return None
    return {
        "worker_id": result.worker_id,
        "games": result.games,
        "sample_count": result.sample_count,
        "total_ms": result.performance.total_ms,
        "samples_per_sec": result.performance.samples_per_sec,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Oetongsu self-play JSONL with parallel workers.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--maxPlies", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatureDropPly", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ruleset", choices=["oetongsu-basic", "kakao-like", "kja-like"], default="kakao-like")
    parser.add_argument("--output", default="../data/selfplay/parallel_selfplay_latest.jsonl")
    parser.add_argument("--summary", default="../data/selfplay/parallel_selfplay_latest_summary.json")
    parser.add_argument("--shardDir", default="../data/selfplay/shards")
    parser.add_argument("--model", default=None)
    parser.add_argument("--randomModel", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ParallelSelfPlayConfig:
    return ParallelSelfPlayConfig(
        games=args.games,
        workers=args.workers,
        simulations=args.simulations,
        max_plies=args.maxPlies,
        temperature=args.temperature,
        temperature_drop_ply=args.temperatureDropPly,
        seed=args.seed,
        ruleset_id=args.ruleset,
        output=args.output,
        summary=args.summary,
        shard_dir=args.shardDir,
        model_checkpoint=args.model,
        random_model=args.randomModel,
    )


def main(argv: list[str] | None = None) -> None:
    result = run_parallel_self_play(config_from_args(parse_args(argv)))
    print("Parallel self-play complete")
    print(f"runId: {result.runId}")
    print(f"games: {result.games}")
    print(f"workers: {result.workers}")
    print(f"samples: {result.sample_count}")
    print(f"games/sec: {result.games_per_sec:.3f}")
    print(f"samples/sec: {result.samples_per_sec:.3f}")
    print(f"duration_ms: {result.total_ms:.1f}")
    print(f"output: {result.output}")
    print(f"summary: {result.summary}")


if __name__ == "__main__":
    main()
