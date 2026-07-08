from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .inference import RandomPolicyValueModel, TorchAlphaZeroModel
from .model_arena import ModelArenaConfig, RandomModelPlayer, TorchModelPlayer, run_model_arena
from .model_registry import find_entry, get_latest_promoted, load_registry, promote_candidate, register_candidate, reject_candidate, save_registry
from .parallel_self_play import ParallelSelfPlayConfig, run_parallel_self_play_with_progress
from .performance import SelfPlayPerformanceStats, aggregate_self_play_performance
from .progress_reporter import ProgressReporter, clamp_percent
from .ruleset import RulesetId
from .self_play import SelfPlayConfig, play_self_play_game, self_play_samples_to_jsonl
from .cheap_validation_gate import summarize_gate_result
from .train_alphazero import train_alphazero

AutoTrainStatus = Literal["idle", "running", "completed", "failed"]


@dataclass
class AutoTrainConfig:
    iterations: int = 1
    games_per_iteration: int = 10
    simulations: int = 64
    arena_simulations: int | None = None
    max_plies: int = 120
    train_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    channels: int = 64
    promotion_games: int = 20
    promotion_threshold: float = 0.55
    adjudication_draw_margin: float = 0.0
    temperature: float = 1.0
    temperature_drop_ply: int = 20
    seed: int = 1
    ruleset_id: RulesetId = "kakao-like"
    registry_path: str = "../data/models/registry.json"
    training_dir: str = "../data/training"
    selfplay_dir: str = "../data/selfplay"
    shard_dir: str = "../data/selfplay/shards"
    model_dir: str = "../data/models"
    arena_dir: str = "../data/models/arena"
    initial_champion: str | None = None
    allow_random_champion: bool = False
    resume: bool = False
    resume_champion: bool = True
    quick: bool = False
    strict: bool = False
    selfplay_workers: int = 1
    parallel_selfplay: bool = False
    progress_path: str = "../data/training/progress.json"
    progress_events_path: str = "../data/training/progress_events.jsonl"
    cheap_validation_before_arena: bool = False
    cheap_validation_games: int = 4
    cheap_validation_simulations: int = 8
    cheap_validation_max_plies: int = 80
    cheap_validation_min_score_rate: float = 0.25
    cheap_validation_fail_fast: bool = False

    def resolved(self) -> "AutoTrainConfig":
        if not self.quick:
            return self
        return AutoTrainConfig(
            iterations=1,
            games_per_iteration=1,
            simulations=2,
            arena_simulations=2,
            max_plies=4,
            train_epochs=1,
            batch_size=2,
            learning_rate=self.learning_rate,
            channels=4,
            promotion_games=2,
            promotion_threshold=self.promotion_threshold,
            adjudication_draw_margin=self.adjudication_draw_margin,
            temperature=self.temperature,
            temperature_drop_ply=self.temperature_drop_ply,
            seed=self.seed,
            ruleset_id=self.ruleset_id,
            registry_path=self.registry_path,
            training_dir=self.training_dir,
            selfplay_dir=self.selfplay_dir,
            shard_dir=self.shard_dir,
            model_dir=self.model_dir,
            arena_dir=self.arena_dir,
            initial_champion=self.initial_champion,
            allow_random_champion=True,
            resume=self.resume,
            resume_champion=self.resume_champion,
            quick=True,
            strict=self.strict,
            selfplay_workers=self.selfplay_workers,
            parallel_selfplay=self.parallel_selfplay,
            progress_path=self.progress_path,
            progress_events_path=self.progress_events_path,
            cheap_validation_before_arena=self.cheap_validation_before_arena,
            cheap_validation_games=min(self.cheap_validation_games, 4),
            cheap_validation_simulations=min(self.cheap_validation_simulations, 8),
            cheap_validation_max_plies=min(self.cheap_validation_max_plies, 80),
            cheap_validation_min_score_rate=self.cheap_validation_min_score_rate,
            cheap_validation_fail_fast=self.cheap_validation_fail_fast,
        )


@dataclass
class AutoTrainState:
    runId: str
    startedAt: str
    updatedAt: str
    currentIteration: int = 0
    completedIterations: int = 0
    latestChampionVersion: str | None = None
    latestCandidateVersion: str | None = None
    lastSelfPlayPath: str | None = None
    lastCandidatePath: str | None = None
    lastArenaResultPath: str | None = None
    status: AutoTrainStatus = "idle"


@dataclass
class AutoTrainIterationResult:
    iteration: int
    startedAt: str
    endedAt: str
    championVersion: str | None
    candidateVersion: str
    selfplayPath: str
    selfplaySummaryPath: str
    sampleCount: int
    modelPath: str
    checkpointPath: str
    metadataPath: str
    arenaResultPath: str
    candidateScoreRate: float
    championScoreRate: float
    promoted: bool
    status: Literal["promoted", "rejected"]
    metrics: dict[str, Any] = field(default_factory=dict)
    cheapValidation: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoTrainRunResult:
    runId: str
    iterations: list[AutoTrainIterationResult]
    completedIterations: int
    promotions: int
    rejections: int
    latestChampionVersion: str | None
    logPath: str
    summaryPath: str


@dataclass
class CandidateResumeResolution:
    championVersion: str | None
    championPath: str | None
    resumePath: str | None
    resumeCandidateFromChampion: bool
    latestCandidateVersion: str | None = None
    latestCandidateStatus: str | None = None
    latestCandidatePath: str | None = None


def run_autotrain(config: AutoTrainConfig | None = None) -> AutoTrainRunResult:
    cfg = (config or AutoTrainConfig()).resolved()
    ensure_directories(cfg)
    training_dir = Path(cfg.training_dir)
    state_path = training_dir / "autotrain_state.json"
    log_path = training_dir / "autotrain_log.jsonl"
    summary_path = training_dir / "autotrain_summary.json"
    progress_path = resolved_progress_path(cfg.progress_path, cfg.training_dir, "progress.json")
    progress_events_path = resolved_progress_path(cfg.progress_events_path, cfg.training_dir, "progress_events.jsonl")

    state = load_autotrain_state(state_path) if cfg.resume and state_path.exists() else create_initial_state(cfg)
    state.status = "running"
    state.updatedAt = utc_now()
    save_autotrain_state(state_path, state)
    reporter = ProgressReporter(progress_path, progress_events_path, state.runId, cfg.iterations)
    reporter.update(
        phase="selfplay",
        phase_percent=0,
        message="AutoTrain started",
        message_ko="AutoTrain을 시작했습니다.",
        current_iteration=max(1, state.currentIteration or 1),
        completed_iterations=state.completedIterations,
    )

    completed: list[AutoTrainIterationResult] = []
    try:
        registry = load_registry(cfg.registry_path)
        start_iteration = state.completedIterations + 1 if cfg.resume else 1
        for iteration in range(start_iteration, cfg.iterations + 1):
            completed_before_current = state.completedIterations
            reporter.start_iteration()
            state.currentIteration = iteration
            state.updatedAt = utc_now()
            save_autotrain_state(state_path, state)

            result = run_autotrain_iteration(
                cfg,
                registry,
                iteration,
                reporter,
                completed_before_current,
                latest_candidate_version=state.latestCandidateVersion,
                run_id=state.runId,
            )
            completed.append(result)
            append_iteration_log(log_path, result)

            state.latestCandidateVersion = result.candidateVersion
            state.latestChampionVersion = result.candidateVersion if result.promoted else result.championVersion
            state.lastSelfPlayPath = result.selfplayPath
            state.lastCandidatePath = result.checkpointPath
            state.lastArenaResultPath = result.arenaResultPath
            state.updatedAt = utc_now()
            save_autotrain_state(state_path, state)
            previous_progress = reporter.last_snapshot
            reporter.update(
                phase="package",
                phase_percent=100,
                message="Iteration result saved",
                message_ko="학습 회차 결과를 저장했습니다.",
                current_iteration=iteration,
                completed_iterations=completed_before_current,
                self_play=previous_progress.selfPlay if previous_progress else {},
                training=previous_progress.training if previous_progress else {},
                arena=previous_progress.arena if previous_progress else {},
                models={
                    **(previous_progress.models if previous_progress else {}),
                    "championVersion": result.championVersion,
                    "candidateVersion": result.candidateVersion,
                    "latestPromotedVersion": state.latestChampionVersion,
                    "promotionThreshold": cfg.promotion_threshold,
                },
                result={"promoted": result.promoted, "status": result.status, "cheapValidation": result.cheapValidation},
                progress_accuracy="iteration_complete",
            )
            state.completedIterations = iteration
            state.updatedAt = utc_now()
            save_autotrain_state(state_path, state)

            if cfg.strict and result.metrics.get("arena", {}).get("illegalMoves", 0) + result.metrics.get("arena", {}).get("forfeits", 0) > 0:
                raise RuntimeError("strict autotrain stopped after illegal moves or forfeits")

        state.status = "completed"
        state.updatedAt = utc_now()
        save_autotrain_state(state_path, state)
        run_result = create_run_result(state, completed, log_path, summary_path)
        write_summary(summary_path, run_result)
        write_markdown_summary(training_dir / "autotrain_summary.md", run_result)
        previous_progress = reporter.last_snapshot
        reporter.update(
            status="completed",
            phase="completed",
            phase_percent=100,
            message="AutoTrain completed",
            message_ko="AutoTrain이 완료되었습니다.",
            current_iteration=max(1, state.currentIteration),
            completed_iterations=state.completedIterations,
            self_play=previous_progress.selfPlay if previous_progress else {},
            training=previous_progress.training if previous_progress else {},
            arena=previous_progress.arena if previous_progress else {},
            models={
                **(previous_progress.models if previous_progress else {}),
                "latestPromotedVersion": state.latestChampionVersion,
            },
            result={"status": "completed"},
            progress_accuracy="terminal_status",
        )
        return run_result
    except BaseException as error:
        state.status = "failed"
        state.updatedAt = utc_now()
        save_autotrain_state(state_path, state)
        reporter.mark_failed(error, max(1, state.currentIteration or 1), state.completedIterations)
        raise


def run_autotrain_iteration(
    cfg: AutoTrainConfig,
    registry: dict[str, Any],
    iteration: int,
    reporter: ProgressReporter | None = None,
    completed_iterations: int = 0,
    latest_candidate_version: str | None = None,
    run_id: str | None = None,
) -> AutoTrainIterationResult:
    started_at = utc_now()
    resume_resolution = resolve_candidate_resume_checkpoint(cfg, registry, latest_candidate_version)
    champion_path = resume_resolution.championPath
    champion_version = resume_resolution.championVersion
    if champion_path is None and not cfg.allow_random_champion:
        raise RuntimeError("no promoted champion found; pass --allowRandomChampion, --quick, or --initialChampion")
    candidate_version = f"az_iter_{iteration:06d}"
    models = {
        "championVersion": champion_version,
        "candidateVersion": candidate_version,
        "latestPromotedVersion": champion_version,
        "promotionThreshold": cfg.promotion_threshold,
        "resumeSourceVersion": resume_resolution.championVersion,
        "resumeSourcePath": resume_resolution.resumePath,
        "resumeCandidateFromChampion": resume_resolution.resumeCandidateFromChampion,
        "latestCandidateVersion": resume_resolution.latestCandidateVersion,
        "latestCandidateStatus": resume_resolution.latestCandidateStatus,
    }
    if reporter:
        reporter.update(
            phase="selfplay",
            phase_percent=0,
            message="Generating self-play games",
            message_ko="후보 AI가 배울 자기대국을 생성하는 중입니다.",
            current_iteration=iteration,
            completed_iterations=completed_iterations,
            self_play={
                "currentGames": 0,
                "totalGames": cfg.games_per_iteration,
                "currentSamples": 0,
                "workers": max(1, cfg.selfplay_workers),
                "mode": "parallel" if cfg.parallel_selfplay or cfg.selfplay_workers > 1 else "sequential",
            },
            training={"currentEpoch": 0, "totalEpochs": cfg.train_epochs, "currentBatch": 0, "totalBatches": 0},
            arena={"currentGames": 0, "totalGames": cfg.promotion_games, "illegalMoves": 0, "forfeits": 0},
            models=models,
        )

    selfplay_path, selfplay_summary_path, sample_count, selfplay_metrics = generate_self_play(
        cfg,
        iteration,
        champion_path,
        progress_callback=selfplay_progress_callback(reporter, cfg, iteration, completed_iterations, models),
    )
    checkpoint_path = Path(cfg.model_dir) / "checkpoints" / f"{candidate_version}.pt"
    resume_path = resume_resolution.resumePath
    train_metrics = train_alphazero(
        data=selfplay_path,
        output=checkpoint_path,
        epochs=cfg.train_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        seed=cfg.seed + iteration,
        channels=cfg.channels,
        resume=resume_path,
        training_metadata={
            "source": "autotrain",
            "run_id": run_id,
            "model_version": candidate_version,
            "candidate_version": candidate_version,
            "champion_version": champion_version,
            "resume_version": resume_resolution.championVersion,
            "resumeCandidateFromChampion": resume_resolution.resumeCandidateFromChampion,
            "latestCandidateVersion": resume_resolution.latestCandidateVersion,
            "latestCandidateStatus": resume_resolution.latestCandidateStatus,
        },
        progress_callback=training_progress_callback(reporter, cfg, iteration, completed_iterations, models),
    )
    metadata_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_metrics.json")

    cheap_validation = run_cheap_validation_if_enabled(cfg, candidate_version, checkpoint_path, champion_path, reporter, iteration, completed_iterations, models)

    register_candidate(
        registry,
        version=candidate_version,
        path=str(checkpoint_path),
        metadata_path=str(metadata_path),
        parent_version=champion_version,
        metrics={"train": latest_train_metrics(train_metrics), "sampleCount": sample_count, "selfPlay": selfplay_metrics, "cheapValidation": cheap_validation},
    )

    arena_result_path = Path(cfg.arena_dir) / f"{candidate_version}_arena.json"
    if cheap_validation.get("status") == "fail" and cfg.cheap_validation_fail_fast:
        arena_json = cheap_validation_rejection_arena(cheap_validation)
    else:
        arena_result = run_candidate_arena(
            cfg,
            candidate_version,
            checkpoint_path,
            champion_path,
            progress_callback=arena_progress_callback(reporter, cfg, iteration, completed_iterations, models, cheap_validation),
        )
        arena_json = arena_result.to_json()
    arena_result_path.parent.mkdir(parents=True, exist_ok=True)
    arena_result_path.write_text(json.dumps(arena_json, indent=2), encoding="utf-8")

    if bool(arena_json.get("promoted")):
        promote_candidate(registry, candidate_version, arena_json)
        status: Literal["promoted", "rejected"] = "promoted"
    else:
        reject_candidate(registry, candidate_version, arena_json)
        status = "rejected"
    save_registry(cfg.registry_path, registry)

    return AutoTrainIterationResult(
        iteration=iteration,
        startedAt=started_at,
        endedAt=utc_now(),
        championVersion=champion_version,
        candidateVersion=candidate_version,
        selfplayPath=str(selfplay_path),
        selfplaySummaryPath=str(selfplay_summary_path),
        sampleCount=sample_count,
        modelPath=str(checkpoint_path),
        checkpointPath=str(checkpoint_path),
        metadataPath=str(metadata_path),
        arenaResultPath=str(arena_result_path),
        candidateScoreRate=float(arena_json.get("candidateScoreRate") or 0.0),
        championScoreRate=float(arena_json.get("championScoreRate") or 0.0),
        promoted=bool(arena_json.get("promoted")),
        status=status,
        metrics={"selfPlay": selfplay_metrics, "train": train_metrics, "arena": arena_json, "cheapValidation": cheap_validation},
        cheapValidation=cheap_validation,
    )


def generate_self_play(
    cfg: AutoTrainConfig,
    iteration: int,
    champion_path: str | None,
    progress_callback=None,
) -> tuple[Path, Path, int, dict[str, Any]]:
    selfplay_path = Path(cfg.selfplay_dir) / f"az_iter_{iteration:06d}.jsonl"
    summary_path = Path(cfg.selfplay_dir) / f"az_iter_{iteration:06d}_summary.json"
    if cfg.parallel_selfplay or cfg.selfplay_workers > 1:
        result = run_parallel_self_play_with_progress(
            ParallelSelfPlayConfig(
                games=cfg.games_per_iteration,
                workers=max(1, cfg.selfplay_workers),
                simulations=cfg.simulations,
                max_plies=cfg.max_plies,
                temperature=cfg.temperature,
                temperature_drop_ply=cfg.temperature_drop_ply,
                seed=cfg.seed + iteration * 1000,
                ruleset_id=cfg.ruleset_id,
                output=str(selfplay_path),
                summary=str(summary_path),
                shard_dir=cfg.shard_dir,
                model_checkpoint=champion_path,
                random_model=champion_path is None,
            ),
            progress_callback=progress_callback,
        )
        return selfplay_path, summary_path, result.sample_count, {
            "mode": "parallel",
            "workers": result.workers,
            "shardCount": len(result.shards),
            "shards": result.shards,
            "sampleCount": result.sample_count,
            "runId": result.runId,
            "gamesPerSecond": result.games_per_sec,
            "samplesPerSecond": result.samples_per_sec,
            "totalSelfPlayMs": result.total_ms,
            "totalInferenceMs": result.inference_total_ms,
            "totalMctsMs": result.mcts_total_ms,
            "slowestWorker": result.slowest_worker,
            "fastestWorker": result.fastest_worker,
        }

    model = TorchAlphaZeroModel(champion_path) if champion_path else RandomPolicyValueModel(seed=cfg.seed + iteration)
    all_samples = []
    summaries = []
    performances: list[SelfPlayPerformanceStats] = []
    for game_index in range(cfg.games_per_iteration):
        result = play_self_play_game(
            model,
            SelfPlayConfig(
                game_id=f"az-iter-{iteration:06d}-{game_index + 1:06d}",
                max_plies=cfg.max_plies,
                mcts_simulations=cfg.simulations,
                temperature=cfg.temperature,
                temperature_drop_ply=cfg.temperature_drop_ply,
                seed=cfg.seed + iteration * 1000 + game_index,
                ruleset_id=cfg.ruleset_id,
            ),
        )
        all_samples.extend(result.samples)
        summaries.append(result.to_summary())
        if result.performance is not None:
            performances.append(result.performance)
        if progress_callback:
            progress_callback(
                {
                    "currentGames": game_index + 1,
                    "totalGames": cfg.games_per_iteration,
                    "currentSamples": len(all_samples),
                    "workers": 1,
                    "mode": "sequential",
                }
            )

    selfplay_path.parent.mkdir(parents=True, exist_ok=True)
    selfplay_path.write_text(self_play_samples_to_jsonl(all_samples), encoding="utf-8")
    summary = {
        "games": cfg.games_per_iteration,
        "workers": 1,
        "sample_count": len(all_samples),
        "shard_count": 0,
        "shards": [],
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    performance = aggregate_self_play_performance(performances)
    return selfplay_path, summary_path, len(all_samples), {
        "mode": "sequential",
        "workers": 1,
        "shardCount": 0,
        "shards": [],
        "sampleCount": len(all_samples),
        "gamesPerSecond": performance.games_per_sec,
        "samplesPerSecond": performance.samples_per_sec,
        "totalSelfPlayMs": performance.total_ms,
        "totalInferenceMs": performance.inference_total_ms,
        "totalMctsMs": performance.mcts_total_ms,
        "slowestWorker": None,
        "fastestWorker": None,
    }


def run_candidate_arena(cfg: AutoTrainConfig, candidate_version: str, checkpoint_path: Path, champion_path: str | None, progress_callback=None):
    candidate = TorchModelPlayer(checkpoint_path, name=candidate_version)
    champion = TorchModelPlayer(champion_path, name="champion") if champion_path else RandomModelPlayer(name="champion", seed=cfg.seed + 99)
    return run_model_arena(
        candidate,
        champion,
        ModelArenaConfig(
            games=cfg.promotion_games,
            simulations=cfg.arena_simulations or cfg.simulations,
            max_plies=cfg.max_plies,
            temperature=0.0,
            seed=cfg.seed,
            promotion_threshold=cfg.promotion_threshold,
            ruleset_id=cfg.ruleset_id,
            adjudication_draw_margin=cfg.adjudication_draw_margin,
        ),
        progress_callback=progress_callback,
    )


def resolve_candidate_resume_checkpoint(
    cfg: AutoTrainConfig,
    registry: dict[str, Any],
    latest_candidate_version: str | None = None,
) -> CandidateResumeResolution:
    champion_entry = get_latest_promoted(registry)
    champion_path = cfg.initial_champion or (champion_entry.get("path") if champion_entry else None)
    champion_version = champion_entry.get("version") if champion_entry else ("initial_champion" if cfg.initial_champion else None)
    latest_candidate = find_entry(registry, latest_candidate_version) if latest_candidate_version else None
    latest_candidate_path = latest_candidate.get("path") if latest_candidate else None
    latest_candidate_status = latest_candidate.get("status") if latest_candidate else None
    resume_path = champion_path if cfg.resume_champion and champion_path else None
    if (
        resume_path
        and latest_candidate
        and latest_candidate_status != "promoted"
        and latest_candidate_path
        and Path(resume_path) == Path(latest_candidate_path)
    ):
        raise RuntimeError("candidate resume source resolved to a rejected/latest candidate checkpoint")
    return CandidateResumeResolution(
        championVersion=champion_version,
        championPath=champion_path,
        resumePath=resume_path,
        resumeCandidateFromChampion=bool(resume_path and champion_path and Path(resume_path) == Path(champion_path)),
        latestCandidateVersion=latest_candidate_version,
        latestCandidateStatus=latest_candidate_status,
        latestCandidatePath=latest_candidate_path,
    )


def run_cheap_validation_if_enabled(
    cfg: AutoTrainConfig,
    candidate_version: str,
    checkpoint_path: Path,
    champion_path: str | None,
    reporter: ProgressReporter | None,
    iteration: int,
    completed_iterations: int,
    models: dict[str, Any],
) -> dict[str, Any]:
    if not cfg.cheap_validation_before_arena:
        return {"enabled": False, "status": "skipped", "warnings": []}
    if champion_path is None:
        return {"enabled": True, "status": "skipped", "warnings": ["no champion checkpoint for cheap validation"]}
    result = run_model_arena(
        TorchModelPlayer(checkpoint_path, name=candidate_version),
        TorchModelPlayer(champion_path, name="champion"),
        ModelArenaConfig(
            games=cfg.cheap_validation_games,
            simulations=cfg.cheap_validation_simulations,
            max_plies=cfg.cheap_validation_max_plies,
            temperature=0.0,
            seed=cfg.seed,
            promotion_threshold=cfg.cheap_validation_min_score_rate,
            ruleset_id=cfg.ruleset_id,
            adjudication_draw_margin=cfg.adjudication_draw_margin,
        ),
    ).to_json()
    output_path = Path(cfg.training_dir) / f"{candidate_version}_cheap_validation.json"
    payload = summarize_gate_result(
        result,
        candidate=str(checkpoint_path),
        champion=str(champion_path),
        simulations=cfg.cheap_validation_simulations,
        max_plies=cfg.cheap_validation_max_plies,
        adjudication_draw_margin=cfg.adjudication_draw_margin,
        min_score_rate=cfg.cheap_validation_min_score_rate,
    )
    payload["enabled"] = True
    payload["outputPath"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if reporter:
        reporter.update(
            phase="arena",
            phase_percent=5,
            message="Cheap validation gate completed",
            message_ko="Cheap validation gate媛 ?꾨즺?섏뿀?듬땲??",
            current_iteration=iteration,
            completed_iterations=completed_iterations,
            self_play={"currentGames": cfg.games_per_iteration, "totalGames": cfg.games_per_iteration},
            training={"currentEpoch": cfg.train_epochs, "totalEpochs": cfg.train_epochs},
            arena={"cheapValidation": compact_cheap_validation(payload)},
            models=models,
            result={"cheapValidation": compact_cheap_validation(payload)},
            progress_accuracy="cheap_validation_gate",
        )
    return payload


def compact_cheap_validation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": payload.get("enabled", False),
        "status": payload.get("status", "skipped"),
        "candidateScoreRate": payload.get("candidateScoreRate"),
        "games": payload.get("games"),
        "simulations": payload.get("simulations"),
        "maxPlies": payload.get("maxPlies"),
        "warnings": payload.get("warnings", []),
        "outputPath": payload.get("outputPath"),
    }


def cheap_validation_rejection_arena(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "games": int(payload.get("games") or 0),
        "candidateWins": int(payload.get("candidateWins") or 0),
        "championWins": int(payload.get("championWins") or 0),
        "draws": int(payload.get("draws") or 0),
        "candidateScoreRate": float(payload.get("candidateScoreRate") or 0.0),
        "championScoreRate": 1.0 - float(payload.get("candidateScoreRate") or 0.0),
        "averagePlies": float(payload.get("averagePlies") or 0.0),
        "promoted": False,
        "illegalMoves": int(payload.get("illegalMoves") or 0),
        "forfeits": int(payload.get("forfeits") or 0),
        "gameSummaries": payload.get("rawArenaResult", {}).get("gameSummaries", []),
        "pairedSummary": payload.get("pairedSummary"),
        "cheapValidationFailFast": True,
        "cheapValidation": compact_cheap_validation(payload),
    }


def selfplay_progress_callback(
    reporter: ProgressReporter | None,
    cfg: AutoTrainConfig,
    iteration: int,
    completed_iterations: int,
    models: dict[str, Any],
):
    if reporter is None:
        return None

    def callback(payload: dict[str, Any]) -> None:
        current = int(payload.get("currentGames") or 0)
        total = max(1, int(payload.get("totalGames") or cfg.games_per_iteration))
        reporter.update(
            phase="selfplay",
            phase_percent=clamp_percent(current / total * 100),
            message="Generating self-play games",
            message_ko="후보 AI가 배울 자기대국을 생성하는 중입니다.",
            current_iteration=iteration,
            completed_iterations=completed_iterations,
            self_play=payload,
            training={"currentEpoch": 0, "totalEpochs": cfg.train_epochs, "currentBatch": 0, "totalBatches": 0},
            arena={"currentGames": 0, "totalGames": cfg.promotion_games, "illegalMoves": 0, "forfeits": 0},
            models=models,
        )

    return callback


def training_progress_callback(
    reporter: ProgressReporter | None,
    cfg: AutoTrainConfig,
    iteration: int,
    completed_iterations: int,
    models: dict[str, Any],
):
    if reporter is None:
        return None

    def callback(payload: dict[str, Any]) -> None:
        current_epoch = int(payload.get("currentEpoch") or 1)
        total_epochs = max(1, int(payload.get("totalEpochs") or cfg.train_epochs))
        current_batch = int(payload.get("currentBatch") or 0)
        total_batches = max(1, int(payload.get("totalBatches") or 1))
        phase = ((current_epoch - 1) + current_batch / total_batches) / total_epochs * 100
        reporter.update(
            phase="train",
            phase_percent=clamp_percent(phase),
            message="Training candidate model",
            message_ko="후보 AI가 자기대국 데이터를 학습하는 중입니다.",
            current_iteration=iteration,
            completed_iterations=completed_iterations,
            self_play={"currentGames": cfg.games_per_iteration, "totalGames": cfg.games_per_iteration},
            training=payload,
            arena={"currentGames": 0, "totalGames": cfg.promotion_games, "illegalMoves": 0, "forfeits": 0},
            models=models,
        )

    return callback


def arena_progress_callback(
    reporter: ProgressReporter | None,
    cfg: AutoTrainConfig,
    iteration: int,
    completed_iterations: int,
    models: dict[str, Any],
    cheap_validation: dict[str, Any] | None = None,
):
    if reporter is None:
        return None

    def callback(payload: dict[str, Any]) -> None:
        current = int(payload.get("currentGames") or 0)
        total = max(1, int(payload.get("totalGames") or cfg.promotion_games))
        reporter.update(
            phase="arena",
            phase_percent=clamp_percent(current / total * 100),
            message="Running promotion arena",
            message_ko="후보 AI와 현재 챔피언의 승격 대국을 진행하는 중입니다.",
            current_iteration=iteration,
            completed_iterations=completed_iterations,
            self_play={"currentGames": cfg.games_per_iteration, "totalGames": cfg.games_per_iteration},
            training={"currentEpoch": cfg.train_epochs, "totalEpochs": cfg.train_epochs},
            arena={**payload, "cheapValidation": compact_cheap_validation(cheap_validation or {})},
            models=models,
        )

    return callback


def load_autotrain_state(path: str | Path) -> AutoTrainState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AutoTrainState(**raw)


def save_autotrain_state(path: str | Path, state: AutoTrainState) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def create_initial_state(config: AutoTrainConfig) -> AutoTrainState:
    now = utc_now()
    return AutoTrainState(runId=f"autotrain-{uuid4().hex[:12]}", startedAt=now, updatedAt=now)


def append_iteration_log(path: str | Path, result: AutoTrainIterationResult) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), separators=(",", ":")) + "\n")


def create_run_result(state: AutoTrainState, iterations: list[AutoTrainIterationResult], log_path: Path, summary_path: Path) -> AutoTrainRunResult:
    promotions = sum(1 for result in iterations if result.promoted)
    return AutoTrainRunResult(
        runId=state.runId,
        iterations=iterations,
        completedIterations=state.completedIterations,
        promotions=promotions,
        rejections=len(iterations) - promotions,
        latestChampionVersion=state.latestChampionVersion,
        logPath=str(log_path),
        summaryPath=str(summary_path),
    )


def write_summary(path: str | Path, result: AutoTrainRunResult) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def write_markdown_summary(path: str | Path, result: AutoTrainRunResult) -> None:
    lines = [
        "# Oetongsu AutoTrain Summary",
        "",
        f"- runId: {result.runId}",
        f"- completedIterations: {result.completedIterations}",
        f"- promotions: {result.promotions}",
        f"- rejections: {result.rejections}",
        f"- latestChampionVersion: {result.latestChampionVersion}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_train_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history") or []
    return history[-1] if history else {}


def ensure_directories(config: AutoTrainConfig) -> None:
    for path in (
        config.training_dir,
        config.selfplay_dir,
        config.shard_dir,
        config.model_dir,
        Path(config.model_dir) / "checkpoints",
        config.arena_dir,
        Path(config.registry_path).parent,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def resolved_progress_path(configured_path: str, training_dir: str, filename: str) -> str:
    if configured_path == f"../data/training/{filename}" and training_dir != "../data/training":
        return str(Path(training_dir) / filename)
    return configured_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Oetongsu AlphaZero AutoTrain iterations.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--gamesPerIteration", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--arenaSimulations", type=int, default=None)
    parser.add_argument("--maxPlies", type=int, default=120)
    parser.add_argument("--trainEpochs", type=int, default=3)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--promotionGames", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--adjudicationDrawMargin", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatureDropPly", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ruleset", choices=["oetongsu-basic", "kakao-like", "kja-like"], default="kakao-like")
    parser.add_argument("--registry", default="../data/models/registry.json")
    parser.add_argument("--trainingDir", default="../data/training")
    parser.add_argument("--selfplayDir", default="../data/selfplay")
    parser.add_argument("--shardDir", default="../data/selfplay/shards")
    parser.add_argument("--modelDir", default="../data/models")
    parser.add_argument("--arenaDir", default="../data/models/arena")
    parser.add_argument("--initialChampion", default=None)
    parser.add_argument("--allowRandomChampion", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resumeChampion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--selfplayWorkers", type=int, default=1)
    parser.add_argument("--parallelSelfPlay", action="store_true")
    parser.add_argument("--progressPath", default="../data/training/progress.json")
    parser.add_argument("--progressEventsPath", default="../data/training/progress_events.jsonl")
    parser.add_argument("--cheapValidationBeforeArena", action="store_true")
    parser.add_argument("--cheapValidationGames", type=int, default=4)
    parser.add_argument("--cheapValidationSimulations", type=int, default=8)
    parser.add_argument("--cheapValidationMaxPlies", type=int, default=80)
    parser.add_argument("--cheapValidationMinScoreRate", type=float, default=0.25)
    parser.add_argument("--cheapValidationFailFast", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AutoTrainConfig:
    return AutoTrainConfig(
        iterations=args.iterations,
        games_per_iteration=args.gamesPerIteration,
        simulations=args.simulations,
        arena_simulations=args.arenaSimulations,
        max_plies=args.maxPlies,
        train_epochs=args.trainEpochs,
        batch_size=args.batchSize,
        learning_rate=args.lr,
        channels=args.channels,
        promotion_games=args.promotionGames,
        promotion_threshold=args.threshold,
        adjudication_draw_margin=args.adjudicationDrawMargin,
        temperature=args.temperature,
        temperature_drop_ply=args.temperatureDropPly,
        seed=args.seed,
        ruleset_id=args.ruleset,
        registry_path=args.registry,
        training_dir=args.trainingDir,
        selfplay_dir=args.selfplayDir,
        shard_dir=args.shardDir,
        model_dir=args.modelDir,
        arena_dir=args.arenaDir,
        initial_champion=args.initialChampion,
        allow_random_champion=args.allowRandomChampion,
        resume=args.resume,
        resume_champion=args.resumeChampion,
        quick=args.quick,
        strict=args.strict,
        selfplay_workers=args.selfplayWorkers,
        parallel_selfplay=args.parallelSelfPlay,
        progress_path=args.progressPath,
        progress_events_path=args.progressEventsPath,
        cheap_validation_before_arena=args.cheapValidationBeforeArena,
        cheap_validation_games=args.cheapValidationGames,
        cheap_validation_simulations=args.cheapValidationSimulations,
        cheap_validation_max_plies=args.cheapValidationMaxPlies,
        cheap_validation_min_score_rate=args.cheapValidationMinScoreRate,
        cheap_validation_fail_fast=args.cheapValidationFailFast,
    )


def main(argv: list[str] | None = None) -> None:
    result = run_autotrain(config_from_args(parse_args(argv)))
    print("AutoTrain complete")
    print(f"runId: {result.runId}")
    print(f"completedIterations: {result.completedIterations}")
    print(f"promotions: {result.promotions}")
    print(f"rejections: {result.rejections}")
    if result.iterations:
        latest = result.iterations[-1]
        self_play_metrics = latest.metrics.get("selfPlay", {})
        print(f"latestSelfPlaySamplesPerSec: {self_play_metrics.get('samplesPerSecond', 0):.3f}")
        print(f"latestCandidateScoreRate: {latest.candidateScoreRate:.3f}")
    print(f"summary: {result.summaryPath}")


if __name__ == "__main__":
    main()
