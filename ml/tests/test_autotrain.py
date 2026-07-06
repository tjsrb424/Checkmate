import json

import torch

from oetongsu_ml.alphazero_model import AlphaZeroNet
from oetongsu_ml.autotrain import AutoTrainConfig, run_autotrain
from oetongsu_ml.model_registry import load_registry, promote_candidate, register_candidate, save_registry


def quick_config(tmp_path, **overrides):
    base = {
        "iterations": 1,
        "games_per_iteration": 1,
        "simulations": 1,
        "arena_simulations": 1,
        "max_plies": 2,
        "train_epochs": 1,
        "batch_size": 2,
        "channels": 4,
        "promotion_games": 1,
        "promotion_threshold": 1.1,
        "seed": 7,
        "registry_path": str(tmp_path / "models" / "registry.json"),
        "training_dir": str(tmp_path / "training"),
        "selfplay_dir": str(tmp_path / "selfplay"),
        "model_dir": str(tmp_path / "models"),
        "arena_dir": str(tmp_path / "models" / "arena"),
        "allow_random_champion": True,
    }
    base.update(overrides)
    return AutoTrainConfig(**base)


def test_quick_autotrain_completes_one_iteration(tmp_path):
    result = run_autotrain(quick_config(tmp_path))

    assert result.completedIterations == 1
    assert len(result.iterations) == 1
    iteration = result.iterations[0]
    assert iteration.sampleCount > 0
    assert iteration.status in {"promoted", "rejected"}
    assert iteration.metrics["arena"]["illegalMoves"] == 0
    assert iteration.metrics["arena"]["forfeits"] == 0

    assert (tmp_path / "selfplay" / "az_iter_000001.jsonl").exists()
    assert (tmp_path / "models" / "checkpoints" / "az_iter_000001.pt").exists()
    assert (tmp_path / "training" / "autotrain_log.jsonl").exists()
    assert (tmp_path / "training" / "autotrain_summary.json").exists()
    assert (tmp_path / "training" / "autotrain_state.json").exists()

    registry = json.loads((tmp_path / "models" / "registry.json").read_text(encoding="utf-8"))
    entry = registry["models"][0]
    assert entry["version"] == "az_iter_000001"
    assert entry["status"] in {"promoted", "rejected"}
    assert entry["arenaResults"][0]["illegalMoves"] == 0

    progress = json.loads((tmp_path / "training" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["overallPercent"] == 100
    assert progress["phaseLabelKo"]
    assert progress["messageKo"]
    assert (tmp_path / "training" / "progress_events.jsonl").exists()


def test_autotrain_requires_champion_without_random_fallback(tmp_path):
    config = quick_config(tmp_path, allow_random_champion=False)

    try:
        run_autotrain(config)
    except RuntimeError as error:
        assert "no promoted champion" in str(error)
    else:
        raise AssertionError("expected autotrain to require a champion")


def test_autotrain_uses_promoted_bootstrap_champion_without_random_fallback(tmp_path):
    champion_path = tmp_path / "models" / "checkpoints" / "supervised_v0001.pt"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    model = AlphaZeroNet(channels=4)
    torch.save({"model_state": model.state_dict(), "channels": 4}, champion_path)
    registry_path = tmp_path / "models" / "registry.json"
    registry = load_registry(registry_path)
    register_candidate(
        registry,
        version="supervised_v0001",
        path=str(champion_path),
        metadata_path=None,
        metrics={"bootstrap": True, "source": "supervised"},
    )
    promote_candidate(registry, "supervised_v0001", {"bootstrap": True})
    save_registry(registry_path, registry)

    result = run_autotrain(
        quick_config(
            tmp_path,
            allow_random_champion=False,
            registry_path=str(registry_path),
            promotion_threshold=1.1,
        )
    )

    assert result.completedIterations == 1
    assert result.iterations[0].championVersion == "supervised_v0001"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = next(entry for entry in registry["models"] if entry["version"] == "az_iter_000001")
    assert candidate["parentVersion"] == "supervised_v0001"


def test_two_iteration_progress_does_not_complete_after_first_iteration(tmp_path):
    result = run_autotrain(quick_config(tmp_path, iterations=2))

    assert result.completedIterations == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "training" / "progress_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    package_events = [event for event in events if event["phaseKey"] == "package"]
    assert len(package_events) == 2
    assert package_events[0]["overallPercent"] == 50
    assert package_events[1]["overallPercent"] < 100

    first_package_index = events.index(package_events[0])
    later_selfplay = next(event for event in events[first_package_index + 1 :] if event["phaseKey"] == "selfplay")
    assert later_selfplay["overallPercent"] >= 50
    assert later_selfplay["overallPercent"] < 100

    terminal_events = [event for event in events if event["status"] == "completed"]
    assert terminal_events[-1]["overallPercent"] == 100
    for event in events[: events.index(terminal_events[-1])]:
        assert event["overallPercent"] < 100


def test_resume_progress_starts_from_completed_iterations(tmp_path):
    training_dir = tmp_path / "training"
    training_dir.mkdir(parents=True)
    state = {
        "runId": "autotrain-resume",
        "startedAt": "2026-07-06T00:00:00+00:00",
        "updatedAt": "2026-07-06T00:00:00+00:00",
        "currentIteration": 1,
        "completedIterations": 1,
        "latestChampionVersion": None,
        "latestCandidateVersion": "az_iter_000001",
        "lastSelfPlayPath": None,
        "lastCandidatePath": None,
        "lastArenaResultPath": None,
        "status": "completed",
    }
    (training_dir / "autotrain_state.json").write_text(json.dumps(state), encoding="utf-8")

    result = run_autotrain(quick_config(tmp_path, iterations=2, resume=True))

    assert result.completedIterations == 2
    events = [
        json.loads(line)
        for line in (training_dir / "progress_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_selfplay = next(event for event in events if event["phaseKey"] == "selfplay")
    assert first_selfplay["overallPercent"] == 50
    assert events[-1]["status"] == "completed"
    assert events[-1]["overallPercent"] == 100
