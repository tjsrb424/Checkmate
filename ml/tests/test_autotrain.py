import json

from oetongsu_ml.autotrain import AutoTrainConfig, run_autotrain


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


def test_autotrain_requires_champion_without_random_fallback(tmp_path):
    config = quick_config(tmp_path, allow_random_champion=False)

    try:
        run_autotrain(config)
    except RuntimeError as error:
        assert "no promoted champion" in str(error)
    else:
        raise AssertionError("expected autotrain to require a champion")
