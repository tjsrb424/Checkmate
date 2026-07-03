import json

from oetongsu_ml.autotrain import AutoTrainConfig, load_autotrain_state, run_autotrain


def config(tmp_path, iterations, resume=False):
    return AutoTrainConfig(
        iterations=iterations,
        games_per_iteration=1,
        simulations=1,
        arena_simulations=1,
        max_plies=2,
        train_epochs=1,
        batch_size=2,
        channels=4,
        promotion_games=1,
        promotion_threshold=1.1,
        seed=11,
        registry_path=str(tmp_path / "models" / "registry.json"),
        training_dir=str(tmp_path / "training"),
        selfplay_dir=str(tmp_path / "selfplay"),
        model_dir=str(tmp_path / "models"),
        arena_dir=str(tmp_path / "models" / "arena"),
        allow_random_champion=True,
        resume=resume,
    )


def test_autotrain_resume_starts_after_completed_iteration(tmp_path):
    first = run_autotrain(config(tmp_path, iterations=1))
    resumed = run_autotrain(config(tmp_path, iterations=2, resume=True))

    assert first.completedIterations == 1
    assert resumed.completedIterations == 2
    assert [row.iteration for row in resumed.iterations] == [2]
    assert (tmp_path / "selfplay" / "az_iter_000002.jsonl").exists()

    state = load_autotrain_state(tmp_path / "training" / "autotrain_state.json")
    assert state.completedIterations == 2
    assert state.latestCandidateVersion == "az_iter_000002"

    log_lines = (tmp_path / "training" / "autotrain_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["iteration"] for line in log_lines] == [1, 2]
