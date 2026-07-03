import json

from oetongsu_ml.autotrain import AutoTrainConfig, run_autotrain


def test_autotrain_uses_parallel_selfplay_workers(tmp_path):
    result = run_autotrain(
        AutoTrainConfig(
            iterations=1,
            games_per_iteration=2,
            simulations=1,
            arena_simulations=1,
            max_plies=2,
            train_epochs=1,
            batch_size=2,
            channels=4,
            promotion_games=1,
            promotion_threshold=1.1,
            seed=41,
            registry_path=str(tmp_path / "models" / "registry.json"),
            training_dir=str(tmp_path / "training"),
            selfplay_dir=str(tmp_path / "selfplay"),
            shard_dir=str(tmp_path / "selfplay" / "shards"),
            model_dir=str(tmp_path / "models"),
            arena_dir=str(tmp_path / "models" / "arena"),
            allow_random_champion=True,
            selfplay_workers=2,
            parallel_selfplay=True,
        )
    )

    assert result.completedIterations == 1
    iteration = result.iterations[0]
    assert iteration.sampleCount > 0
    assert (tmp_path / "selfplay" / "az_iter_000001.jsonl").exists()
    assert (tmp_path / "selfplay" / "az_iter_000001_summary.json").exists()
    assert (tmp_path / "models" / "checkpoints" / "az_iter_000001.pt").exists()

    summary = json.loads((tmp_path / "selfplay" / "az_iter_000001_summary.json").read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / "models" / "registry.json").read_text(encoding="utf-8"))
    assert summary["workers"] == 2
    assert summary["sample_count"] == iteration.sampleCount
    assert len(summary["shards"]) == 2
    assert iteration.metrics["selfPlay"]["workers"] == 2
    assert registry["models"][0]["version"] == "az_iter_000001"
