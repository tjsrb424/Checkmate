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
