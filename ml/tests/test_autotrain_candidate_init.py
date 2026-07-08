import json

import pytest
import torch

from oetongsu_ml.alphazero_model import AlphaZeroNet
from oetongsu_ml.autotrain import AutoTrainConfig, resolve_candidate_resume_checkpoint, run_autotrain_iteration
from oetongsu_ml.model_registry import promote_candidate, register_candidate


def save_checkpoint(path, channels=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = AlphaZeroNet(channels=channels)
    torch.save({"model_state": model.state_dict(), "channels": channels}, path)


def registry_with_promoted_and_rejected(tmp_path):
    champion = tmp_path / "models" / "checkpoints" / "supervised_v0001.pt"
    rejected = tmp_path / "models" / "checkpoints" / "az_iter_000002.pt"
    save_checkpoint(champion)
    save_checkpoint(rejected)
    registry = {"models": []}
    register_candidate(registry, "supervised_v0001", str(champion), parent_version=None)
    promote_candidate(registry, "supervised_v0001", {"bootstrap": True})
    rejected_entry = register_candidate(registry, "az_iter_000002", str(rejected), parent_version="supervised_v0001")
    rejected_entry["status"] = "rejected"
    return registry, champion, rejected


def test_rejected_candidate_does_not_become_resume_source(tmp_path):
    registry, champion, _rejected = registry_with_promoted_and_rejected(tmp_path)

    resolution = resolve_candidate_resume_checkpoint(AutoTrainConfig(), registry, latest_candidate_version="az_iter_000002")

    assert resolution.championVersion == "supervised_v0001"
    assert resolution.resumePath == str(champion)
    assert resolution.resumeCandidateFromChampion is True
    assert resolution.latestCandidateStatus == "rejected"


def test_resume_guard_rejects_latest_candidate_path_as_resume_source(tmp_path):
    registry, _champion, rejected = registry_with_promoted_and_rejected(tmp_path)

    with pytest.raises(RuntimeError, match="rejected/latest candidate"):
        resolve_candidate_resume_checkpoint(
            AutoTrainConfig(initial_champion=str(rejected)),
            registry,
            latest_candidate_version="az_iter_000002",
        )


def test_cheap_validation_fail_fast_skips_full_arena_and_rejects(tmp_path, monkeypatch):
    registry, champion, _rejected = registry_with_promoted_and_rejected(tmp_path)
    cfg = AutoTrainConfig(
        games_per_iteration=1,
        train_epochs=1,
        batch_size=2,
        channels=4,
        registry_path=str(tmp_path / "models" / "registry.json"),
        training_dir=str(tmp_path / "training"),
        selfplay_dir=str(tmp_path / "selfplay"),
        model_dir=str(tmp_path / "models"),
        arena_dir=str(tmp_path / "models" / "arena"),
        cheap_validation_before_arena=True,
        cheap_validation_fail_fast=True,
    )

    def fake_selfplay(*args, **kwargs):
        path = tmp_path / "selfplay" / "az_iter_000003.jsonl"
        summary = tmp_path / "selfplay" / "az_iter_000003_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        summary.write_text("{}", encoding="utf-8")
        return path, summary, 1, {"sampleCount": 1}

    def fake_train(**kwargs):
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("checkpoint", encoding="utf-8")
        metrics = output.with_name(f"{output.stem}_metrics.json")
        metrics.write_text(json.dumps({"history": [{"total_loss": 1.0}]}), encoding="utf-8")
        return {"history": [{"total_loss": 1.0}], "training_metadata": kwargs.get("training_metadata")}

    def fail_full_arena(*args, **kwargs):
        raise AssertionError("full arena should be skipped")

    monkeypatch.setattr("oetongsu_ml.autotrain.generate_self_play", fake_selfplay)
    monkeypatch.setattr("oetongsu_ml.autotrain.train_alphazero", fake_train)
    monkeypatch.setattr(
        "oetongsu_ml.autotrain.run_cheap_validation_if_enabled",
        lambda *args, **kwargs: {
            "enabled": True,
            "status": "fail",
            "candidateScoreRate": 0.0,
            "games": 4,
            "candidateWins": 0,
            "championWins": 4,
            "draws": 0,
            "averagePlies": 80,
            "illegalMoves": 0,
            "forfeits": 0,
            "warnings": ["candidateScoreRate is zero"],
            "rawArenaResult": {"gameSummaries": []},
        },
    )
    monkeypatch.setattr("oetongsu_ml.autotrain.run_candidate_arena", fail_full_arena)

    result = run_autotrain_iteration(cfg, registry, iteration=3, latest_candidate_version="az_iter_000002", run_id="run-test")

    assert result.status == "rejected"
    assert result.promoted is False
    assert result.cheapValidation["status"] == "fail"
    assert result.championVersion == "supervised_v0001"
    assert result.metrics["arena"]["cheapValidationFailFast"] is True
