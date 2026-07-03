from oetongsu_ml.model_registry import (
    get_latest_promoted,
    load_registry,
    promote_candidate,
    register_candidate,
    reject_candidate,
    save_registry,
)


def test_registry_create_register_promote_reject_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    registry = load_registry(path)
    first = register_candidate(registry, "az_v0001", "az_v0001.pt", metadata_path="az_v0001.json")
    second = register_candidate(registry, "az_v0002", "az_v0002.pt", parent_version="az_v0001")

    assert first["status"] == "candidate"
    assert second["parentVersion"] == "az_v0001"

    promote_candidate(registry, "az_v0001", {"candidateScoreRate": 0.75})
    reject_candidate(registry, "az_v0002", {"candidateScoreRate": 0.25})
    save_registry(path, registry)
    restored = load_registry(path)

    assert len(restored["models"]) == 2
    assert get_latest_promoted(restored)["version"] == "az_v0001"
    assert restored["models"][0]["arenaResults"][0]["candidateScoreRate"] == 0.75
    assert restored["models"][1]["status"] == "rejected"


def test_register_candidate_is_idempotent():
    registry = load_registry("missing-registry-for-unit-test.json")

    first = register_candidate(registry, "candidate", "model.pt")
    second = register_candidate(registry, "candidate", "other.pt")

    assert first is second
    assert len(registry["models"]) == 1
