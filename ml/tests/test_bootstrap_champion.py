import json

import pytest

from oetongsu_ml.bootstrap_champion import bootstrap_champion
from oetongsu_ml.dataset import write_jsonl
from oetongsu_ml.model_registry import get_latest_promoted, load_registry


def empty_board():
    return [[None for _ in range(9)] for _ in range(10)]


def row(index, value, turn="CHO"):
    board = empty_board()
    board[9][4] = {"side": "CHO", "kind": "GENERAL"}
    board[0][4] = {"side": "HAN", "kind": "GENERAL"}
    return {
        "position": {"board": board, "turn": turn, "history": [], "positionHistory": ["key"], "winner": None, "metadata": {}},
        "policy_target": [{"index": index, "prob": 1.0}],
        "value_target": value,
        "move": {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}},
        "ply": 0,
        "game_id": f"unit-{index}",
        "to_play": turn,
        "final_winner": "CHO" if value > 0 else "HAN" if value < 0 else None,
    }


def test_bootstrap_champion_trains_and_promotes_registry_entry(tmp_path):
    data = tmp_path / "az_supervised.jsonl"
    output = tmp_path / "models" / "checkpoints" / "supervised_v0001.pt"
    registry_path = tmp_path / "models" / "registry.json"
    write_jsonl(data, [row(10, 1.0), row(20, -1.0), row(30, 0.0), row(40, 1.0)])

    result = bootstrap_champion(
        data=data,
        output=output,
        registry_path=registry_path,
        version="supervised_v0001",
        epochs=1,
        batch_size=2,
        channels=4,
    )

    assert output.exists()
    assert output.with_name("supervised_v0001_metrics.json").exists()
    assert result["latestPromotedVersion"] == "supervised_v0001"
    registry = load_registry(registry_path)
    latest = get_latest_promoted(registry)
    assert latest["version"] == "supervised_v0001"
    assert latest["status"] == "promoted"
    assert latest["metrics"]["bootstrap"] is True
    assert latest["metrics"]["source"] == "supervised"
    assert latest["arenaResults"][0]["promotedWithoutArena"] is True


def test_bootstrap_champion_requires_overwrite_for_existing_version(tmp_path):
    data = tmp_path / "az_supervised.jsonl"
    output = tmp_path / "models" / "checkpoints" / "supervised_v0001.pt"
    registry_path = tmp_path / "models" / "registry.json"
    write_jsonl(data, [row(10, 1.0), row(20, -1.0)])

    bootstrap_champion(data=data, output=output, registry_path=registry_path, version="supervised_v0001", epochs=1, batch_size=2, channels=4)

    with pytest.raises(ValueError, match="model version already exists"):
        bootstrap_champion(data=data, output=output, registry_path=registry_path, version="supervised_v0001", epochs=1, batch_size=2, channels=4)

    bootstrap_champion(
        data=data,
        output=output,
        registry_path=registry_path,
        version="supervised_v0001",
        epochs=1,
        batch_size=2,
        channels=4,
        overwrite=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [entry["version"] for entry in registry["models"]] == ["supervised_v0001"]
