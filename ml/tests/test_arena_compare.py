import json
from pathlib import Path

from oetongsu_ml import arena_compare


class DummyPlayer:
    def __init__(self, checkpoint, name="model"):
        self.checkpoint = checkpoint
        self.name = name


class DummyResult:
    def to_json(self):
        return {
            "games": 2,
            "candidateWins": 1,
            "championWins": 1,
            "draws": 0,
            "candidateScoreRate": 0.5,
            "championScoreRate": 0.5,
            "averagePlies": 4.0,
            "promoted": True,
            "illegalMoves": 0,
            "forfeits": 0,
            "gameSummaries": [
                {"candidateSide": "CHO", "championSide": "HAN", "winner": "CHO", "outcome": "score_adjudication", "plies": 4, "maxPlies": 4, "finalScore": {"margin": 2.0}},
                {"candidateSide": "HAN", "championSide": "CHO", "winner": "CHO", "outcome": "score_adjudication", "plies": 4, "maxPlies": 4, "finalScore": {"margin": 3.0}},
            ],
            "pairedSummary": {"pairs": 1, "sideDominatedPairs": 1},
        }


def test_arena_compare_writes_output_without_registry_mutation(tmp_path, monkeypatch):
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    registry = tmp_path / "registry.json"
    output = tmp_path / "compare.json"
    left.write_text("x")
    right.write_text("x")
    registry.write_text(json.dumps({"models": []}), encoding="utf-8")

    monkeypatch.setattr(arena_compare, "TorchModelPlayer", DummyPlayer)
    monkeypatch.setattr(arena_compare, "run_model_arena", lambda *args, **kwargs: DummyResult())

    status = arena_compare.main(
        [
            "--left",
            str(left),
            "--right",
            str(right),
            "--leftName",
            "left",
            "--rightName",
            "right",
            "--output",
            str(output),
            "--no-report",
        ]
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["leftWins"] == 1
    assert json.loads(registry.read_text(encoding="utf-8")) == {"models": []}


def test_arena_compare_missing_checkpoint_fails(tmp_path, capsys):
    status = arena_compare.main(["--left", str(tmp_path / "missing.pt"), "--right", str(tmp_path / "also_missing.pt"), "--no-report"])

    assert status == 1
    assert "missing left checkpoint" in capsys.readouterr().out
