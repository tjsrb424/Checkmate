import json

from oetongsu_ml import ablation_evaluate


class DummyPlayer:
    def __init__(self, checkpoint, name="model"):
        self.checkpoint = checkpoint
        self.name = name


class DummyResult:
    def __init__(self, wins):
        self.wins = wins

    def to_json(self):
        return {
            "games": 2,
            "candidateWins": self.wins,
            "championWins": 2 - self.wins,
            "draws": 0,
            "candidateScoreRate": self.wins / 2,
            "championScoreRate": (2 - self.wins) / 2,
            "averagePlies": 4.0,
            "promoted": False,
            "illegalMoves": 0,
            "forfeits": 0,
            "gameSummaries": [{"finalScore": {"margin": 2.0}}],
            "pairedSummary": {"pairs": 1},
        }


def test_ablation_evaluate_writes_summary_for_multiple_candidates(tmp_path, monkeypatch):
    champion = tmp_path / "champion.pt"
    c1 = tmp_path / "c1.pt"
    c2 = tmp_path / "c2.pt"
    output = tmp_path / "evaluation_summary.json"
    for path in (champion, c1, c2):
        path.write_text("x")
    monkeypatch.setattr(ablation_evaluate, "TorchModelPlayer", DummyPlayer)
    calls = iter([DummyResult(0), DummyResult(1)])
    monkeypatch.setattr(ablation_evaluate, "run_model_arena", lambda *args, **kwargs: next(calls))

    status = ablation_evaluate.main(
        [
            "--champion",
            str(champion),
            "--candidates",
            str(c1),
            str(c2),
            "--output",
            str(output),
            "--no-report",
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 2
    assert payload["bestCandidate"]["candidateName"] == "c2"
