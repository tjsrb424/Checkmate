from oetongsu_ml.model_regression_diagnostics import main


def test_missing_checkpoint_returns_nonzero(tmp_path, capsys):
    status = main(
        [
            "--champion",
            str(tmp_path / "missing_champion.pt"),
            "--previous",
            str(tmp_path / "missing_previous.pt"),
            "--candidate",
            str(tmp_path / "missing_candidate.pt"),
            "--samples",
            str(tmp_path / "missing.jsonl"),
            "--no-report",
        ]
    )

    assert status == 1
    assert "missing champion checkpoint" in capsys.readouterr().out
