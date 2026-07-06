import json

from oetongsu_ml.progress_viewer import format_kst, load_progress, promotion_need_text, run_once


def test_format_kst_converts_utc_iso_time():
    assert format_kst("2026-07-06T07:28:55+00:00") == "2026-07-06 16:28:55 KST"
    assert format_kst("2026-07-06T07:28:55Z") == "2026-07-06 16:28:55 KST"


def test_promotion_need_text_calculates_remaining_score():
    arena = {
        "currentGames": 11,
        "totalGames": 40,
        "candidateWins": 6,
        "draws": 0,
    }
    assert promotion_need_text(arena, 0.55) == "남은 29판에서 16.0점 이상"


def test_load_progress_reports_missing_file(tmp_path):
    payload, error = load_progress(tmp_path / "missing.json")
    assert payload is None
    assert error is not None
    assert "진행률 파일을 찾을 수 없습니다" in error


def test_run_once_prints_missing_file_message(tmp_path, capsys):
    status = run_once(tmp_path / "missing.json")

    assert status == 1
    assert "진행률 파일을 찾을 수 없습니다" in capsys.readouterr().out


def test_run_once_renders_progress_snapshot(tmp_path, capsys):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "status": "running",
                "statusLabelKo": "진행 중",
                "phaseKey": "arena",
                "phaseLabelKo": "승격 대국",
                "messageKo": "승격 대국 진행 중입니다.",
                "overallPercent": 90.3,
                "phasePercent": 27.5,
                "startedAt": "2026-07-06T05:29:26+00:00",
                "updatedAt": "2026-07-06T07:28:55+00:00",
                "elapsedText": "01:59:28",
                "etaText": "00:12:54",
                "iteration": {"current": 2, "total": 2, "completed": 1},
                "models": {
                    "championVersion": "supervised_v0001",
                    "candidateVersion": "az_iter_000002",
                    "latestPromotedVersion": "supervised_v0001",
                    "promotionThreshold": 0.55,
                },
                "selfPlay": {"currentGames": 100, "totalGames": 100},
                "training": {"currentEpoch": 1, "totalEpochs": 1},
                "arena": {
                    "currentGames": 11,
                    "totalGames": 40,
                    "candidateWins": 6,
                    "championWins": 5,
                    "draws": 0,
                    "candidateScoreRate": 0.545,
                    "illegalMoves": 0,
                    "forfeits": 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert run_once(progress_path) == 0
    output = capsys.readouterr().out
    assert "외통수 AlphaZero 학습 진행 상황" in output
    assert "2026-07-06 16:28:55 KST" in output
    assert "남은 29판에서 16.0점 이상" in output
