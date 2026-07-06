from oetongsu_ml.arena_diagnostics import analyze_arena_payload, render_diagnostics


def cho_biased_fixture():
    summaries = []
    for game in range(1, 41):
        candidate_side = "CHO" if game % 2 == 1 else "HAN"
        summaries.append(
            {
                "game": game,
                "candidateSide": candidate_side,
                "championSide": "HAN" if candidate_side == "CHO" else "CHO",
                "winner": "CHO",
                "outcome": "score_adjudication",
                "plies": 100,
                "maxPlies": 100,
                "illegalMoves": 0,
                "forfeits": 0,
                "finalScore": {"cho": 42, "han": 40.5, "winner": "CHO", "margin": 1.5},
            }
        )
    return {
        "games": 40,
        "candidateWins": 20,
        "championWins": 20,
        "draws": 0,
        "candidateScoreRate": 0.5,
        "championScoreRate": 0.5,
        "averagePlies": 100,
        "promoted": False,
        "illegalMoves": 0,
        "forfeits": 0,
        "gameSummaries": summaries,
    }


def test_detects_cho_winner_bias():
    diagnostics = analyze_arena_payload(cho_biased_fixture(), "az_iter_000002_arena.json")

    assert diagnostics.winner_counts["CHO"] == 40
    assert diagnostics.winner_counts["HAN"] == 0
    assert diagnostics.candidate_cho_wins == 20
    assert diagnostics.candidate_han_wins == 0


def test_detects_score_adjudication_and_max_plies_rates():
    diagnostics = analyze_arena_payload(cho_biased_fixture())

    assert diagnostics.outcome_counts["score_adjudication"] == 40
    assert diagnostics.max_plies_reached == 40
    assert diagnostics.inferred_max_plies == 100


def test_generates_bias_warnings():
    diagnostics = analyze_arena_payload(cho_biased_fixture())

    assert any("점수 판정" in warning for warning in diagnostics.warnings)
    assert any("CHO" in warning for warning in diagnostics.warnings)
    assert any("모델 강도보다 진영/점수 판정 편향" in warning for warning in diagnostics.warnings)


def test_render_includes_expected_summary_lines():
    rendered = render_diagnostics(analyze_arena_payload(cho_biased_fixture(), "az_iter_000002_arena.json"))

    assert "파일: az_iter_000002_arena.json" in rendered
    assert "score_adjudication: 40 / 40, 100.0%" in rendered
    assert "CHO 승리: 40 / 40, 100.0%" in rendered
