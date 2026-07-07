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
        "pairedSummary": {
            "pairs": 20,
            "sideDominatedPairs": 20,
            "candidateDominatedPairs": 0,
            "championDominatedPairs": 0,
            "splitPairs": 20,
            "drawPairs": 0,
            "warnings": ["side dominated"],
        },
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

    assert any("score_adjudication" in warning for warning in diagnostics.warnings)
    assert any("CHO" in warning for warning in diagnostics.warnings)
    assert any("side/scoring policy" in warning for warning in diagnostics.warnings)


def test_render_includes_expected_summary_lines():
    rendered = render_diagnostics(analyze_arena_payload(cho_biased_fixture(), "az_iter_000002_arena.json"))

    assert "file: az_iter_000002_arena.json" in rendered
    assert "- score_adjudication: 40 / 40, 100.0%" in rendered
    assert "- CHO: 40 / 40, 100.0%" in rendered
    assert "- sideDominatedPairs: 20" in rendered


def test_margin_summary_counts_draw_margin_buckets():
    diagnostics = analyze_arena_payload(cho_biased_fixture(), draw_margin=1.5)

    assert diagnostics.margin_summary.count == 40
    assert diagnostics.margin_summary.within_draw_margin == 40
    assert diagnostics.margin_summary.outside_draw_margin == 0
