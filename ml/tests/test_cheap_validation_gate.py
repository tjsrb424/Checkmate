from oetongsu_ml.cheap_validation_gate import summarize_gate_result


def arena(score_rate, *, illegal=0, forfeits=0, side_dominated=0, pairs=2):
    games = 4
    return {
        "games": games,
        "candidateWins": int(score_rate * games),
        "championWins": games - int(score_rate * games),
        "draws": 0,
        "candidateScoreRate": score_rate,
        "averagePlies": 80,
        "illegalMoves": illegal,
        "forfeits": forfeits,
        "gameSummaries": [
            {"plies": 80, "outcome": "score_adjudication", "finalScore": {"margin": 2.0}},
            {"plies": 80, "outcome": "score_adjudication", "finalScore": {"margin": 2.0}},
        ],
        "pairedSummary": {"pairs": pairs, "sideDominatedPairs": side_dominated, "warnings": []},
    }


def summarize(payload):
    return summarize_gate_result(
        payload,
        candidate="candidate.pt",
        champion="champion.pt",
        simulations=8,
        max_plies=80,
        adjudication_draw_margin=1.5,
        min_score_rate=0.25,
    )


def test_cheap_validation_passes_viable_candidate_with_warnings():
    payload = summarize(arena(0.5))

    assert payload["status"] == "pass"
    assert "all games reached maxPlies" in payload["warnings"]


def test_cheap_validation_fails_zero_score_rate():
    payload = summarize(arena(0.0))

    assert payload["status"] == "fail"


def test_cheap_validation_fails_side_dominated_pairs():
    payload = summarize(arena(0.5, side_dominated=2, pairs=2))

    assert payload["status"] == "fail"
