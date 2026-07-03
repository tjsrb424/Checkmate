from oetongsu_ml.model_arena import ModelArenaConfig, RandomModelPlayer, run_model_arena


def test_model_arena_result_score_rate_and_threshold():
    result = run_model_arena(
        RandomModelPlayer(name="candidate", seed=1),
        RandomModelPlayer(name="champion", seed=2),
        ModelArenaConfig(games=2, simulations=2, max_plies=4, temperature=0, promotion_threshold=0.0),
    )

    assert result.games == 2
    assert result.candidateWins + result.championWins + result.draws == 2
    assert result.candidateScoreRate + result.championScoreRate == 1.0
    assert result.averagePlies <= 4
    assert result.illegalMoves == 0
    assert result.forfeits == 0
    assert result.promoted is True


def test_model_arena_threshold_can_reject():
    result = run_model_arena(
        RandomModelPlayer(name="candidate", seed=3),
        RandomModelPlayer(name="champion", seed=4),
        ModelArenaConfig(games=2, simulations=2, max_plies=4, temperature=0, promotion_threshold=1.1),
    )

    assert result.promoted is False


def test_quick_model_arena_finishes():
    result = run_model_arena(
        RandomModelPlayer(name="candidate", seed=5),
        RandomModelPlayer(name="champion", seed=6),
        ModelArenaConfig(games=1, simulations=1, max_plies=2, temperature=0),
    )

    assert result.games == 1
    assert result.gameSummaries


def test_model_arena_can_score_adjudicate_max_plies():
    result = run_model_arena(
        RandomModelPlayer(name="candidate", seed=7),
        RandomModelPlayer(name="champion", seed=8),
        ModelArenaConfig(games=1, simulations=1, max_plies=0, temperature=0, ruleset_id="kakao-like"),
    )

    assert result.games == 1
    assert result.candidateWins == 0
    assert result.championWins == 1
    assert result.draws == 0
    assert result.gameSummaries[0]["outcome"] == "score_adjudication"
    assert result.gameSummaries[0]["winner"] == "HAN"
