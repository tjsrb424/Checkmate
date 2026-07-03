import pytest

from oetongsu_ml.ruleset import get_ruleset, resolve_ruleset


def test_basic_ruleset_preserves_current_engine_policy():
    ruleset = get_ruleset("oetongsu-basic")

    assert ruleset.repetition_policy == "ban-third-position"
    assert ruleset.bikjang_policy == "off"
    assert ruleset.pass_policy == "off"
    assert ruleset.scoring_policy == "off"
    assert ruleset.max_ply_policy == "draw"


def test_kakao_like_ruleset_uses_material_adjudication():
    ruleset = get_ruleset("kakao-like")

    assert ruleset.repetition_policy == "ban-third-position"
    assert ruleset.bikjang_policy == "score-adjudication"
    assert ruleset.pass_policy == "allow-when-not-in-check"
    assert ruleset.scoring_policy == "material-score-with-han-deom"
    assert ruleset.max_ply_policy == "score-adjudication"


def test_kja_like_ruleset_is_documented_placeholder():
    ruleset = get_ruleset("kja-like")

    assert ruleset.repetition_policy == "ban-third-position"
    assert ruleset.bikjang_policy == "score-adjudication"
    assert ruleset.scoring_policy == "material-score-with-han-deom"


def test_unknown_ruleset_raises():
    with pytest.raises(ValueError, match="unknown janggi ruleset"):
        resolve_ruleset("missing")
