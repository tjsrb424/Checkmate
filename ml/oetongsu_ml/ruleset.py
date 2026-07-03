from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RulesetId = Literal["oetongsu-basic", "kakao-like", "kja-like"]
RepetitionPolicy = Literal["off", "ban-third-position", "draw-third-position", "adjudicate-fault"]
BikjangPolicy = Literal["off", "draw-claim", "must-break", "score-adjudication"]
PassPolicy = Literal["off", "allow-when-not-in-check", "allow-only-no-legal-move"]
ScoringPolicy = Literal["off", "material-score-with-han-deom"]
MaxPlyPolicy = Literal["draw", "score-adjudication"]


@dataclass(frozen=True)
class JanggiRuleset:
    id: RulesetId
    label: str
    repetition_policy: RepetitionPolicy
    bikjang_policy: BikjangPolicy
    pass_policy: PassPolicy
    scoring_policy: ScoringPolicy
    max_ply_policy: MaxPlyPolicy


RULESETS: dict[RulesetId, JanggiRuleset] = {
    "oetongsu-basic": JanggiRuleset(
        id="oetongsu-basic",
        label="Oetongsu Basic",
        repetition_policy="ban-third-position",
        bikjang_policy="off",
        pass_policy="off",
        scoring_policy="off",
        max_ply_policy="draw",
    ),
    "kakao-like": JanggiRuleset(
        id="kakao-like",
        label="Kakao-like Janggi",
        repetition_policy="ban-third-position",
        bikjang_policy="score-adjudication",
        pass_policy="allow-when-not-in-check",
        scoring_policy="material-score-with-han-deom",
        max_ply_policy="score-adjudication",
    ),
    "kja-like": JanggiRuleset(
        id="kja-like",
        label="KJA-like Janggi",
        repetition_policy="ban-third-position",
        bikjang_policy="score-adjudication",
        pass_policy="allow-when-not-in-check",
        scoring_policy="material-score-with-han-deom",
        max_ply_policy="score-adjudication",
    ),
}


def get_ruleset(ruleset_id: RulesetId = "oetongsu-basic") -> JanggiRuleset:
    try:
        return RULESETS[ruleset_id]
    except KeyError as error:
        raise ValueError(f"unknown janggi ruleset: {ruleset_id}") from error


def resolve_ruleset(ruleset: RulesetId | JanggiRuleset | None = None) -> JanggiRuleset:
    if ruleset is None:
        return get_ruleset("oetongsu-basic")
    if isinstance(ruleset, JanggiRuleset):
        return ruleset
    return get_ruleset(ruleset)
