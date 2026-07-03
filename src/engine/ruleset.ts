export type RulesetId = 'oetongsu-basic' | 'kakao-like' | 'kja-like';

export type RepetitionPolicy = 'off' | 'ban-third-position' | 'draw-third-position' | 'adjudicate-fault';
export type BikjangPolicy = 'off' | 'draw-claim' | 'must-break' | 'score-adjudication';
export type PassPolicy = 'off' | 'allow-when-not-in-check' | 'allow-only-no-legal-move';
export type ScoringPolicy = 'off' | 'material-score-with-han-deom';
export type MaxPlyPolicy = 'draw' | 'score-adjudication';

export interface JanggiRuleset {
  id: RulesetId;
  label: string;
  repetitionPolicy: RepetitionPolicy;
  bikjangPolicy: BikjangPolicy;
  passPolicy: PassPolicy;
  scoringPolicy: ScoringPolicy;
  maxPlyPolicy: MaxPlyPolicy;
}

export const rulesets = {
  'oetongsu-basic': {
    id: 'oetongsu-basic',
    label: 'Oetongsu Basic',
    repetitionPolicy: 'ban-third-position',
    bikjangPolicy: 'off',
    passPolicy: 'off',
    scoringPolicy: 'off',
    maxPlyPolicy: 'draw'
  },
  'kakao-like': {
    id: 'kakao-like',
    label: 'Kakao-like Janggi',
    repetitionPolicy: 'ban-third-position',
    bikjangPolicy: 'score-adjudication',
    passPolicy: 'allow-when-not-in-check',
    scoringPolicy: 'material-score-with-han-deom',
    maxPlyPolicy: 'score-adjudication'
  },
  'kja-like': {
    id: 'kja-like',
    label: 'KJA-like Janggi',
    repetitionPolicy: 'ban-third-position',
    bikjangPolicy: 'score-adjudication',
    passPolicy: 'allow-when-not-in-check',
    scoringPolicy: 'material-score-with-han-deom',
    maxPlyPolicy: 'score-adjudication'
  }
} satisfies Record<RulesetId, JanggiRuleset>;

export function getRuleset(id: RulesetId = 'oetongsu-basic'): JanggiRuleset {
  const ruleset = rulesets[id];
  if (!ruleset) throw new Error(`unknown janggi ruleset: ${id}`);
  return ruleset;
}

export function resolveRuleset(ruleset?: RulesetId | JanggiRuleset): JanggiRuleset {
  if (!ruleset) return getRuleset('oetongsu-basic');
  return typeof ruleset === 'string' ? getRuleset(ruleset) : ruleset;
}
