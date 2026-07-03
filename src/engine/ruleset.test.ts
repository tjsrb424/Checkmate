import { describe, expect, it } from 'vitest';
import { getRuleset, resolveRuleset } from './ruleset';

describe('janggi rulesets', () => {
  it('defines the current compatibility ruleset', () => {
    expect(getRuleset('oetongsu-basic')).toMatchObject({
      repetitionPolicy: 'ban-third-position',
      bikjangPolicy: 'off',
      passPolicy: 'off',
      scoringPolicy: 'off',
      maxPlyPolicy: 'draw'
    });
  });

  it('defines the kakao-like ruleset with material adjudication', () => {
    expect(getRuleset('kakao-like')).toMatchObject({
      repetitionPolicy: 'ban-third-position',
      bikjangPolicy: 'score-adjudication',
      passPolicy: 'allow-when-not-in-check',
      scoringPolicy: 'material-score-with-han-deom',
      maxPlyPolicy: 'score-adjudication'
    });
  });

  it('defines the kja-like placeholder with the same initial enforcement policy', () => {
    expect(getRuleset('kja-like')).toMatchObject({
      repetitionPolicy: 'ban-third-position',
      bikjangPolicy: 'score-adjudication',
      scoringPolicy: 'material-score-with-han-deom'
    });
  });

  it('rejects unknown rulesets', () => {
    expect(() => resolveRuleset('missing' as never)).toThrow(/unknown janggi ruleset/);
  });
});
