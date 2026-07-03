import { describe, expect, it } from 'vitest';
import {
  OpeningBookRecord,
  exportPolicySamplesFromOpeningRecords,
  formationToChalimString,
  moveToPolicyIndex,
  parseOpeningRecordsCsv,
  policySampleToJsonLine,
  policySamplesToJsonl
} from './index';

function fixtureRecords(): OpeningBookRecord[] {
  const innerElephant = formationToChalimString('inner-elephant');
  return [
    {
      id: 'r1',
      source: 'fixture-1',
      choChalim: innerElephant,
      hanChalim: innerElephant,
      result: 'cho',
      first16: '1.06P05 2.03P04',
      moves16Ok: true
    },
    {
      id: 'illegal',
      source: 'fixture-illegal',
      choChalim: innerElephant,
      hanChalim: innerElephant,
      result: 'han',
      first16: '1.88P87 bad-token',
      moves16Ok: true
    },
    {
      id: 'unknown',
      choChalim: innerElephant,
      hanChalim: innerElephant,
      result: 'unknown',
      first16: '1.06P05',
      moves16Ok: true
    }
  ];
}

describe('policy training export', () => {
  it('exports legal policy samples with Sprint 17 compatible positions', () => {
    const result = exportPolicySamplesFromOpeningRecords(fixtureRecords(), { maxPly: 16 });

    expect(result.stats).toMatchObject({
      recordCount: 3,
      sourceGameCount: 2,
      sampleCount: 2,
      skippedGameCount: 1,
      illegalMoveCount: 1,
      parseFailureCount: 1,
      unknownResultSkipCount: 1
    });

    const first = result.samples[0];
    expect(first.position.board).toHaveLength(10);
    expect(first.position.board[0]).toHaveLength(9);
    expect(first.position.turn).toBe('CHO');
    expect(first.position.history).toEqual([]);
    expect(first.move).toEqual({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } });
    expect(first.move_index).toBe(moveToPolicyIndex(first.move));
    expect(first.choFormation).toBe('inner-elephant');
    expect(first.hanFormation).toBe('inner-elephant');
    expect(first.position.metadata.outcomeForSide).toBe('win');
  });

  it('serializes JSONL rows', () => {
    const { samples } = exportPolicySamplesFromOpeningRecords(fixtureRecords(), { maxPly: 1 });
    const line = policySampleToJsonLine(samples[0]);
    const jsonl = policySamplesToJsonl(samples);

    expect(JSON.parse(line).move_index).toBe(samples[0].move_index);
    expect(jsonl.endsWith('\n')).toBe(true);
    expect(jsonl.trim().split('\n')).toHaveLength(1);
  });

  it('exports from parsed CSV fixtures', () => {
    const innerElephant = formationToChalimString('inner-elephant');
    const csv = [
      'source,group,game_index,cho_chalim,han_chalim,result,moves16_ok,first16',
      `sample,A,1,${innerElephant},${innerElephant},cho,true,"1.06P05 2.03P04"`
    ].join('\n');
    const records = parseOpeningRecordsCsv(csv);
    const { samples, stats } = exportPolicySamplesFromOpeningRecords(records);

    expect(stats.sampleCount).toBe(2);
    expect(samples[1].position.turn).toBe('HAN');
    expect(samples[1].position.history).toHaveLength(1);
    expect(samples[1].position.metadata.outcomeForSide).toBe('loss');
  });
});
