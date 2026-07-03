import { describe, expect, it } from 'vitest';
import {
  OpeningBookRecord,
  alphaZeroSupervisedSampleToJsonLine,
  alphaZeroSupervisedSamplesToJsonl,
  exportAlphaZeroSupervisedSamplesFromOpeningRecords,
  exportPolicySamplesFromOpeningRecords,
  exportValueSamplesFromOpeningRecords,
  formationToChalimString,
  moveToPolicyIndex,
  parseOpeningRecordsCsv,
  policySampleToJsonLine,
  policySamplesToJsonl,
  valueSampleToJsonLine,
  valueSamplesToJsonl
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
    expect(first.position.positionHistory).toHaveLength(1);
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
    expect(samples[1].position.positionHistory).toHaveLength(2);
    expect(samples[1].position.metadata.outcomeForSide).toBe('loss');
  });
});

describe('value training export', () => {
  it('exports value samples from legal positions and skips unknown results', () => {
    const result = exportValueSamplesFromOpeningRecords(fixtureRecords(), { maxPly: 16 });

    expect(result.stats).toMatchObject({
      recordCount: 3,
      sourceGameCount: 2,
      sampleCount: 2,
      skippedGameCount: 1,
      illegalMoveCount: 1,
      parseFailureCount: 1,
      unknownResultSkipCount: 1
    });

    expect(result.samples[0]).toMatchObject({
      value: 1,
      result: 'cho',
      sideToMove: 'CHO',
      ply: 0,
      source: 'fixture-1'
    });
    expect(result.samples[1]).toMatchObject({
      value: -1,
      result: 'cho',
      sideToMove: 'HAN',
      ply: 1
    });
    expect(result.samples[0].position.board).toHaveLength(10);
    expect(result.samples[0].position.metadata.outcomeForSide).toBe('win');
  });

  it('maps draws to zero value', () => {
    const innerElephant = formationToChalimString('inner-elephant');
    const { samples } = exportValueSamplesFromOpeningRecords([
      {
        id: 'draw',
        choChalim: innerElephant,
        hanChalim: innerElephant,
        result: 'draw',
        first16: '1.06P05',
        moves16Ok: true
      }
    ]);

    expect(samples).toHaveLength(1);
    expect(samples[0].value).toBe(0);
  });

  it('serializes value JSONL rows', () => {
    const { samples } = exportValueSamplesFromOpeningRecords(fixtureRecords(), { maxPly: 1 });
    const line = valueSampleToJsonLine(samples[0]);
    const jsonl = valueSamplesToJsonl(samples);

    expect(JSON.parse(line).value).toBe(1);
    expect(jsonl.endsWith('\n')).toBe(true);
    expect(jsonl.trim().split('\n')).toHaveLength(1);
  });
});

describe('AlphaZero supervised export', () => {
  it('exports one-hot policy and value targets for legal opening positions', () => {
    const result = exportAlphaZeroSupervisedSamplesFromOpeningRecords(fixtureRecords(), { maxPly: 16 });

    expect(result.stats).toMatchObject({
      recordCount: 3,
      sourceGameCount: 2,
      sampleCount: 2,
      skippedGameCount: 1,
      illegalMoveCount: 1,
      parseFailureCount: 1,
      unknownResultSkipCount: 1
    });

    expect(result.samples[0]).toMatchObject({
      policy_target: [{ index: moveToPolicyIndex({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } }), prob: 1.0 }],
      value_target: 1,
      ply: 0,
      game_id: 'r1',
      to_play: 'CHO',
      final_winner: 'CHO'
    });
    expect(result.samples[0].position.positionHistory).toHaveLength(1);
    expect(result.samples[1]).toMatchObject({
      value_target: -1,
      to_play: 'HAN',
      final_winner: 'CHO'
    });
  });

  it('serializes AlphaZero supervised JSONL rows', () => {
    const { samples } = exportAlphaZeroSupervisedSamplesFromOpeningRecords(fixtureRecords(), { maxPly: 1 });
    const line = alphaZeroSupervisedSampleToJsonLine(samples[0]);
    const jsonl = alphaZeroSupervisedSamplesToJsonl(samples);

    expect(JSON.parse(line).policy_target[0].prob).toBe(1);
    expect(jsonl.endsWith('\n')).toBe(true);
    expect(jsonl.trim().split('\n')).toHaveLength(1);
  });
});
