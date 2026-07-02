import { describe, expect, it } from 'vitest';
import { analyzePosition, moveKey, parseMoveLine, searchBestMove } from './index';
import { blunderPositionFixtures } from './__fixtures__/blunderPositions';

function parseMoveKeys(moves: string[]): string[] {
  return moves.flatMap((text) => {
    const move = parseMoveLine(text);
    return move ? [moveKey(move)] : [];
  });
}

describe('blunder fixture regression', () => {
  it('keeps all forbidden fixture moves out of the selected move', () => {
    for (const fixture of blunderPositionFixtures) {
      const state = fixture.createState();
      const result = searchBestMove(state, { maxDepth: 1, timeMs: 2500 }, { maxCandidates: 80, useOpeningBook: false });
      const selected = result.move ? moveKey(result.move) : '';

      expect(parseMoveKeys(fixture.forbiddenMoves), fixture.id).not.toContain(selected);
      expect(result.candidates?.[0]?.move, fixture.id).toEqual(result.move);
      expect(result.candidates?.[0]?.finalScore, fixture.id).toBe(result.score);
    }
  });

  it('keeps expected fixture moves in the top three candidates when specified', () => {
    for (const fixture of blunderPositionFixtures.filter((item) => item.expectedMoves.length > 0)) {
      const analysis = analyzePosition(fixture.createState(), {
        limits: { maxDepth: 1, timeMs: 2500 },
        maxCandidates: 80,
        searchOptions: { useOpeningBook: false }
      });
      const topThree = analysis.candidates.slice(0, 3).map((candidate) => moveKey(candidate.move));

      for (const expected of parseMoveKeys(fixture.expectedMoves)) {
        expect(topThree, fixture.id).toContain(expected);
      }
    }
  });

  it('marks losing blunder fixture moves as danger or losing', () => {
    for (const fixture of blunderPositionFixtures.filter((item) => item.tags.includes('losing-blunder'))) {
      const result = searchBestMove(fixture.createState(), { maxDepth: 1, timeMs: 2500 }, { maxCandidates: 80, useOpeningBook: false });
      const forbidden = new Set(parseMoveKeys(fixture.forbiddenMoves));
      const riskyCandidates = result.candidates?.filter((candidate) => forbidden.has(moveKey(candidate.move))) ?? [];

      expect(riskyCandidates.length, fixture.id).toBeGreaterThan(0);
      expect(riskyCandidates.every((candidate) => candidate.riskLevel === 'losing' || candidate.riskLevel === 'danger'), fixture.id).toBe(true);
    }
  });
});
