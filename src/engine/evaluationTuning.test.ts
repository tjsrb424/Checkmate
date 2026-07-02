import { describe, expect, it } from 'vitest';
import {
  applyMove,
  builtInOpeningBook,
  compareEvaluationBeforeAfter,
  createGameState,
  createInitialBoard,
  explainCandidateEvaluation,
  lookupOpeningMoves,
  moveKey,
  parseMoveLine,
  searchBestMove
} from './index';
import { blunderPositionFixtures } from './__fixtures__/blunderPositions';

function fixture(id: string) {
  const found = blunderPositionFixtures.find((item) => item.id === id);
  if (!found) throw new Error(`Missing fixture: ${id}`);
  return found;
}

function fixtureMove(id: string, text: string) {
  const move = parseMoveLine(text);
  if (!move) throw new Error(`Invalid fixture move ${id}: ${text}`);
  return move;
}

describe('evaluation tuning fixtures', () => {
  it('gives free-chariot fixture a large tactical safety penalty', () => {
    const item = fixture('synthetic-free-chariot');
    const explanation = compareEvaluationBeforeAfter(item.createState(), fixtureMove(item.id, item.forbiddenMoves[0]));

    expect(explanation.safety.riskScore).toBeGreaterThanOrEqual(2500);
    expect(explanation.safety.risks.map((risk) => risk.reason)).toContain('HANGS_CHARIOT');
    expect(explanation.deltaBreakdown.tacticalSafety).toBeLessThan(0);
  });

  it('gives free-cannon fixture a large tactical safety penalty', () => {
    const item = fixture('synthetic-free-cannon');
    const explanation = compareEvaluationBeforeAfter(item.createState(), fixtureMove(item.id, item.forbiddenMoves[0]));

    expect(explanation.safety.riskScore).toBeGreaterThanOrEqual(1600);
    expect(explanation.safety.risks.map((risk) => risk.reason)).toContain('HANGS_CANNON');
  });

  it('gives immediate mate fixture a decisive tactical safety penalty', () => {
    const item = fixture('synthetic-allows-immediate-mate');
    const explanation = compareEvaluationBeforeAfter(item.createState(), fixtureMove(item.id, item.forbiddenMoves[0]));

    expect(explanation.safety.allowsImmediateMate).toBe(true);
    expect(explanation.safety.riskLevel).toBe('losing');
    expect(explanation.afterBreakdown.tacticalSafety).toBeLessThan(-4000);
  });

  it('separates raw score, safety penalty, and final score for dangerous candidates', () => {
    const item = fixture('synthetic-bad-trade');
    const result = searchBestMove(item.createState(), { maxDepth: 1, timeMs: 2500 }, { maxCandidates: 80, useOpeningBook: false });
    const forbiddenKey = moveKey(fixtureMove(item.id, item.forbiddenMoves[0]));
    const expectedKey = moveKey(fixtureMove(item.id, item.expectedMoves[0]));
    const forbidden = result.candidates?.find((candidate) => moveKey(candidate.move) === forbiddenKey);
    const expected = result.candidates?.find((candidate) => moveKey(candidate.move) === expectedKey);

    expect(forbidden).toBeDefined();
    expect(expected).toBeDefined();
    expect(forbidden!.rawScore).toBeGreaterThan(forbidden!.finalScore);
    expect(forbidden!.safetyPenalty).toBeGreaterThan(3000);
    expect(expected!.finalScore - forbidden!.finalScore).toBeGreaterThan(1500);
  });

  it('explains candidate evaluation with tactical safety deltas', () => {
    const item = fixture('synthetic-bad-trade');
    const result = searchBestMove(item.createState(), { maxDepth: 1, timeMs: 2500 }, { maxCandidates: 80, useOpeningBook: false });
    const candidate = result.candidates?.find((entry) => moveKey(entry.move) === moveKey(fixtureMove(item.id, item.forbiddenMoves[0])));

    expect(candidate).toBeDefined();
    const explanation = explainCandidateEvaluation(item.createState(), candidate!);
    expect(explanation.summary).toContain('safety=');
    expect(explanation.safety.risks.map((risk) => risk.reason)).toContain('BAD_TRADE');
  });

  it('keeps search fallback stable after an opening book move', () => {
    const initial = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const bookMove = lookupOpeningMoves(builtInOpeningBook, initial, { minPlayCount: 1 })[0].move;
    const afterBook = applyMove(initial, bookMove, true);
    const result = searchBestMove(afterBook, { maxDepth: 1, timeMs: 2500 }, { maxCandidates: 8, useOpeningBook: false });

    expect(result.source).toBe('search');
    expect(result.candidates?.length).toBeGreaterThan(0);
    expect(result.selectedMoveSafety?.riskLevel).not.toBe('losing');
  });
});
