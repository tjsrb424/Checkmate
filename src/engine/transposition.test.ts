import { describe, expect, it } from 'vitest';
import { TranspositionEntry, TranspositionTable } from './index';

function entry(overrides: Partial<TranspositionEntry> = {}): TranspositionEntry {
  return {
    key: 'abc',
    depth: 1,
    score: 10,
    flag: 'EXACT',
    ...overrides
  };
}

describe('transposition table', () => {
  it('stores and retrieves entries', () => {
    const table = new TranspositionTable();
    table.set(entry());

    expect(table.get('abc')?.score).toBe(10);
    expect(table.size()).toBe(1);
  });

  it('replaces shallow entries with deeper entries', () => {
    const table = new TranspositionTable();
    table.set(entry({ depth: 1, score: 10 }));
    table.set(entry({ depth: 3, score: 30 }));

    expect(table.get('abc')?.score).toBe(30);
  });

  it('does not replace deeper entries with shallow entries', () => {
    const table = new TranspositionTable();
    table.set(entry({ depth: 3, score: 30 }));
    table.set(entry({ depth: 1, score: 10 }));

    expect(table.get('abc')?.score).toBe(30);
  });

  it('prefers exact entries at the same depth', () => {
    const table = new TranspositionTable();
    table.set(entry({ depth: 2, score: 20, flag: 'LOWERBOUND' }));
    table.set(entry({ depth: 2, score: 25, flag: 'EXACT' }));

    expect(table.get('abc')?.score).toBe(25);
  });

  it('tracks stats', () => {
    const table = new TranspositionTable();
    table.get('missing');
    table.set(entry({ depth: 1 }));
    table.set(entry({ depth: 2 }));
    table.get('abc');

    expect(table.getStats()).toEqual({
      hits: 1,
      misses: 1,
      stores: 2,
      overwrites: 1
    });
  });

  it('clears entries and stats', () => {
    const table = new TranspositionTable();
    table.set(entry());
    table.get('abc');
    table.clear();

    expect(table.size()).toBe(0);
    expect(table.getStats()).toEqual({
      hits: 0,
      misses: 0,
      stores: 0,
      overwrites: 0
    });
  });
});
