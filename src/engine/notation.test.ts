import { describe, expect, it } from 'vitest';
import { formatMove, formatMoveWithPiece, formatPlyNumber, formatPosition } from './notation';
import { Move } from './types';

describe('move notation', () => {
  it('formats board positions', () => {
    expect(formatPosition({ x: 0, y: 6 })).toBe('0,6');
  });

  it('formats normal moves with a piece label', () => {
    expect(
      formatMoveWithPiece(
        { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } },
        { side: 'CHO', kind: 'CHARIOT' }
      )
    ).toBe('차 0,6→0,5');
  });

  it('formats captures with a capture marker', () => {
    const move: Move = {
      from: { x: 1, y: 7 },
      to: { x: 1, y: 3 },
      captured: { side: 'HAN', kind: 'HORSE' }
    };

    expect(formatMoveWithPiece(move, { side: 'CHO', kind: 'CANNON' })).toBe('포 1,7×1,3');
  });

  it('distinguishes cho and han soldiers', () => {
    const move = { from: { x: 4, y: 6 }, to: { x: 4, y: 5 } };

    expect(formatMoveWithPiece(move, { side: 'CHO', kind: 'SOLDIER' })).toContain('졸');
    expect(formatMoveWithPiece(move, { side: 'HAN', kind: 'SOLDIER' })).toContain('병');
  });

  it('falls back when the moving piece is unknown', () => {
    expect(formatMove({ from: { x: 2, y: 2 }, to: { x: 2, y: 3 } })).toBe('? 2,2→2,3');
  });

  it('formats ply numbers by side', () => {
    expect(formatPlyNumber(0)).toBe('1.');
    expect(formatPlyNumber(1)).toBe('1...');
    expect(formatPlyNumber(2)).toBe('2.');
  });
});
