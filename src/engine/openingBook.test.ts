import { describe, expect, it } from 'vitest';
import {
  Board,
  Formation,
  OpeningBookRecord,
  applyMove,
  buildOpeningBookFromRecords,
  chalimStringToFormation,
  chooseOpeningBookMove,
  createGameState,
  createInitialBoard,
  formationToChalimString,
  isLegalMove,
  lookupOpeningMoves,
  openingBookFromJson,
  openingBookToJson,
  parseFirst16Moves,
  parseOpeningMoveToken,
  parseOpeningRecordsCsv,
  resultForSide,
  summarizeOpeningBook,
  getTopOpeningMoves
} from './index';
import { builtInOpeningBook } from './openingBookData';

function fixtureRecords(): OpeningBookRecord[] {
  return [
    {
      id: 'r1',
      source: 'fixture-1',
      choChalim: '마상상마',
      hanChalim: '마상상마',
      result: 'cho',
      first16: '1.06졸05 2.03병04',
      moves16Ok: true
    },
    {
      id: 'r2',
      source: 'fixture-2',
      choChalim: '마상상마',
      hanChalim: '마상상마',
      result: 'cho',
      first16: '1.06졸05 2.03병04',
      moves16Ok: true
    },
    {
      id: 'r3',
      source: 'fixture-3',
      choChalim: '마상상마',
      hanChalim: '마상상마',
      result: 'draw',
      first16: '1.06졸05 2.03병04',
      moves16Ok: true
    },
    {
      id: 'illegal',
      choChalim: '마상상마',
      hanChalim: '마상상마',
      result: 'cho',
      first16: '1.88졸87 bad-token',
      moves16Ok: true
    }
  ];
}

describe('opening book move parsing', () => {
  it('parses first16 move tokens with suffixes', () => {
    expect(parseOpeningMoveToken('1.79졸78')).toEqual({
      plyNumber: 1,
      from: { x: 7, y: 9 },
      to: { x: 7, y: 8 },
      pieceLabel: '졸',
      suffix: ''
    });
    expect(parseOpeningMoveToken('10.45병46장군')).toEqual({
      plyNumber: 10,
      from: { x: 4, y: 5 },
      to: { x: 4, y: 6 },
      pieceLabel: '병',
      suffix: '장군'
    });
    expect(parseOpeningMoveToken('11.11차81차')).toEqual({
      plyNumber: 11,
      from: { x: 1, y: 1 },
      to: { x: 8, y: 1 },
      pieceLabel: '차',
      suffix: '차'
    });
    expect(parseOpeningMoveToken('bad')).toBeNull();
  });

  it('parses full first16 strings and skips invalid tokens', () => {
    const moves = parseFirst16Moves('1.06졸05 bad 2.03병04 3.05졸04');

    expect(moves).toHaveLength(3);
    expect(moves.map((move) => move.plyNumber)).toEqual([1, 2, 3]);
  });
});

describe('opening chalim mapping', () => {
  it('maps all four chalim strings to engine formations', () => {
    const formations: Formation[] = ['left-elephant', 'right-elephant', 'inner-elephant', 'outer-elephant'];

    for (const formation of formations) {
      expect(chalimStringToFormation(formationToChalimString(formation))).toBe(formation);
    }
  });

  it('matches createInitialBoard home row placement', () => {
    const formations: Formation[] = ['left-elephant', 'right-elephant', 'inner-elephant', 'outer-elephant'];

    for (const formation of formations) {
      const board = createInitialBoard(formation, formation);
      expect(homePattern(board, 'CHO')).toBe(formationToChalimString(formation));
      expect(homePattern(board, 'HAN')).toBe(formationToChalimString(formation));
    }
  });
});

describe('opening records CSV parsing', () => {
  it('parses header based CSV records and quoted commas', () => {
    const csv = [
      'source,group,game_index,cho_chalim,han_chalim,result,moves16_ok,first16',
      '"book,one",A,7,마상상마,상마마상,cho,True,"1.06졸05 2.03병04"',
      'book-two,B,8,상마상마,마상마상,draw,false,"1.06졸05"'
    ].join('\n');
    const records = parseOpeningRecordsCsv(csv);

    expect(records).toHaveLength(2);
    expect(records[0]).toMatchObject({
      id: '7',
      source: 'book,one',
      choChalim: '마상상마',
      hanChalim: '상마마상',
      result: 'cho',
      moves16Ok: true,
      first16: '1.06졸05 2.03병04'
    });
    expect(records[1].moves16Ok).toBe(false);
  });

  it('parses CSV alias columns and broader result values', () => {
    const csv = [
      'source,choChalim,han_formation,winner,opening_ok,moves16',
      'alias,마상상마,마상상마,"한 승",yes,"1.06졸05 2.03병04"'
    ].join('\n');
    const records = parseOpeningRecordsCsv(csv);

    expect(records[0]).toMatchObject({
      source: 'alias',
      choChalim: '마상상마',
      hanChalim: '마상상마',
      result: 'han',
      moves16Ok: true,
      first16: '1.06졸05 2.03병04'
    });
  });
});

describe('opening book build and lookup', () => {
  it('builds a book from fixture records and tracks stats', () => {
    const book = buildOpeningBookFromRecords(fixtureRecords(), { minPlayCount: 2 });

    expect(book.positionCount).toBeGreaterThan(0);
    expect(book.moveCount).toBeGreaterThan(0);
    expect(book.sourceGameCount).toBe(4);
    expect(book.illegalMoveCount).toBe(1);
    expect(book.parseFailureCount).toBe(1);
  });

  it('looks up legal candidate moves from the initial position', () => {
    const book = buildOpeningBookFromRecords(fixtureRecords(), { minPlayCount: 2 });
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const moves = lookupOpeningMoves(book, state, {
      choFormation: 'inner-elephant',
      hanFormation: 'inner-elephant',
      minPlayCount: 2
    });

    expect(moves).toHaveLength(1);
    expect(moves[0].playCount).toBe(3);
    expect(isLegalMove(state, moves[0].move)).toBe(true);
    expect(chooseOpeningBookMove(book, state, { minPlayCount: 2 })).toEqual(moves[0].move);
    expect(lookupOpeningMoves(book, state, { minPlayCount: 4 })).toHaveLength(0);
  });

  it('aggregates results from the side that played the move', () => {
    const book = buildOpeningBookFromRecords(fixtureRecords(), { minPlayCount: 1 });
    const initial = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const choMove = lookupOpeningMoves(book, initial, { minPlayCount: 1 })[0];
    const afterCho = applyMove(initial, choMove.move, true);
    const hanMove = lookupOpeningMoves(book, afterCho, { minPlayCount: 1 })[0];

    expect(choMove.winCount).toBe(2);
    expect(choMove.drawCount).toBe(1);
    expect(choMove.scoreRate).toBeCloseTo(2.5 / 3);
    expect(hanMove.lossCount).toBe(2);
    expect(hanMove.drawCount).toBe(1);
    expect(hanMove.scoreRate).toBeCloseTo(0.5 / 3);
    expect(resultForSide('cho', 'CHO')).toBe('win');
    expect(resultForSide('cho', 'HAN')).toBe('loss');
    expect(resultForSide('draw', 'HAN')).toBe('draw');
  });

  it('round trips through JSON', () => {
    const book = buildOpeningBookFromRecords(fixtureRecords(), { minPlayCount: 2 });
    const restored = openingBookFromJson(openingBookToJson(book));

    expect(restored.positionCount).toBe(book.positionCount);
    expect(restored.moveCount).toBe(book.moveCount);
    expect(Object.keys(restored.positions)).toEqual(Object.keys(book.positions));
  });

  it('summarizes top positions and moves', () => {
    const book = buildOpeningBookFromRecords(fixtureRecords(), { minPlayCount: 2 });
    const summary = summarizeOpeningBook(book, { top: 2 });
    const topMoves = getTopOpeningMoves(book, 2);

    expect(summary.positionCount).toBe(book.positionCount);
    expect(summary.moveCount).toBe(book.moveCount);
    expect(summary.topPositions.length).toBeGreaterThan(0);
    expect(summary.topOpeningMoves).toEqual(topMoves);
    expect(topMoves[0]).toMatchObject({
      ply: expect.any(Number),
      turn: expect.any(String),
      playCount: expect.any(Number),
      scoreRate: expect.any(Number),
      bookScore: expect.any(Number)
    });
  });

  it('provides a built-in seed book with an initial candidate', () => {
    const state = createGameState(createInitialBoard('inner-elephant', 'inner-elephant'), 'CHO');
    const moves = lookupOpeningMoves(builtInOpeningBook, state, { minPlayCount: 2 });

    expect(builtInOpeningBook.positionCount).toBeGreaterThan(0);
    expect(moves.length).toBeGreaterThan(0);
  });
});

function homePattern(board: Board, side: 'CHO' | 'HAN'): string {
  const y = side === 'CHO' ? 9 : 0;
  return [1, 2, 6, 7]
    .map((x) => {
      const kind = board[y][x]?.kind;
      if (kind === 'ELEPHANT') return '상';
      if (kind === 'HORSE') return '마';
      return '?';
    })
    .join('');
}
