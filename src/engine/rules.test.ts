import { describe, expect, it } from 'vitest';
import {
  Board,
  GameState,
  Move,
  PieceKind,
  Side,
  applyMove,
  countPositionOccurrences,
  createGameState,
  createInitialBoard,
  emptyBoard,
  generateLegalMoves,
  generatePseudoMoves,
  hashToKey,
  computeZobristHash,
  isCheckmate,
  isInCheck,
  isLegalMove,
  pieceMoves,
  setPiece,
  wouldRepeatPosition
} from './index';

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}

function hasMove(moves: Move[], fromX: number, fromY: number, toX: number, toY: number): boolean {
  return moves.some((move) => move.from.x === fromX && move.from.y === fromY && move.to.x === toX && move.to.y === toY);
}

function state(board: Board, turn: Side = 'CHO'): GameState {
  return { board, turn, history: [] };
}

function repeatedPositionBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  place(board, 0, 5, 'CHO', 'CHARIOT');
  return board;
}

function homePattern(board: Board, side: Side): string {
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

function stateKey(game: GameState): string {
  return hashToKey(computeZobristHash(game));
}

describe('initial formations', () => {
  it('places all four elephant formations on the home row', () => {
    const inner = createInitialBoard('inner-elephant', 'outer-elephant');
    expect(inner[9][1]?.kind).toBe('HORSE');
    expect(inner[9][2]?.kind).toBe('ELEPHANT');
    expect(inner[9][6]?.kind).toBe('ELEPHANT');
    expect(inner[9][7]?.kind).toBe('HORSE');

    expect(inner[0][1]?.kind).toBe('ELEPHANT');
    expect(inner[0][2]?.kind).toBe('HORSE');
    expect(inner[0][6]?.kind).toBe('HORSE');
    expect(inner[0][7]?.kind).toBe('ELEPHANT');

    const left = createInitialBoard('left-elephant', 'right-elephant');
    expect(left[9][1]?.kind).toBe('ELEPHANT');
    expect(left[9][2]?.kind).toBe('HORSE');
    expect(left[9][6]?.kind).toBe('ELEPHANT');
    expect(left[9][7]?.kind).toBe('HORSE');
    expect(left[0][1]?.kind).toBe('HORSE');
    expect(left[0][2]?.kind).toBe('ELEPHANT');
    expect(left[0][6]?.kind).toBe('HORSE');
    expect(left[0][7]?.kind).toBe('ELEPHANT');
  });

  it('uses bottom-view left and right elephant labels for both sides', () => {
    const left = createInitialBoard('left-elephant', 'left-elephant');
    const right = createInitialBoard('right-elephant', 'right-elephant');

    expect(homePattern(left, 'CHO')).toBe('상마상마');
    expect(homePattern(left, 'HAN')).toBe('상마상마');
    expect(homePattern(right, 'CHO')).toBe('마상마상');
    expect(homePattern(right, 'HAN')).toBe('마상마상');
  });
});

describe('piece movement rules', () => {
  it('blocks horse moves at the first orthogonal step', () => {
    const board = emptyBoard();
    place(board, 4, 4, 'CHO', 'HORSE');
    place(board, 5, 4, 'CHO', 'SOLDIER');

    const moves = pieceMoves(board, { x: 4, y: 4 }, board[4][4]!);
    expect(hasMove(moves, 4, 4, 6, 5)).toBe(false);
    expect(hasMove(moves, 4, 4, 6, 3)).toBe(false);
    expect(hasMove(moves, 4, 4, 5, 6)).toBe(true);
  });

  it('blocks elephant moves at either intervening point', () => {
    const board = emptyBoard();
    place(board, 4, 4, 'CHO', 'ELEPHANT');
    place(board, 5, 4, 'HAN', 'SOLDIER');
    place(board, 2, 3, 'HAN', 'SOLDIER');

    const moves = pieceMoves(board, { x: 4, y: 4 }, board[4][4]!);
    expect(hasMove(moves, 4, 4, 7, 6)).toBe(false);
    expect(hasMove(moves, 4, 4, 1, 2)).toBe(false);
    expect(hasMove(moves, 4, 4, 6, 7)).toBe(true);
  });

  it('requires exactly one non-cannon screen for cannon movement', () => {
    const board = emptyBoard();
    place(board, 0, 0, 'CHO', 'CANNON');
    place(board, 0, 2, 'CHO', 'SOLDIER');
    place(board, 0, 5, 'HAN', 'HORSE');

    const moves = pieceMoves(board, { x: 0, y: 0 }, board[0][0]!);
    expect(hasMove(moves, 0, 0, 0, 1)).toBe(false);
    expect(hasMove(moves, 0, 0, 0, 3)).toBe(true);
    expect(hasMove(moves, 0, 0, 0, 4)).toBe(true);
    expect(hasMove(moves, 0, 0, 0, 5)).toBe(true);
  });

  it('prevents cannons from jumping cannons or capturing cannons', () => {
    const blockedByCannon = emptyBoard();
    place(blockedByCannon, 0, 0, 'CHO', 'CANNON');
    place(blockedByCannon, 0, 2, 'HAN', 'CANNON');
    expect(pieceMoves(blockedByCannon, { x: 0, y: 0 }, blockedByCannon[0][0]!).length).toBe(0);

    const targetCannon = emptyBoard();
    place(targetCannon, 0, 0, 'CHO', 'CANNON');
    place(targetCannon, 0, 2, 'HAN', 'SOLDIER');
    place(targetCannon, 0, 5, 'HAN', 'CANNON');
    const moves = pieceMoves(targetCannon, { x: 0, y: 0 }, targetCannon[0][0]!);
    expect(hasMove(moves, 0, 0, 0, 5)).toBe(false);
  });

  it('allows palace diagonal movement on marked lines', () => {
    const board = emptyBoard();
    place(board, 4, 8, 'CHO', 'GENERAL');
    const guardBoard = emptyBoard();
    place(guardBoard, 3, 7, 'CHO', 'GUARD');

    const generalMoves = pieceMoves(board, { x: 4, y: 8 }, board[8][4]!);
    const guardMoves = pieceMoves(guardBoard, { x: 3, y: 7 }, guardBoard[7][3]!);
    expect(hasMove(generalMoves, 4, 8, 5, 7)).toBe(true);
    expect(hasMove(guardMoves, 3, 7, 4, 8)).toBe(true);
  });

  it('detects check from a chariot', () => {
    const board = emptyBoard();
    place(board, 4, 1, 'HAN', 'GENERAL');
    place(board, 4, 4, 'CHO', 'CHARIOT');
    expect(isInCheck(board, 'HAN')).toBe(true);
  });

  it('detects checkmate when the general has no safe palace move', () => {
    const board = emptyBoard();
    place(board, 3, 0, 'HAN', 'GENERAL');
    place(board, 3, 3, 'CHO', 'CHARIOT');
    place(board, 4, 3, 'CHO', 'CHARIOT');

    expect(isInCheck(board, 'HAN')).toBe(true);
    expect(isCheckmate(board, 'HAN')).toBe(true);
  });

  it('filters legal moves that leave the general in check', () => {
    const board = emptyBoard();
    place(board, 4, 8, 'CHO', 'GENERAL');
    place(board, 4, 7, 'CHO', 'GUARD');
    place(board, 4, 4, 'HAN', 'CHARIOT');

    const legalMoves = generateLegalMoves(state(board, 'CHO'));
    const pseudoMoves = generatePseudoMoves(board, 'CHO');
    expect(hasMove(pseudoMoves, 4, 7, 3, 7)).toBe(true);
    expect(hasMove(legalMoves, 4, 7, 3, 7)).toBe(false);
  });

  it('detects facing generals as check', () => {
    const board = emptyBoard();
    place(board, 4, 1, 'HAN', 'GENERAL');
    place(board, 4, 8, 'CHO', 'GENERAL');

    expect(isInCheck(board, 'CHO')).toBe(true);
    expect(isInCheck(board, 'HAN')).toBe(true);
  });

  it('prevents moves that expose facing generals', () => {
    const board = emptyBoard();
    place(board, 4, 1, 'HAN', 'GENERAL');
    place(board, 4, 8, 'CHO', 'GENERAL');
    place(board, 4, 5, 'CHO', 'CHARIOT');

    const pseudoMoves = generatePseudoMoves(board, 'CHO');
    const legalMoves = generateLegalMoves(state(board, 'CHO'));
    expect(hasMove(pseudoMoves, 4, 5, 3, 5)).toBe(true);
    expect(hasMove(legalMoves, 4, 5, 3, 5)).toBe(false);
  });

  it('allows soldiers to advance diagonally inside the enemy palace only', () => {
    const board = emptyBoard();
    place(board, 4, 1, 'CHO', 'SOLDIER');
    const moves = pieceMoves(board, { x: 4, y: 1 }, board[1][4]!);

    expect(hasMove(moves, 4, 1, 3, 0)).toBe(true);
    expect(hasMove(moves, 4, 1, 5, 0)).toBe(true);
    expect(hasMove(moves, 4, 1, 3, 2)).toBe(false);
    expect(hasMove(moves, 4, 1, 5, 2)).toBe(false);
  });

  it('allows chariot palace diagonal movement and capture when the line is clear', () => {
    const board = emptyBoard();
    place(board, 3, 0, 'CHO', 'CHARIOT');
    place(board, 5, 2, 'HAN', 'HORSE');
    const moves = pieceMoves(board, { x: 3, y: 0 }, board[0][3]!);
    expect(hasMove(moves, 3, 0, 5, 2)).toBe(true);

    place(board, 4, 1, 'CHO', 'SOLDIER');
    const blockedMoves = pieceMoves(board, { x: 3, y: 0 }, board[0][3]!);
    expect(hasMove(blockedMoves, 3, 0, 5, 2)).toBe(false);
  });

  it('allows cannon palace diagonal movement with exactly one non-cannon screen', () => {
    const board = emptyBoard();
    place(board, 3, 0, 'CHO', 'CANNON');
    place(board, 4, 1, 'CHO', 'SOLDIER');
    place(board, 5, 2, 'HAN', 'HORSE');
    const moves = pieceMoves(board, { x: 3, y: 0 }, board[0][3]!);
    expect(hasMove(moves, 3, 0, 5, 2)).toBe(true);

    const cannonScreen = emptyBoard();
    place(cannonScreen, 3, 0, 'CHO', 'CANNON');
    place(cannonScreen, 4, 1, 'CHO', 'CANNON');
    place(cannonScreen, 5, 2, 'HAN', 'HORSE');
    expect(hasMove(pieceMoves(cannonScreen, { x: 3, y: 0 }, cannonScreen[0][3]!), 3, 0, 5, 2)).toBe(false);

    const cannonTarget = emptyBoard();
    place(cannonTarget, 3, 0, 'CHO', 'CANNON');
    place(cannonTarget, 4, 1, 'CHO', 'SOLDIER');
    place(cannonTarget, 5, 2, 'HAN', 'CANNON');
    expect(hasMove(pieceMoves(cannonTarget, { x: 3, y: 0 }, cannonTarget[0][3]!), 3, 0, 5, 2)).toBe(false);
  });
});

describe('threefold repetition move ban', () => {
  it('initializes position history with the initial position key', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');

    expect(game.positionHistory).toEqual([stateKey(game)]);
  });

  it('accumulates position history even when move history is not appended', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');
    const next = applyMove(game, { from: { x: 0, y: 5 }, to: { x: 1, y: 5 } }, false);

    expect(next.history).toHaveLength(0);
    expect(next.positionHistory).toHaveLength(2);
    expect(next.positionHistory?.[0]).toBe(stateKey(game));
    expect(next.positionHistory?.[1]).toBe(stateKey(next));
  });

  it('allows a second occurrence but not a third occurrence', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');
    const move: Move = { from: { x: 0, y: 5 }, to: { x: 1, y: 5 } };
    const targetKey = stateKey(applyMove(game, move, false));
    const currentKey = stateKey(game);
    const secondOccurrenceState: GameState = {
      ...game,
      positionHistory: [targetKey, currentKey]
    };
    const thirdOccurrenceState: GameState = {
      ...game,
      positionHistory: [targetKey, currentKey, targetKey, currentKey]
    };

    expect(countPositionOccurrences(secondOccurrenceState, targetKey)).toBe(1);
    expect(wouldRepeatPosition(secondOccurrenceState, move)).toBe(false);
    expect(hasMove(generateLegalMoves(secondOccurrenceState), 0, 5, 1, 5)).toBe(true);

    expect(countPositionOccurrences(thirdOccurrenceState, targetKey)).toBe(2);
    expect(wouldRepeatPosition(thirdOccurrenceState, move)).toBe(true);
    expect(hasMove(generateLegalMoves(thirdOccurrenceState), 0, 5, 1, 5)).toBe(false);
  });

  it('prevents moves that create a third repeated position', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');
    const move: Move = { from: { x: 0, y: 5 }, to: { x: 1, y: 5 } };
    const targetKey = stateKey(applyMove(game, move, false));
    const currentKey = stateKey(game);
    const repeatedState: GameState = {
      ...game,
      positionHistory: [targetKey, currentKey, targetKey, currentKey]
    };

    expect(generateLegalMoves(repeatedState).some((legal) => stateKey(applyMove(repeatedState, legal, false)) === targetKey)).toBe(false);
  });

  it('makes isLegalMove reject repeated positions', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');
    const move: Move = { from: { x: 0, y: 5 }, to: { x: 1, y: 5 } };
    const targetKey = stateKey(applyMove(game, move, false));
    const repeatedState: GameState = {
      ...game,
      positionHistory: [targetKey, stateKey(game), targetKey, stateKey(game)]
    };

    expect(isLegalMove(repeatedState, move)).toBe(false);
  });

  it('allows repeated positions when the ruleset repetition policy is off', () => {
    const game = createGameState(repeatedPositionBoard(), 'CHO');
    const move: Move = { from: { x: 0, y: 5 }, to: { x: 1, y: 5 } };
    const targetKey = stateKey(applyMove(game, move, false));
    const repeatedState: GameState = {
      ...game,
      positionHistory: [targetKey, stateKey(game), targetKey, stateKey(game)]
    };
    const repetitionOff = {
      id: 'oetongsu-basic' as const,
      label: 'repetition off',
      repetitionPolicy: 'off' as const,
      bikjangPolicy: 'off' as const,
      passPolicy: 'off' as const,
      scoringPolicy: 'off' as const,
      maxPlyPolicy: 'draw' as const
    };

    expect(wouldRepeatPosition(repeatedState, move, repetitionOff)).toBe(false);
    expect(isLegalMove(repeatedState, move, repetitionOff)).toBe(true);
  });
});
