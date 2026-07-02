import {
  Board,
  BOARD_HEIGHT,
  BOARD_WIDTH,
  GameState,
  Move,
  Piece,
  Position,
  Side,
  cloneBoard,
  getPiece,
  inBounds,
  otherSide,
  samePosition,
  setPiece
} from './types';

const orthogonalDirections: Position[] = [
  { x: 1, y: 0 },
  { x: -1, y: 0 },
  { x: 0, y: 1 },
  { x: 0, y: -1 }
];

const horseSteps = [
  { dx: 2, dy: 1, block: { x: 1, y: 0 } },
  { dx: 2, dy: -1, block: { x: 1, y: 0 } },
  { dx: -2, dy: 1, block: { x: -1, y: 0 } },
  { dx: -2, dy: -1, block: { x: -1, y: 0 } },
  { dx: 1, dy: 2, block: { x: 0, y: 1 } },
  { dx: -1, dy: 2, block: { x: 0, y: 1 } },
  { dx: 1, dy: -2, block: { x: 0, y: -1 } },
  { dx: -1, dy: -2, block: { x: 0, y: -1 } }
];

const elephantSteps = [
  { dx: 3, dy: 2, blocks: [{ x: 1, y: 0 }, { x: 2, y: 1 }] },
  { dx: 3, dy: -2, blocks: [{ x: 1, y: 0 }, { x: 2, y: -1 }] },
  { dx: -3, dy: 2, blocks: [{ x: -1, y: 0 }, { x: -2, y: 1 }] },
  { dx: -3, dy: -2, blocks: [{ x: -1, y: 0 }, { x: -2, y: -1 }] },
  { dx: 2, dy: 3, blocks: [{ x: 0, y: 1 }, { x: 1, y: 2 }] },
  { dx: -2, dy: 3, blocks: [{ x: 0, y: 1 }, { x: -1, y: 2 }] },
  { dx: 2, dy: -3, blocks: [{ x: 0, y: -1 }, { x: 1, y: -2 }] },
  { dx: -2, dy: -3, blocks: [{ x: 0, y: -1 }, { x: -1, y: -2 }] }
];

const palaceLines: Position[][] = [
  [{ x: 3, y: 0 }, { x: 4, y: 1 }, { x: 5, y: 2 }],
  [{ x: 5, y: 0 }, { x: 4, y: 1 }, { x: 3, y: 2 }],
  [{ x: 3, y: 7 }, { x: 4, y: 8 }, { x: 5, y: 9 }],
  [{ x: 5, y: 7 }, { x: 4, y: 8 }, { x: 3, y: 9 }]
];

export function createGameState(board: Board, turn: Side = 'CHO'): GameState {
  return { board, turn, history: [] };
}

export function generateLegalMoves(state: GameState, side = state.turn): Move[] {
  return generatePseudoMoves(state.board, side).filter((move) => {
    const next = applyMove(state, move, false);
    return !isInCheck(next.board, side);
  });
}

export function generatePseudoMoves(board: Board, side: Side): Move[] {
  const moves: Move[] = [];
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const piece = board[y][x];
      if (piece?.side !== side) continue;
      moves.push(...pieceMoves(board, { x, y }, piece));
    }
  }
  return moves;
}

export function applyMove(state: GameState, move: Move, appendHistory = true): GameState {
  const board = cloneBoard(state.board);
  const movingPiece = getPiece(board, move.from);
  const captured = getPiece(board, move.to) ?? undefined;
  setPiece(board, move.from, null);
  setPiece(board, move.to, movingPiece);
  const nextTurn = otherSide(state.turn);
  const next: GameState = {
    board,
    turn: nextTurn,
    history: appendHistory ? [...state.history, { ...move, piece: movingPiece ?? move.piece, captured }] : state.history
  };
  if (appendHistory && isCheckmate(board, nextTurn)) {
    next.winner = state.turn;
  }
  return next;
}

export function isLegalMove(state: GameState, move: Move): boolean {
  return generateLegalMoves(state).some((legal) => samePosition(legal.from, move.from) && samePosition(legal.to, move.to));
}

export function isInCheck(board: Board, side: Side): boolean {
  const general = findGeneral(board, side);
  if (!general) return true;
  if (isFacingEnemyGeneral(board, side, general)) return true;
  return isSquareAttacked(board, general, otherSide(side));
}

export function isCheckmate(board: Board, side: Side): boolean {
  return isInCheck(board, side) && generateLegalMoves({ board, turn: side, history: [] }, side).length === 0;
}

export function findGeneral(board: Board, side: Side): Position | null {
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const piece = board[y][x];
      if (piece?.side === side && piece.kind === 'GENERAL') return { x, y };
    }
  }
  return null;
}

function isFacingEnemyGeneral(board: Board, side: Side, general: Position): boolean {
  const enemyGeneral = findGeneral(board, otherSide(side));
  if (!enemyGeneral || enemyGeneral.x !== general.x) return false;

  const step = enemyGeneral.y > general.y ? 1 : -1;
  for (let y = general.y + step; y !== enemyGeneral.y; y += step) {
    if (board[y][general.x]) return false;
  }
  return true;
}

export function isSquareAttacked(board: Board, target: Position, bySide: Side): boolean {
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const piece = board[y][x];
      if (piece?.side !== bySide) continue;
      if (pieceMoves(board, { x, y }, piece).some((move) => samePosition(move.to, target))) {
        return true;
      }
    }
  }
  return false;
}

export function pieceMoves(board: Board, from: Position, piece: Piece): Move[] {
  switch (piece.kind) {
    case 'GENERAL':
    case 'GUARD':
      return palaceStepMoves(board, from, piece);
    case 'CHARIOT':
      return [...rayMoves(board, from, piece, orthogonalDirections), ...palaceRayMoves(board, from, piece, false)];
    case 'HORSE':
      return horseMoves(board, from, piece);
    case 'ELEPHANT':
      return elephantMoves(board, from, piece);
    case 'CANNON':
      return [...cannonRayMoves(board, from, piece, orthogonalDirections), ...palaceRayMoves(board, from, piece, true)];
    case 'SOLDIER':
      return soldierMoves(board, from, piece);
  }
}

function pushIfAvailable(moves: Move[], board: Board, from: Position, to: Position, piece: Piece): void {
  if (!inBounds(to)) return;
  const target = getPiece(board, to);
  if (target?.side === piece.side) return;
  moves.push({ from, to, captured: target ?? undefined });
}

function rayMoves(board: Board, from: Position, piece: Piece, directions: Position[]): Move[] {
  const moves: Move[] = [];
  for (const dir of directions) {
    let to = { x: from.x + dir.x, y: from.y + dir.y };
    while (inBounds(to)) {
      const target = getPiece(board, to);
      if (!target) {
        moves.push({ from, to: { ...to } });
      } else {
        if (target.side !== piece.side) moves.push({ from, to: { ...to }, captured: target });
        break;
      }
      to = { x: to.x + dir.x, y: to.y + dir.y };
    }
  }
  return moves;
}

function horseMoves(board: Board, from: Position, piece: Piece): Move[] {
  const moves: Move[] = [];
  for (const step of horseSteps) {
    if (getPiece(board, { x: from.x + step.block.x, y: from.y + step.block.y })) continue;
    pushIfAvailable(moves, board, from, { x: from.x + step.dx, y: from.y + step.dy }, piece);
  }
  return moves;
}

function elephantMoves(board: Board, from: Position, piece: Piece): Move[] {
  const moves: Move[] = [];
  for (const step of elephantSteps) {
    if (step.blocks.some((block) => getPiece(board, { x: from.x + block.x, y: from.y + block.y }))) continue;
    pushIfAvailable(moves, board, from, { x: from.x + step.dx, y: from.y + step.dy }, piece);
  }
  return moves;
}

function cannonRayMoves(board: Board, from: Position, piece: Piece, directions: Position[]): Move[] {
  const moves: Move[] = [];
  for (const dir of directions) {
    let screen: Piece | null = null;
    let to = { x: from.x + dir.x, y: from.y + dir.y };
    while (inBounds(to)) {
      const target = getPiece(board, to);
      if (!screen) {
        if (target) {
          if (target.kind === 'CANNON') break;
          screen = target;
        }
      } else if (!target) {
        moves.push({ from, to: { ...to } });
      } else {
        if (target.side !== piece.side && target.kind !== 'CANNON') {
          moves.push({ from, to: { ...to }, captured: target });
        }
        break;
      }
      to = { x: to.x + dir.x, y: to.y + dir.y };
    }
  }
  return moves;
}

function palaceStepMoves(board: Board, from: Position, piece: Piece): Move[] {
  const moves: Move[] = [];
  for (const dir of orthogonalDirections) {
    const to = { x: from.x + dir.x, y: from.y + dir.y };
    if (isInPalace(to, piece.side)) pushIfAvailable(moves, board, from, to, piece);
  }
  for (const to of adjacentPalaceDiagonalPositions(from)) {
    if (isInPalace(to, piece.side)) pushIfAvailable(moves, board, from, to, piece);
  }
  return moves;
}

function soldierMoves(board: Board, from: Position, piece: Piece): Move[] {
  const moves: Move[] = [];
  const forward = piece.side === 'CHO' ? -1 : 1;
  for (const dir of [
    { x: 0, y: forward },
    { x: 1, y: 0 },
    { x: -1, y: 0 }
  ]) {
    pushIfAvailable(moves, board, from, { x: from.x + dir.x, y: from.y + dir.y }, piece);
  }

  for (const to of adjacentPalaceDiagonalPositions(from)) {
    if (to.y - from.y === forward && isInPalace(to, otherSide(piece.side))) {
      pushIfAvailable(moves, board, from, to, piece);
    }
  }
  return moves;
}

function palaceRayMoves(board: Board, from: Position, piece: Piece, cannon: boolean): Move[] {
  const moves: Move[] = [];
  for (const line of palaceLinesContaining(from)) {
    const index = line.findIndex((pos) => samePosition(pos, from));
    for (const direction of [-1, 1]) {
      let screen: Piece | null = null;
      for (let i = index + direction; i >= 0 && i < line.length; i += direction) {
        const to = line[i];
        const target = getPiece(board, to);
        if (!cannon) {
          if (!target) {
            moves.push({ from, to: { ...to } });
            continue;
          }
          if (target.side !== piece.side) moves.push({ from, to: { ...to }, captured: target });
          break;
        }

        if (!screen) {
          if (target) {
            if (target.kind === 'CANNON') break;
            screen = target;
          }
          continue;
        }

        if (!target) {
          moves.push({ from, to: { ...to } });
          continue;
        }
        if (target.side !== piece.side && target.kind !== 'CANNON') {
          moves.push({ from, to: { ...to }, captured: target });
        }
        break;
      }
    }
  }
  return moves;
}

function palaceLinesContaining(pos: Position): Position[][] {
  return palaceLines.filter((line) => line.some((linePos) => samePosition(linePos, pos)));
}

function adjacentPalaceDiagonalPositions(pos: Position): Position[] {
  const result: Position[] = [];
  for (const line of palaceLinesContaining(pos)) {
    const index = line.findIndex((linePos) => samePosition(linePos, pos));
    for (const nextIndex of [index - 1, index + 1]) {
      const next = line[nextIndex];
      if (next) result.push({ ...next });
    }
  }
  return result;
}

export function isInPalace(pos: Position, side: Side): boolean {
  const minY = side === 'HAN' ? 0 : 7;
  const maxY = side === 'HAN' ? 2 : 9;
  return pos.x >= 3 && pos.x <= 5 && pos.y >= minY && pos.y <= maxY;
}
