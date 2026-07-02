export type Side = 'CHO' | 'HAN';

export type PieceKind =
  | 'GENERAL'
  | 'GUARD'
  | 'ELEPHANT'
  | 'HORSE'
  | 'CHARIOT'
  | 'CANNON'
  | 'SOLDIER';

export type Formation = 'left-elephant' | 'right-elephant' | 'inner-elephant' | 'outer-elephant';

export type Difficulty = 'easy' | 'normal' | 'hard';

export interface Piece {
  side: Side;
  kind: PieceKind;
}

export interface Position {
  x: number;
  y: number;
}

export interface Move {
  from: Position;
  to: Position;
  piece?: Piece;
  captured?: Piece;
  score?: number;
}

export type Board = Array<Array<Piece | null>>;

export interface GameState {
  board: Board;
  turn: Side;
  history: Move[];
  positionHistory?: string[];
  winner?: Side;
}

export interface SearchLimits {
  maxDepth: number;
  timeMs: number;
}

export const BOARD_WIDTH = 9;
export const BOARD_HEIGHT = 10;

export const sides: Side[] = ['CHO', 'HAN'];

export function otherSide(side: Side): Side {
  return side === 'CHO' ? 'HAN' : 'CHO';
}

export function samePosition(a: Position, b: Position): boolean {
  return a.x === b.x && a.y === b.y;
}

export function inBounds(pos: Position): boolean {
  return pos.x >= 0 && pos.x < BOARD_WIDTH && pos.y >= 0 && pos.y < BOARD_HEIGHT;
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => row.map((piece) => (piece ? { ...piece } : null)));
}

export function emptyBoard(): Board {
  return Array.from({ length: BOARD_HEIGHT }, () => Array.from({ length: BOARD_WIDTH }, () => null));
}

export function getPiece(board: Board, pos: Position): Piece | null {
  if (!inBounds(pos)) return null;
  return board[pos.y][pos.x];
}

export function setPiece(board: Board, pos: Position, piece: Piece | null): void {
  if (!inBounds(pos)) return;
  board[pos.y][pos.x] = piece;
}

export function moveKey(move: Move): string {
  return `${move.from.x},${move.from.y}-${move.to.x},${move.to.y}`;
}

export function positionKey(pos: Position): string {
  return `${pos.x},${pos.y}`;
}
