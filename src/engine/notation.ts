import { Move, Piece, PieceKind, Position, Side } from './types';

const pieceLabels: Record<Exclude<PieceKind, 'SOLDIER'>, string> = {
  GENERAL: '장',
  GUARD: '사',
  ELEPHANT: '상',
  HORSE: '마',
  CHARIOT: '차',
  CANNON: '포'
};

export function formatPosition(pos: Position): string {
  return `${pos.x},${pos.y}`;
}

export function formatPlyNumber(index: number): string {
  const moveNumber = Math.floor(index / 2) + 1;
  return index % 2 === 0 ? `${moveNumber}.` : `${moveNumber}...`;
}

export function formatMove(move: Move, side?: Side): string {
  return formatMoveWithPiece(move, move.piece ?? (side ? { side, kind: 'SOLDIER' } : undefined));
}

export function formatMoveWithPiece(move: Move, movingPiece?: Piece): string {
  const label = movingPiece ? formatPieceLabel(movingPiece) : '?';
  const separator = move.captured ? '×' : '→';
  return `${label} ${formatPosition(move.from)}${separator}${formatPosition(move.to)}`;
}

export function formatPieceLabel(piece: Piece): string {
  if (piece.kind === 'SOLDIER') return piece.side === 'CHO' ? '졸' : '병';
  return pieceLabels[piece.kind];
}
