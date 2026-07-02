import { Board, GameState, Move, Side, cloneBoard } from './types';
import { applyMove, createGameState, isLegalMove } from './rules';

export function replayGame(initialBoard: Board, moves: Move[], startingTurn: Side = 'CHO'): GameState {
  return moves.reduce(
    (state, move) => applyMove(state, move, true),
    createGameState(cloneBoard(initialBoard), startingTurn)
  );
}

export function undoMoves(initialBoard: Board, history: Move[], count: number, startingTurn: Side = 'CHO'): GameState {
  const remainingMoves = history.slice(0, Math.max(0, history.length - Math.max(0, count)));
  return replayGame(initialBoard, remainingMoves, startingTurn);
}

export function undoLastMove(game: GameState, initialBoard: Board, startingTurn: Side = 'CHO'): GameState {
  return undoMoves(initialBoard, game.history, 1, startingTurn);
}

export function movesToUndoForHumanTurn(game: GameState, initialBoard: Board, humanSide: Side, startingTurn: Side = 'CHO'): number {
  for (let count = 1; count <= game.history.length; count += 1) {
    const candidate = undoMoves(initialBoard, game.history, count, startingTurn);
    if (candidate.turn === humanSide) return count;
  }
  return Math.min(1, game.history.length);
}

export function undoToHumanTurn(
  game: GameState,
  initialBoard: Board,
  humanSide: Side,
  startingTurn: Side = 'CHO'
): GameState {
  return undoMoves(initialBoard, game.history, movesToUndoForHumanTurn(game, initialBoard, humanSide, startingTurn), startingTurn);
}

export function canRedoMove(game: GameState, move: Move): boolean {
  return isLegalMove(game, move);
}
