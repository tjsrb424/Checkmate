import { Board, GameState, Piece, PieceKind, Side } from './types';

const MASK_64 = (1n << 64n) - 1n;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;

const pieceKinds: PieceKind[] = ['GENERAL', 'GUARD', 'ELEPHANT', 'HORSE', 'CHARIOT', 'CANNON', 'SOLDIER'];
const randomTokenCache = new Map<string, bigint>();

export function computeZobristHash(state: GameState): bigint {
  return computeBoardHash(state.board, state.turn);
}

export function computeBoardHash(board: Board, turn: Side): bigint {
  let hash = randomForToken(`turn:${turn}`);
  for (let y = 0; y < board.length; y += 1) {
    for (let x = 0; x < board[y].length; x += 1) {
      const piece = board[y][x];
      if (piece) {
        hash ^= randomForPiece(piece, x, y);
      }
    }
  }
  return hash & MASK_64;
}

export function hashToKey(hash: bigint): string {
  return hash.toString(16).padStart(16, '0');
}

function randomForPiece(piece: Piece, x: number, y: number): bigint {
  const kindIndex = pieceKinds.indexOf(piece.kind);
  return randomForToken(`piece:${piece.side}:${kindIndex}:${x}:${y}`);
}

function randomForToken(token: string): bigint {
  const cached = randomTokenCache.get(token);
  if (cached !== undefined) return cached;
  let hash = FNV_OFFSET;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= BigInt(token.charCodeAt(i));
    hash = (hash * FNV_PRIME) & MASK_64;
  }
  const value = splitMix64(hash);
  randomTokenCache.set(token, value);
  return value;
}

function splitMix64(seed: bigint): bigint {
  let value = (seed + 0x9e3779b97f4a7c15n) & MASK_64;
  value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK_64;
  value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & MASK_64;
  return (value ^ (value >> 31n)) & MASK_64;
}
