import { inBounds, moveKey } from './types';
import type { Move } from './types';

export interface MoveTextParseError {
  line: number;
  text: string;
  reason: string;
}

export interface MoveTextParseResult {
  moves: Move[];
  errors: MoveTextParseError[];
}

const moveLinePattern = /^\s*(\d+)\s*,\s*(\d+)\s*(?:->|[-x×])\s*(\d+)\s*,\s*(\d+)\s*$/;

export function parseMoveLine(line: string): Move | null {
  const match = moveLinePattern.exec(line);
  if (!match) return null;

  const move: Move = {
    from: { x: Number(match[1]), y: Number(match[2]) },
    to: { x: Number(match[3]), y: Number(match[4]) }
  };
  return inBounds(move.from) && inBounds(move.to) ? move : null;
}

export function parseMoveText(text: string): Move[] {
  return parseMoveTextDetailed(text).moves;
}

export function parseMoveTextDetailed(text: string): MoveTextParseResult {
  const moves: Move[] = [];
  const errors: MoveTextParseError[] = [];

  text.split(/\r?\n/).forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) return;

    const move = parseMoveLine(line);
    if (move) {
      moves.push(move);
      return;
    }

    errors.push({
      line: index + 1,
      text: rawLine,
      reason: moveLinePattern.test(line) ? 'out-of-bounds' : 'invalid-format'
    });
  });

  return { moves, errors };
}

export function movesToText(moves: Move[]): string {
  return moves.map(moveKey).join('\n');
}
