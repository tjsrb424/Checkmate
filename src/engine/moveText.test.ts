import { describe, expect, it } from 'vitest';
import { movesToText, parseMoveLine, parseMoveText, parseMoveTextDetailed } from './index';

describe('move text parser', () => {
  it('parses dash coordinate moves', () => {
    expect(parseMoveLine('0,6-0,5')).toEqual({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } });
  });

  it('parses arrow coordinate moves', () => {
    expect(parseMoveLine('0,6 -> 0,5')).toEqual({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } });
  });

  it('parses capture coordinate moves', () => {
    expect(parseMoveLine('0,6x0,5')).toEqual({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } });
    expect(parseMoveLine('0,6×0,5')).toEqual({ from: { x: 0, y: 6 }, to: { x: 0, y: 5 } });
  });

  it('parses multiple lines and skips invalid lines in the simple API', () => {
    expect(parseMoveText(['0,6-0,5', 'bad', '1,6->1,5'].join('\n'))).toEqual([
      { from: { x: 0, y: 6 }, to: { x: 0, y: 5 } },
      { from: { x: 1, y: 6 }, to: { x: 1, y: 5 } }
    ]);
  });

  it('reports out-of-bounds and invalid lines in the detailed API', () => {
    const result = parseMoveTextDetailed(['9,6-0,5', 'bad'].join('\n'));

    expect(result.moves).toEqual([]);
    expect(result.errors).toEqual([
      { line: 1, text: '9,6-0,5', reason: 'out-of-bounds' },
      { line: 2, text: 'bad', reason: 'invalid-format' }
    ]);
  });

  it('round trips moves to text', () => {
    const moves = parseMoveText('0,6->0,5\n1,6x1,5');

    expect(parseMoveText(movesToText(moves))).toEqual(moves);
  });
});
