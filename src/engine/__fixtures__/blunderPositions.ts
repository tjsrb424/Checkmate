import { createGameState } from '../rules';
import { emptyBoard, setPiece } from '../types';
import type { Board, Formation, GameState, PieceKind, Side } from '../types';

export interface BlunderPositionFixture {
  id: string;
  label: string;
  choFormation: Formation;
  hanFormation: Formation;
  moveText: string;
  sideToMove: Side;
  forbiddenMoves: string[];
  expectedMoves: string[];
  notes: string;
  tags: string[];
  createState: () => GameState;
}

export const blunderPositionFixtures: BlunderPositionFixture[] = [
  {
    id: 'synthetic-allows-immediate-mate',
    label: '즉시 외통 허용 회귀 포지션',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    moveText: '',
    sideToMove: 'CHO',
    forbiddenMoves: ['8,8-7,8', '2,6-2,7'],
    expectedMoves: ['2,6-2,1'],
    tags: ['allows-mate', 'losing-blunder'],
    notes: 'Non-forcing quiet moves allow HAN immediate mate; CHO should keep a high-scoring defensive chariot move.',
    createState: () => createGameState(mateThreatBoard(), 'CHO')
  },
  {
    id: 'synthetic-free-chariot',
    label: '차 헌납 회귀 포지션',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    moveText: '',
    sideToMove: 'CHO',
    forbiddenMoves: ['0,4-1,4'],
    expectedMoves: ['0,4-0,3'],
    tags: ['hangs-chariot'],
    notes: 'Moving the soldier leaves the CHO chariot hanging to HAN chariot pressure.',
    createState: () => createGameState(hangingChariotBoard(), 'CHO')
  },
  {
    id: 'synthetic-free-cannon',
    label: '포 헌납 회귀 포지션',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    moveText: '',
    sideToMove: 'CHO',
    forbiddenMoves: ['0,7-0,2'],
    expectedMoves: [],
    tags: ['hangs-cannon'],
    notes: 'CHO cannon is under immediate chariot capture pressure; aggressive cannon movement should not become best.',
    createState: () => createGameState(hangingCannonBoard(), 'CHO')
  },
  {
    id: 'synthetic-bad-trade',
    label: '나쁜 교환 회귀 포지션',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    moveText: '',
    sideToMove: 'CHO',
    forbiddenMoves: ['0,6-1,6'],
    expectedMoves: ['0,6-0,1'],
    tags: ['bad-trade', 'hangs-chariot'],
    notes: 'CHO can win a soldier but then loses a chariot to the recapture file.',
    createState: () => createGameState(badTradeBoard(), 'CHO')
  }
];

function baseBoard(): Board {
  const board = emptyBoard();
  place(board, 4, 8, 'CHO', 'GENERAL');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 4, 5, 'CHO', 'SOLDIER');
  return board;
}

function hangingChariotBoard(): Board {
  const board = baseBoard();
  place(board, 0, 6, 'CHO', 'CHARIOT');
  place(board, 0, 4, 'CHO', 'SOLDIER');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

function hangingCannonBoard(): Board {
  const board = baseBoard();
  place(board, 0, 7, 'CHO', 'CANNON');
  place(board, 0, 3, 'HAN', 'CHARIOT');
  return board;
}

function badTradeBoard(): Board {
  const board = baseBoard();
  place(board, 0, 6, 'CHO', 'CHARIOT');
  place(board, 1, 6, 'HAN', 'SOLDIER');
  place(board, 1, 3, 'HAN', 'CHARIOT');
  return board;
}

function mateThreatBoard(): Board {
  const board = emptyBoard();
  place(board, 3, 9, 'CHO', 'GENERAL');
  place(board, 3, 8, 'CHO', 'GUARD');
  place(board, 2, 6, 'CHO', 'CHARIOT');
  place(board, 8, 8, 'CHO', 'SOLDIER');
  place(board, 4, 1, 'HAN', 'GENERAL');
  place(board, 3, 6, 'HAN', 'CHARIOT');
  place(board, 5, 7, 'HAN', 'HORSE');
  return board;
}

function place(board: Board, x: number, y: number, side: Side, kind: PieceKind): void {
  setPiece(board, { x, y }, { side, kind });
}
