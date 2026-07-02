import { Board, Formation, PieceKind, Side, emptyBoard } from './types';

export const formationLabels: Record<Formation, string> = {
  'inner-elephant': '안상차림',
  'outer-elephant': '바깥상차림',
  'left-elephant': '왼상차림',
  'right-elephant': '오른상차림'
};

const formationHomePieces: Record<Formation, Record<number, PieceKind>> = {
  'inner-elephant': {
    1: 'HORSE',
    2: 'ELEPHANT',
    6: 'ELEPHANT',
    7: 'HORSE'
  },
  'outer-elephant': {
    1: 'ELEPHANT',
    2: 'HORSE',
    6: 'HORSE',
    7: 'ELEPHANT'
  },
  'left-elephant': {
    1: 'ELEPHANT',
    2: 'HORSE',
    6: 'ELEPHANT',
    7: 'HORSE'
  },
  'right-elephant': {
    1: 'HORSE',
    2: 'ELEPHANT',
    6: 'HORSE',
    7: 'ELEPHANT'
  }
};

export function createInitialBoard(choFormation: Formation, hanFormation: Formation): Board {
  const board = emptyBoard();
  placeSide(board, 'HAN', hanFormation);
  placeSide(board, 'CHO', choFormation);
  return board;
}

function placeSide(board: Board, side: Side, formation: Formation): void {
  const homeY = side === 'HAN' ? 0 : 9;
  const generalY = side === 'HAN' ? 1 : 8;
  const cannonY = side === 'HAN' ? 2 : 7;
  const soldierY = side === 'HAN' ? 3 : 6;

  board[homeY][0] = { side, kind: 'CHARIOT' };
  board[homeY][8] = { side, kind: 'CHARIOT' };
  board[homeY][3] = { side, kind: 'GUARD' };
  board[homeY][5] = { side, kind: 'GUARD' };
  board[generalY][4] = { side, kind: 'GENERAL' };
  board[cannonY][1] = { side, kind: 'CANNON' };
  board[cannonY][7] = { side, kind: 'CANNON' };

  for (const x of [0, 2, 4, 6, 8]) {
    board[soldierY][x] = { side, kind: 'SOLDIER' };
  }

  const homePieces = formationHomePieces[formation];
  for (const [x, kind] of Object.entries(homePieces)) {
    board[homeY][Number(x)] = { side, kind };
  }
}
