import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import {
  Difficulty,
  Formation,
  GameState,
  Move,
  Position,
  Side,
  applyMove,
  chooseBestMove,
  createGameState,
  createInitialBoard,
  formationLabels,
  generateLegalMoves,
  isInCheck,
  moveKey,
  otherSide,
  positionKey,
  samePosition
} from './engine';

const pieceLabels = {
  CHO: {
    GENERAL: '楚',
    GUARD: '士',
    ELEPHANT: '象',
    HORSE: '馬',
    CHARIOT: '車',
    CANNON: '包',
    SOLDIER: '卒'
  },
  HAN: {
    GENERAL: '漢',
    GUARD: '士',
    ELEPHANT: '象',
    HORSE: '馬',
    CHARIOT: '車',
    CANNON: '包',
    SOLDIER: '兵'
  }
} as const;

const sideLabels: Record<Side, string> = {
  CHO: '초',
  HAN: '한'
};

const difficultyLabels: Record<Difficulty, string> = {
  easy: '쉬움',
  normal: '보통',
  hard: '어려움'
};

const formationOptions: Formation[] = ['inner-elephant', 'outer-elephant', 'left-elephant', 'right-elephant'];
const difficultyOptions: Difficulty[] = ['easy', 'normal', 'hard'];

export default function App() {
  const [humanSide, setHumanSide] = useState<Side>('CHO');
  const [choFormation, setChoFormation] = useState<Formation>('inner-elephant');
  const [hanFormation, setHanFormation] = useState<Formation>('inner-elephant');
  const [difficulty, setDifficulty] = useState<Difficulty>('normal');
  const [game, setGame] = useState<GameState>(() => createGameState(createInitialBoard('inner-elephant', 'inner-elephant')));
  const [selected, setSelected] = useState<Position | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastSearch, setLastSearch] = useState('');

  const aiSide = otherSide(humanSide);
  const legalMoves = useMemo(() => generateLegalMoves(game), [game]);
  const selectedMoves = useMemo(
    () => (selected ? legalMoves.filter((move) => samePosition(move.from, selected)) : []),
    [legalMoves, selected]
  );
  const legalTargetKeys = new Set(selectedMoves.map((move) => positionKey(move.to)));
  const isAiTurn = game.turn === aiSide && !game.winner;
  const checkSide = isInCheck(game.board, game.turn) ? game.turn : null;

  useEffect(() => {
    if (!isAiTurn) return;
    setAiThinking(true);
    const timer = window.setTimeout(() => {
      const result = chooseBestMove(game, difficulty);
      if (result.move) {
        setGame((current) => applyMove(current, result.move!, true));
        setLastSearch(
          `깊이 ${result.depth || 1}, 평가 ${Math.round(result.score)}, 노드 ${result.nodes}, NPS ${result.nps}, TT ${result.ttHits}`
        );
      }
      setSelected(null);
      setAiThinking(false);
    }, 180);
    return () => window.clearTimeout(timer);
  }, [difficulty, game, isAiTurn]);

  function startNewGame() {
    setGame(createGameState(createInitialBoard(choFormation, hanFormation), 'CHO'));
    setSelected(null);
    setLastSearch('');
    setAiThinking(false);
  }

  function handleCellClick(pos: Position) {
    if (isAiTurn || game.winner) return;
    const piece = game.board[pos.y][pos.x];
    if (piece?.side === game.turn) {
      setSelected(pos);
      return;
    }

    if (!selected) return;
    const move = selectedMoves.find((candidate) => samePosition(candidate.to, pos));
    if (!move) {
      setSelected(null);
      return;
    }
    setGame((current) => applyMove(current, move, true));
    setSelected(null);
  }

  return (
    <main className="appShell">
      <section className="gameSurface">
        <header className="topBar">
          <div>
            <p className="eyebrow">프로젝트 외통수</p>
            <h1>사람 vs AI 장기</h1>
          </div>
          <div className="statusPanel">
            <span className={`turnBadge ${game.turn.toLowerCase()}`}>{sideLabels[game.turn]} 차례</span>
            <strong>{game.winner ? `${sideLabels[game.winner]} 승리` : checkSide ? '장군' : aiThinking ? 'AI 사고 중' : '대국 진행'}</strong>
            <small>{lastSearch || '탐색 기반 착수'}</small>
          </div>
        </header>

        <div className="playLayout">
          <BoardView
            game={game}
            humanSide={humanSide}
            selected={selected}
            legalTargetKeys={legalTargetKeys}
            lastMove={game.history.length > 0 ? game.history[game.history.length - 1] : null}
            onCellClick={handleCellClick}
          />

          <aside className="controlRail">
            <div className="controlGroup">
              <span className="groupLabel">사용자 진영</span>
              <div className="segmented">
                {(['CHO', 'HAN'] as Side[]).map((side) => (
                  <button key={side} className={humanSide === side ? 'active' : ''} onClick={() => setHumanSide(side)}>
                    {sideLabels[side]}
                  </button>
                ))}
              </div>
            </div>

            <FormationSelect label="초 차림" value={choFormation} onChange={setChoFormation} />
            <FormationSelect label="한 차림" value={hanFormation} onChange={setHanFormation} />

            <div className="controlGroup">
              <span className="groupLabel">AI 난이도</span>
              <div className="segmented">
                {difficultyOptions.map((level) => (
                  <button key={level} className={difficulty === level ? 'active' : ''} onClick={() => setDifficulty(level)}>
                    {difficultyLabels[level]}
                  </button>
                ))}
              </div>
            </div>

            <button className="primaryAction" onClick={startNewGame}>
              새 대국
            </button>

            <MoveList history={game.history} />
          </aside>
        </div>
      </section>
    </main>
  );
}

function FormationSelect({
  label,
  value,
  onChange
}: {
  label: string;
  value: Formation;
  onChange: (value: Formation) => void;
}) {
  return (
    <label className="controlGroup">
      <span className="groupLabel">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value as Formation)}>
        {formationOptions.map((formation) => (
          <option key={formation} value={formation}>
            {formationLabels[formation]}
          </option>
        ))}
      </select>
    </label>
  );
}

function BoardView({
  game,
  humanSide,
  selected,
  legalTargetKeys,
  lastMove,
  onCellClick
}: {
  game: GameState;
  humanSide: Side;
  selected: Position | null;
  legalTargetKeys: Set<string>;
  lastMove: Move | null;
  onCellClick: (pos: Position) => void;
}) {
  const points: Array<{ display: Position; board: Position }> = [];
  for (let displayY = 0; displayY < 10; displayY += 1) {
    for (let displayX = 0; displayX < 9; displayX += 1) {
      const display = { x: displayX, y: displayY };
      points.push({ display, board: displayToBoard(display, humanSide) });
    }
  }

  return (
    <div className="boardFrame">
      <div className="janggiBoard">
        <svg className="boardLines" viewBox="0 0 8 9" preserveAspectRatio="none" aria-hidden="true">
          {Array.from({ length: 9 }, (_, x) => (
            <line key={`file-${x}`} x1={x} y1="0" x2={x} y2="9" vectorEffect="non-scaling-stroke" />
          ))}
          {Array.from({ length: 10 }, (_, y) => (
            <line key={`rank-${y}`} x1="0" y1={y} x2="8" y2={y} vectorEffect="non-scaling-stroke" />
          ))}
          <line x1="3" y1="0" x2="5" y2="2" vectorEffect="non-scaling-stroke" />
          <line x1="5" y1="0" x2="3" y2="2" vectorEffect="non-scaling-stroke" />
          <line x1="3" y1="7" x2="5" y2="9" vectorEffect="non-scaling-stroke" />
          <line x1="5" y1="7" x2="3" y2="9" vectorEffect="non-scaling-stroke" />
        </svg>
        {points.map(({ display, board }) => {
          const piece = game.board[board.y][board.x];
          const key = positionKey(board);
          const selectedCell = selected ? samePosition(selected, board) : false;
          const last = lastMove && (samePosition(lastMove.from, board) || samePosition(lastMove.to, board));
          return (
            <button
              key={key}
              className={`intersectionPoint ${selectedCell ? 'selected' : ''} ${
                legalTargetKeys.has(key) ? 'legalTarget' : ''
              } ${last ? 'lastMove' : ''}`}
              style={intersectionStyle(display)}
              onClick={() => onCellClick(board)}
              aria-label={`${board.x},${board.y}`}
            >
              {piece && (
                <span className={`piece ${piece.side.toLowerCase()}`}>
                  {pieceLabels[piece.side][piece.kind]}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function intersectionStyle(display: Position): CSSProperties {
  return {
    '--x': display.x,
    '--y': display.y,
    left: boardOffset(display.x),
    top: boardOffset(display.y)
  } as CSSProperties;
}

function boardOffset(index: number): string {
  if (index === 0) return 'var(--board-pad)';
  return `calc(var(--board-pad) + ${Array.from({ length: index }, () => 'var(--point-gap)').join(' + ')})`;
}

function MoveList({ history }: { history: Move[] }) {
  return (
    <div className="moveList">
      <div className="moveListHeader">
        <span className="groupLabel">수순 기록</span>
        <strong>{history.length}</strong>
      </div>
      <ol>
        {history.map((move, index) => (
          <li key={`${moveKey(move)}-${index}`}>
            <span>{index % 2 === 0 ? '초' : '한'}</span>
            <code>
              {move.from.x},{move.from.y} → {move.to.x},{move.to.y}
            </code>
          </li>
        ))}
      </ol>
    </div>
  );
}

function displayToBoard(pos: Position, humanSide: Side): Position {
  if (humanSide === 'CHO') return pos;
  return { x: 8 - pos.x, y: 9 - pos.y };
}
