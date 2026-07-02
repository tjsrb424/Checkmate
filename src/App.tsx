import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import {
  Difficulty,
  Formation,
  GameState,
  Move,
  type PositionAnalysis,
  Position,
  Side,
  analyzePosition,
  applyMove,
  builtInOpeningBook,
  canRedoMove,
  createGameState,
  createInitialBoard,
  formationLabels,
  formatMoveWithPiece,
  formatPlyNumber,
  generateLegalMoves,
  isInCheck,
  isLegalMove,
  movesToUndoForHumanTurn,
  moveKey,
  otherSide,
  positionKey,
  samePosition,
  undoLastMove,
  undoMoves
} from './engine';
import {
  createAiSearchRequest,
  formatSearchSummary,
  isAiWorkerResponse,
  isLatestWorkerResponse
} from './workers/aiWorkerProtocol';

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

type BoardSeat = 'bottom' | 'top';

const boardSeatLabels: Record<BoardSeat, string> = {
  bottom: '아래',
  top: '위'
};

const difficultyLabels: Record<Difficulty, string> = {
  easy: '초급',
  normal: '중급',
  hard: '고급 실험'
};

const formationOptions: Formation[] = ['inner-elephant', 'outer-elephant', 'left-elephant', 'right-elephant'];
const difficultyOptions: Difficulty[] = ['easy', 'normal', 'hard'];

interface GameStartConfig {
  choFormation: Formation;
  hanFormation: Formation;
}

export default function App() {
  const [humanSide, setHumanSide] = useState<Side>('CHO');
  const [humanSeat, setHumanSeat] = useState<BoardSeat>('bottom');
  const [choFormation, setChoFormation] = useState<Formation>('inner-elephant');
  const [hanFormation, setHanFormation] = useState<Formation>('inner-elephant');
  const [difficulty, setDifficulty] = useState<Difficulty>('normal');
  const [useOpeningBook, setUseOpeningBook] = useState(true);
  const [game, setGame] = useState<GameState>(() => createGameState(createInitialBoard('inner-elephant', 'inner-elephant')));
  const [gameStartConfig, setGameStartConfig] = useState<GameStartConfig>({
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant'
  });
  const [selected, setSelected] = useState<Position | null>(null);
  const [redoStack, setRedoStack] = useState<Move[]>([]);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastSearch, setLastSearch] = useState('');
  const [aiError, setAiError] = useState('');
  const [analysis, setAnalysis] = useState<PositionAnalysis | null>(null);
  const [analysisError, setAnalysisError] = useState('');
  const [analysisThinking, setAnalysisThinking] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef<string | null>(null);
  const gameRef = useRef(game);

  const aiSide = otherSide(humanSide);
  gameRef.current = game;
  const legalMoves = useMemo(() => generateLegalMoves(game), [game]);
  const selectedMoves = useMemo(
    () => (selected ? legalMoves.filter((move) => samePosition(move.from, selected)) : []),
    [legalMoves, selected]
  );
  const legalTargetKeys = new Set(selectedMoves.map((move) => positionKey(move.to)));
  const isAiTurn = game.turn === aiSide && !game.winner;
  const checkSide = isInCheck(game.board, game.turn) ? game.turn : null;
  const gameStatusLabel = getGameStatusLabel(game, checkSide, aiThinking, aiError);

  const cancelAiSearch = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
    requestIdRef.current = null;
    setAiThinking(false);
  }, []);

  useEffect(() => {
    if (!isAiTurn) return;
    workerRef.current?.terminate();

    const request = createAiSearchRequest(game, difficulty, {
      useOpeningBook,
      openingBook: useOpeningBook ? builtInOpeningBook : undefined,
      openingBookContext: {
        choFormation: gameStartConfig.choFormation,
        hanFormation: gameStartConfig.hanFormation,
        minPlayCount: 2,
        maxMoves: 5
      },
      maxBookPly: 16,
      maxCandidates: 5
    });
    const worker = new Worker(new URL('./workers/aiWorker.ts', import.meta.url), { type: 'module' });
    let searchStarted = false;
    workerRef.current = worker;
    requestIdRef.current = request.requestId;
    setAiThinking(true);
    setAiError('');

    function finishWorker() {
      worker.terminate();
      if (workerRef.current === worker) {
        workerRef.current = null;
        requestIdRef.current = null;
        setAiThinking(false);
      }
    }

    worker.onmessage = (event: MessageEvent<unknown>) => {
      const response = event.data;
      if (!isAiWorkerResponse(response)) {
        finishWorker();
        setAiError('AI 오류: worker 응답 형식이 올바르지 않습니다.');
        console.error('Invalid AI worker response', response);
        return;
      }

      if (!isLatestWorkerResponse(response, requestIdRef.current)) {
        worker.terminate();
        return;
      }

      if (response.type === 'error') {
        finishWorker();
        setAiError(`AI 오류: ${response.message}`);
        console.error(response.stack ?? response.message);
        return;
      }

      const move = response.result.move;
      if (!move) {
        finishWorker();
        setLastSearch(formatSearchSummary(response.result));
        setSelected(null);
        return;
      }

      const current = gameRef.current;
      if (current.turn !== aiSide || current.winner) {
        finishWorker();
        console.warn('Ignored stale AI result for a non-AI turn.');
        return;
      }

      if (!isLegalMove(current, move)) {
        finishWorker();
        setAiError('AI 오류: 현재 포지션에서 불법 수가 반환되었습니다.');
        console.warn('Ignored illegal AI move.', move);
        return;
      }

      setGame(applyMove(current, move, true));
      setRedoStack([]);
      setLastSearch(formatSearchSummary(response.result));
      setSelected(null);
      setAnalysis(null);
      setAnalysisError('');
      finishWorker();
    };

    worker.onerror = (event) => {
      finishWorker();
      setAiError(`AI 오류: ${event.message}`);
      console.error(event.message);
    };

    const timer = window.setTimeout(() => {
      searchStarted = true;
      worker.postMessage(request);
    }, 180);
    return () => {
      window.clearTimeout(timer);
      if (!searchStarted || workerRef.current === worker) {
        finishWorker();
      }
    };
  }, [difficulty, game, isAiTurn, aiSide, useOpeningBook, gameStartConfig]);

  useEffect(() => cancelAiSearch, [cancelAiSearch]);

  function startNewGame() {
    cancelAiSearch();
    const nextConfig = { choFormation, hanFormation };
    setGameStartConfig(nextConfig);
    setGame(createGameState(createInitialBoard(nextConfig.choFormation, nextConfig.hanFormation), 'CHO'));
    setSelected(null);
    setRedoStack([]);
    setLastSearch('');
    setAiError('');
    setAnalysis(null);
    setAnalysisError('');
  }

  function initialBoardForCurrentGame() {
    return createInitialBoard(gameStartConfig.choFormation, gameStartConfig.hanFormation);
  }

  function undoOneMove() {
    if (game.history.length === 0 || aiThinking) return;
    cancelAiSearch();
    const undoneMove = game.history[game.history.length - 1];
    setGame(undoLastMove(game, initialBoardForCurrentGame(), 'CHO'));
    setRedoStack((current) => [undoneMove, ...current]);
    setSelected(null);
    setAiError('');
    setAnalysis(null);
    setAnalysisError('');
  }

  function undoUntilHumanTurn() {
    if (game.history.length === 0 || aiThinking) return;
    cancelAiSearch();
    const count = movesToUndoForHumanTurn(game, initialBoardForCurrentGame(), humanSide, 'CHO');
    const undoneMoves = game.history.slice(game.history.length - count);
    setGame(undoMoves(initialBoardForCurrentGame(), game.history, count, 'CHO'));
    setRedoStack((current) => [...undoneMoves, ...current]);
    setSelected(null);
    setAiError('');
    setAnalysis(null);
    setAnalysisError('');
  }

  function redoMove() {
    if (redoStack.length === 0 || aiThinking || isAiTurn || game.winner) return;
    cancelAiSearch();
    const [move, ...remaining] = redoStack;
    if (!canRedoMove(game, move)) {
      setRedoStack([]);
      setAiError('다시두기할 수 없는 수입니다.');
      return;
    }

    setGame(applyMove(game, move, true));
    setRedoStack(remaining);
    setSelected(null);
    setAiError('');
    setAnalysis(null);
    setAnalysisError('');
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
    setRedoStack([]);
    setSelected(null);
    setAiError('');
    setAnalysis(null);
    setAnalysisError('');
  }

  function analyzeCurrentPosition() {
    if (aiThinking || analysisThinking) return;
    setAnalysisThinking(true);
    setAnalysisError('');
    try {
      setAnalysis(
        analyzePosition(game, {
          limits: { maxDepth: 3, timeMs: 1500 },
          maxCandidates: 5,
          searchOptions: {
            useOpeningBook,
            openingBook: useOpeningBook ? builtInOpeningBook : undefined,
            openingBookContext: {
              choFormation: gameStartConfig.choFormation,
              hanFormation: gameStartConfig.hanFormation,
              minPlayCount: 2,
              maxMoves: 5
            },
            maxBookPly: 16
          }
        })
      );
    } catch (error) {
      setAnalysis(null);
      setAnalysisError(error instanceof Error ? error.message : String(error));
    } finally {
      setAnalysisThinking(false);
    }
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
            <strong>{gameStatusLabel}</strong>
            <small>{aiError || lastSearch || '탐색 기반 착수'}</small>
          </div>
        </header>

        <div className="playLayout">
          <BoardView
            game={game}
            humanSide={humanSide}
            humanSeat={humanSeat}
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
                  <button
                    key={side}
                    className={humanSide === side ? 'active' : ''}
                    onClick={() => {
                      cancelAiSearch();
                      setHumanSide(side);
                      setSelected(null);
                      setRedoStack([]);
                    }}
                  >
                    {sideLabels[side]}
                  </button>
                ))}
              </div>
            </div>

            <div className="controlGroup">
              <span className="groupLabel">사용자 위치</span>
              <div className="segmented">
                {(['bottom', 'top'] as BoardSeat[]).map((seat) => (
                  <button
                    key={seat}
                    className={humanSeat === seat ? 'active' : ''}
                    onClick={() => {
                      cancelAiSearch();
                      setHumanSeat(seat);
                      setSelected(null);
                    }}
                  >
                    {boardSeatLabels[seat]}
                  </button>
                ))}
              </div>
            </div>

            <FormationSelect
              label="초 차림"
              value={choFormation}
              onChange={(formation) => {
                cancelAiSearch();
                setChoFormation(formation);
                setSelected(null);
              }}
            />
            <FormationSelect
              label="한 차림"
              value={hanFormation}
              onChange={(formation) => {
                cancelAiSearch();
                setHanFormation(formation);
                setSelected(null);
              }}
            />
            <p className="controlHint">차림 변경은 새 대국부터 적용</p>

            <div className="controlGroup">
              <span className="groupLabel">AI 난이도</span>
              <div className="segmented">
                {difficultyOptions.map((level) => (
                  <button
                    key={level}
                    className={difficulty === level ? 'active' : ''}
                    onClick={() => {
                      cancelAiSearch();
                      setDifficulty(level);
                      setSelected(null);
                    }}
                  >
                    {difficultyLabels[level]}
                  </button>
                ))}
              </div>
            </div>

            <div className="controlGroup">
              <span className="groupLabel">오프닝북</span>
              <div className="segmented">
                <button
                  className={useOpeningBook ? 'active' : ''}
                  onClick={() => {
                    cancelAiSearch();
                    setUseOpeningBook(true);
                    setSelected(null);
                  }}
                >
                  사용
                </button>
                <button
                  className={!useOpeningBook ? 'active' : ''}
                  onClick={() => {
                    cancelAiSearch();
                    setUseOpeningBook(false);
                    setSelected(null);
                  }}
                >
                  미사용
                </button>
              </div>
            </div>

            <button className="primaryAction" onClick={startNewGame}>
              새 대국
            </button>

            <div className="gameActions">
              <button onClick={undoOneMove} disabled={game.history.length === 0 || aiThinking}>
                한 수 무르기
              </button>
              <button onClick={undoUntilHumanTurn} disabled={game.history.length === 0 || aiThinking}>
                내 차례까지 무르기
              </button>
              <button onClick={redoMove} disabled={redoStack.length === 0 || aiThinking || isAiTurn || Boolean(game.winner)}>
                다시두기
              </button>
            </div>

            <AnalysisPanel
              analysis={analysis}
              error={analysisError}
              thinking={analysisThinking}
              disabled={aiThinking}
              onAnalyze={analyzeCurrentPosition}
            />

            <MoveList history={game.history} humanSide={humanSide} />
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
  humanSeat,
  selected,
  legalTargetKeys,
  lastMove,
  onCellClick
}: {
  game: GameState;
  humanSide: Side;
  humanSeat: BoardSeat;
  selected: Position | null;
  legalTargetKeys: Set<string>;
  lastMove: Move | null;
  onCellClick: (pos: Position) => void;
}) {
  const points: Array<{ display: Position; board: Position }> = [];
  for (let displayY = 0; displayY < 10; displayY += 1) {
    for (let displayX = 0; displayX < 9; displayX += 1) {
      const display = { x: displayX, y: displayY };
      points.push({ display, board: displayToBoard(display, humanSide, humanSeat) });
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
                <span className={`piece ${piece.side.toLowerCase()} ${piece.kind.toLowerCase()}`}>
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

function MoveList({ history, humanSide }: { history: Move[]; humanSide: Side }) {
  return (
    <div className="moveList">
      <div className="moveListHeader">
        <span className="groupLabel">수순 기록</span>
        <strong>{history.length}</strong>
      </div>
      <ol>
        {history.map((move, index) => (
          <li key={`${moveKey(move)}-${index}`}>
            <span>{formatPlyNumber(index)}</span>
            <code>{formatMoveWithPiece(move, move.piece)}</code>
            <em>{(move.piece?.side ?? (index % 2 === 0 ? 'CHO' : 'HAN')) === humanSide ? '사람' : 'AI'}</em>
          </li>
        ))}
      </ol>
    </div>
  );
}

function displayToBoard(pos: Position, humanSide: Side, humanSeat: BoardSeat): Position {
  const shouldRotate = humanSeat === 'bottom' ? humanSide === 'HAN' : humanSide === 'CHO';
  if (!shouldRotate) return pos;
  return { x: 8 - pos.x, y: 9 - pos.y };
}

function AnalysisPanel({
  analysis,
  error,
  thinking,
  disabled,
  onAnalyze
}: {
  analysis: PositionAnalysis | null;
  error: string;
  thinking: boolean;
  disabled: boolean;
  onAnalyze: () => void;
}) {
  return (
    <div className="analysisPanel">
      <div className="analysisHeader">
        <span className="groupLabel">AI 분석</span>
        <button onClick={onAnalyze} disabled={disabled || thinking}>
          {thinking ? '분석 중' : '현재 포지션 분석'}
        </button>
      </div>
      {error && <p className="analysisError">{error}</p>}
      {analysis ? (
        <div className="analysisBody">
          <div className="analysisStats">
            <span>{sideLabels[analysis.turn]} 차례</span>
            <span>{analysis.source}</span>
            <span>깊이 {analysis.depth || 1}</span>
            <span>평가 {Math.round(analysis.score)}</span>
            <span>노드 {analysis.nodes}</span>
            <span>Q {analysis.qNodes}</span>
            <span>NPS {analysis.nps}</span>
          </div>
          <ol className="candidateList">
            {analysis.candidates.slice(0, 5).map((candidate, index) => (
              <li key={`${moveKey(candidate.move)}-${index}`}>
                <strong>{index + 1}</strong>
                <div>
                  <code>{formatMoveWithPiece(candidate.move, candidate.move.piece)}</code>
                  <small>
                    {candidate.source} / {candidate.riskLevel ?? 'safe'}
                    {candidate.safety?.risks[0] ? ` / ${candidate.safety.risks[0].reason}` : ''}
                    {candidate.safety
                      ? ` / raw ${Math.round(candidate.rawScore)} / penalty ${Math.round(candidate.safetyPenalty)} / swing ${candidate.safety.materialSwing}`
                      : ''}
                  </small>
                </div>
                <span>{Math.round(candidate.finalScore)}</span>
              </li>
            ))}
          </ol>
        </div>
      ) : (
        <p className="analysisEmpty">후보수와 평가값을 확인할 수 있습니다.</p>
      )}
    </div>
  );
}

function getGameStatusLabel(game: GameState, checkSide: Side | null, aiThinking: boolean, aiError: string): string {
  if (game.winner) return `외통 - ${sideLabels[game.winner]} 승리`;
  if (aiError) return 'AI 오류';
  if (checkSide) return `${sideLabels[checkSide]} 장군`;
  if (aiThinking) return 'AI 사고 중';
  return '대국 진행';
}
