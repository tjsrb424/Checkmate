import { applyMove, generateLegalMoves, generatePseudoMoves, isCheckmate, isInCheck } from './rules';
import { BOARD_HEIGHT, BOARD_WIDTH, getPiece, moveKey, otherSide, samePosition } from './types';
import type { Board, GameState, Move, Piece, PieceKind, Position, Side } from './types';

export type TacticalRiskLevel = 'safe' | 'warning' | 'danger' | 'losing';

export type TacticalRiskReason =
  | 'ALLOWS_IMMEDIATE_MATE'
  | 'ALLOWS_CHECK'
  | 'HANGS_GENERAL_DEFENSE'
  | 'HANGS_CHARIOT'
  | 'HANGS_CANNON'
  | 'HANGS_MAJOR_PIECE'
  | 'LOSES_MATERIAL'
  | 'OPENS_CHARIOT_FILE'
  | 'OPENS_CANNON_SCREEN'
  | 'BAD_TRADE'
  | 'LOW_KING_SAFETY';

export interface TacticalRisk {
  reason: TacticalRiskReason;
  level: TacticalRiskLevel;
  score: number;
  description: string;
}

export interface HangingPiece {
  position: Position;
  piece: Piece;
  attackedBy: Piece[];
  defendedBy: Piece[];
  estimatedLoss: number;
}

export interface MoveSafety {
  move: Move;
  riskLevel: TacticalRiskLevel;
  riskScore: number;
  materialSwing: number;
  allowsImmediateMate: boolean;
  allowsCheck: boolean;
  hangingPieces: HangingPiece[];
  risks: TacticalRisk[];
}

export interface PositionSafety {
  side: Side;
  immediateMateThreat: boolean;
  hangingPieces: HangingPiece[];
  riskScore: number;
}

export interface SafetyScanOptions {
  includeMinorPieces?: boolean;
  majorLossThreshold?: number;
}

const safetyPieceValues: Record<PieceKind, number> = {
  GENERAL: 100000,
  CHARIOT: 1300,
  CANNON: 700,
  HORSE: 500,
  ELEPHANT: 300,
  GUARD: 300,
  SOLDIER: 220
};

const majorPieces = new Set<PieceKind>(['CHARIOT', 'CANNON', 'HORSE', 'ELEPHANT']);

export function analyzeMoveSafety(state: GameState, move: Move, options: SafetyScanOptions = {}): MoveSafety {
  const side = state.turn;
  const next = applyMove(state, move, false);
  const allowsImmediateMate = detectImmediateMateThreat(next, side);
  const allowsCheck = detectImmediateCheckThreat(next, side);
  const hangingPieces = detectHangingPieces(next, side, options);
  const materialSwing = estimateMaterialSwingAfterMove(state, move);
  const risks: TacticalRisk[] = [];

  if (allowsImmediateMate) {
    risks.push({
      reason: 'ALLOWS_IMMEDIATE_MATE',
      level: 'losing',
      score: 100000,
      description: 'allows immediate mate'
    });
  }

  if (allowsCheck) {
    risks.push({
      reason: 'ALLOWS_CHECK',
      level: 'warning',
      score: 180,
      description: 'allows an immediate checking reply'
    });
  }

  for (const hanging of hangingPieces) {
    const reason = hangingReason(hanging.piece.kind);
    risks.push({
      reason,
      level: hanging.estimatedLoss >= 1000 ? 'danger' : 'warning',
      score: hangingRiskScore(hanging),
      description: `${hanging.piece.kind} can be captured`
    });
  }

  if (materialSwing < -optionsMajorLossThreshold(options)) {
    risks.push({
      reason: 'LOSES_MATERIAL',
      level: materialSwing <= -1000 ? 'danger' : 'warning',
      score: -materialSwing,
      description: `material swing ${materialSwing}`
    });
  }

  if (materialSwing < -700) {
    risks.push({
      reason: 'BAD_TRADE',
      level: materialSwing <= -1200 ? 'danger' : 'warning',
      score: Math.round(-materialSwing * 0.75),
      description: `bad immediate trade ${materialSwing}`
    });
  }

  const riskScore = risks.reduce((sum, risk) => sum + risk.score, 0);
  return {
    move: withMovePieces(state, move),
    riskLevel: riskLevelFromScore(riskScore, allowsImmediateMate),
    riskScore,
    materialSwing,
    allowsImmediateMate,
    allowsCheck,
    hangingPieces,
    risks
  };
}

export function analyzePositionSafety(state: GameState, side: Side = state.turn, options: SafetyScanOptions = {}): PositionSafety {
  const immediateMateThreat = detectImmediateMateThreat(state, side);
  const hangingPieces = detectHangingPieces(state, side, options);
  const riskScore = (immediateMateThreat ? 5000 : 0) + hangingPieces.reduce((sum, piece) => sum + hangingRiskScore(piece), 0);
  return { side, immediateMateThreat, hangingPieces, riskScore };
}

export function detectImmediateMateThreat(state: GameState, side: Side): boolean {
  const attacker = otherSide(side);
  const attackState: GameState = { board: state.board, turn: attacker, history: state.history };
  return generateLegalMoves(attackState, attacker).some((move) => {
    const next = applyMove(attackState, move, false);
    return isCheckmate(next.board, side);
  });
}

export function detectHangingPieces(state: GameState, side: Side, options: SafetyScanOptions = {}): HangingPiece[] {
  const result: HangingPiece[] = [];
  const kinds = options.includeMinorPieces ? undefined : majorPieces;

  forEachPiece(state.board, side, (position, piece) => {
    if (kinds && !kinds.has(piece.kind)) return;
    const attackers = legalAttackers(state, otherSide(side), position);
    if (attackers.length === 0) return;

    const defenders = pseudoDefenders(state.board, side, position);
    const minAttackerValue = Math.min(...attackers.map((attacker) => safetyPieceValues[attacker.kind]));
    const minDefenderValue = defenders.length > 0 ? Math.min(...defenders.map((defender) => safetyPieceValues[defender.kind])) : 0;
    const targetValue = safetyPieceValues[piece.kind];
    const estimatedLoss = defenders.length === 0 ? targetValue : Math.max(0, targetValue - Math.max(minDefenderValue, minAttackerValue));

    if (estimatedLoss >= optionsMajorLossThreshold(options) || defenders.length === 0 || minAttackerValue < targetValue) {
      result.push({
        position,
        piece,
        attackedBy: attackers,
        defendedBy: defenders,
        estimatedLoss: Math.max(estimatedLoss, Math.min(targetValue, targetValue - minAttackerValue))
      });
    }
  });

  return result.sort((a, b) => b.estimatedLoss - a.estimatedLoss);
}

export function detectMajorPieceLoss(state: GameState, side: Side, options: SafetyScanOptions = {}): TacticalRisk[] {
  return detectHangingPieces(state, side, options).map((hanging) => ({
    reason: hangingReason(hanging.piece.kind),
    level: hanging.estimatedLoss >= 1000 ? 'danger' : 'warning',
    score: hangingRiskScore(hanging),
    description: `${hanging.piece.kind} loss around ${hanging.estimatedLoss}`
  }));
}

export function estimateMaterialSwingAfterMove(state: GameState, move: Move): number {
  const movingSide = state.turn;
  const captureGain = safetyPieceValues[(getPiece(state.board, move.to) ?? move.captured)?.kind ?? 'SOLDIER'] ?? 0;
  const actualCaptureGain = getPiece(state.board, move.to) || move.captured ? captureGain : 0;
  const next = applyMove(state, move, false);
  return actualCaptureGain - estimateOpponentBestCapture(next, movingSide);
}

export function estimateOpponentBestCapture(state: GameState, sideWhoMoved: Side): number {
  const opponent = otherSide(sideWhoMoved);
  const opponentState: GameState = { board: state.board, turn: opponent, history: state.history };
  let best = 0;

  for (const move of generateLegalMoves(opponentState, opponent)) {
    const victim = getPiece(state.board, move.to) ?? move.captured;
    const attacker = getPiece(state.board, move.from) ?? move.piece;
    if (!victim || victim.side !== sideWhoMoved || !attacker) continue;
    const gain = safetyPieceValues[victim.kind] - Math.max(0, Math.round(safetyPieceValues[attacker.kind] * 0.15));
    best = Math.max(best, gain);
  }

  return best;
}

export function scoreMoveSafety(safety: MoveSafety): number {
  return safety.riskScore;
}

export function formatMoveSafety(safety: MoveSafety): string {
  if (safety.riskLevel === 'safe') return 'safe';
  const reasons = safety.risks.map((risk) => risk.reason).join(',');
  return `${safety.riskLevel}: ${reasons} swing=${safety.materialSwing}`;
}

function detectImmediateCheckThreat(state: GameState, side: Side): boolean {
  const attacker = otherSide(side);
  const attackState: GameState = { board: state.board, turn: attacker, history: state.history };
  return generateLegalMoves(attackState, attacker).some((move) => {
    const next = applyMove(attackState, move, false);
    return isInCheck(next.board, side);
  });
}

function legalAttackers(state: GameState, bySide: Side, target: Position): Piece[] {
  const attackState: GameState = { board: state.board, turn: bySide, history: state.history };
  return generateLegalMoves(attackState, bySide)
    .filter((move) => samePosition(move.to, target))
    .map((move) => getPiece(state.board, move.from) ?? move.piece)
    .filter((piece): piece is Piece => Boolean(piece));
}

function pseudoDefenders(board: Board, side: Side, target: Position): Piece[] {
  return generatePseudoMoves(board, side)
    .filter((move) => samePosition(move.to, target))
    .map((move) => getPiece(board, move.from) ?? move.piece)
    .filter((piece): piece is Piece => Boolean(piece));
}

function hangingReason(kind: PieceKind): TacticalRiskReason {
  if (kind === 'CHARIOT') return 'HANGS_CHARIOT';
  if (kind === 'CANNON') return 'HANGS_CANNON';
  if (kind === 'GENERAL' || kind === 'GUARD') return 'HANGS_GENERAL_DEFENSE';
  return 'HANGS_MAJOR_PIECE';
}

function hangingRiskScore(hanging: HangingPiece): number {
  if (hanging.piece.kind === 'CHARIOT') return Math.max(2000, hanging.estimatedLoss);
  if (hanging.piece.kind === 'CANNON') return Math.max(1200, hanging.estimatedLoss);
  if (hanging.piece.kind === 'HORSE') return Math.max(800, hanging.estimatedLoss);
  if (hanging.piece.kind === 'ELEPHANT') return Math.max(500, hanging.estimatedLoss);
  return hanging.estimatedLoss;
}

function riskLevelFromScore(riskScore: number, allowsImmediateMate: boolean): TacticalRiskLevel {
  if (allowsImmediateMate || riskScore >= 100000) return 'losing';
  if (riskScore >= 1200) return 'danger';
  if (riskScore >= 300) return 'warning';
  return 'safe';
}

function withMovePieces(state: GameState, move: Move): Move {
  const piece = move.piece ?? getPiece(state.board, move.from) ?? undefined;
  const captured = move.captured ?? getPiece(state.board, move.to) ?? undefined;
  return {
    ...move,
    ...(piece ? { piece } : {}),
    ...(captured ? { captured } : {})
  };
}

function optionsMajorLossThreshold(options: SafetyScanOptions): number {
  return options.majorLossThreshold ?? 250;
}

function forEachPiece(board: Board, side: Side, visitor: (position: Position, piece: Piece) => void): void {
  for (let y = 0; y < BOARD_HEIGHT; y += 1) {
    for (let x = 0; x < BOARD_WIDTH; x += 1) {
      const piece = board[y][x];
      if (piece?.side === side) visitor({ x, y }, piece);
    }
  }
}
