import { Move } from './types';

export type TranspositionFlag = 'EXACT' | 'LOWERBOUND' | 'UPPERBOUND';

export interface TranspositionEntry {
  key: string;
  depth: number;
  score: number;
  flag: TranspositionFlag;
  bestMove?: Move;
  age?: number;
}

export interface TranspositionStats {
  hits: number;
  misses: number;
  stores: number;
  overwrites: number;
}

export class TranspositionTable {
  private entries = new Map<string, TranspositionEntry>();
  private stats: TranspositionStats = {
    hits: 0,
    misses: 0,
    stores: 0,
    overwrites: 0
  };

  get(key: string): TranspositionEntry | undefined {
    const entry = this.entries.get(key);
    if (entry) {
      this.stats.hits += 1;
      return entry;
    }
    this.stats.misses += 1;
    return undefined;
  }

  peek(key: string): TranspositionEntry | undefined {
    return this.entries.get(key);
  }

  set(entry: TranspositionEntry): void {
    const existing = this.entries.get(entry.key);
    if (existing && !shouldReplace(existing, entry)) return;

    if (existing) this.stats.overwrites += 1;
    this.stats.stores += 1;
    this.entries.set(entry.key, { ...entry, bestMove: entry.bestMove ? { ...entry.bestMove } : undefined });
  }

  clear(): void {
    this.entries.clear();
    this.stats = {
      hits: 0,
      misses: 0,
      stores: 0,
      overwrites: 0
    };
  }

  size(): number {
    return this.entries.size;
  }

  getStats(): TranspositionStats {
    return { ...this.stats };
  }
}

function shouldReplace(existing: TranspositionEntry, next: TranspositionEntry): boolean {
  if (next.depth > existing.depth) return true;
  if (next.depth < existing.depth) return false;
  if (next.flag === 'EXACT' && existing.flag !== 'EXACT') return true;
  if (next.flag !== 'EXACT' && existing.flag === 'EXACT') return false;
  return true;
}
