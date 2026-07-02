import { buildOpeningBookFromRecords } from './openingBook';
import type { OpeningBookRecord } from './openingBook';

export const builtInOpeningBookRecords: OpeningBookRecord[] = [
  {
    id: 'seed-inner-001',
    source: 'built-in-seed',
    choChalim: '마상상마',
    hanChalim: '마상상마',
    result: 'cho',
    first16: '1.06졸05 2.03병04',
    moves16Ok: true
  },
  {
    id: 'seed-inner-002',
    source: 'built-in-seed',
    choChalim: '마상상마',
    hanChalim: '마상상마',
    result: 'cho',
    first16: '1.06졸05 2.03병04',
    moves16Ok: true
  },
  {
    id: 'seed-inner-003',
    source: 'built-in-seed',
    choChalim: '마상상마',
    hanChalim: '마상상마',
    result: 'draw',
    first16: '1.06졸05 2.03병04',
    moves16Ok: true
  }
];

export const builtInOpeningBook = buildOpeningBookFromRecords(builtInOpeningBookRecords, {
  minPlayCount: 2,
  maxMovesPerPosition: 5
});
