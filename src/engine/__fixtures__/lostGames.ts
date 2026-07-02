import type { Formation, Side } from '../types';

export interface LostGameFixture {
  id: string;
  label: string;
  choFormation: Formation;
  hanFormation: Formation;
  humanSide: Side;
  moveText: string;
  notes: string;
  riskKind?: 'free-chariot' | 'free-cannon' | 'allows-mate';
}

export const lostGameFixtures: LostGameFixture[] = [
  {
    id: 'synthetic-free-chariot',
    label: '차를 공짜로 내주는 인공 블런더',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    humanSide: 'CHO',
    moveText: '',
    riskKind: 'free-chariot',
    notes: 'Custom tacticalSafety fixture board is built in tests until a real user game is captured.'
  },
  {
    id: 'synthetic-free-cannon',
    label: '포를 공짜로 내주는 인공 블런더',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    humanSide: 'CHO',
    moveText: '',
    riskKind: 'free-cannon',
    notes: 'Custom tacticalSafety fixture board is built in tests until a real user game is captured.'
  },
  {
    id: 'synthetic-allows-mate',
    label: '즉시 외통을 허용하는 인공 블런더',
    choFormation: 'inner-elephant',
    hanFormation: 'inner-elephant',
    humanSide: 'CHO',
    moveText: '',
    riskKind: 'allows-mate',
    notes: 'Custom tacticalSafety fixture board is built in tests until a real user game is captured.'
  }
];
