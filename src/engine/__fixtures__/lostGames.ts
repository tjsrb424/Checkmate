import type { Formation, Side } from '../types';

export interface LostGameFixture {
  id: string;
  label: string;
  choFormation: Formation;
  hanFormation: Formation;
  humanSide: Side;
  moveText: string;
  notes: string;
}

export const lostGameFixtures: LostGameFixture[] = [];
