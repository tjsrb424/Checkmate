import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { exportAlphaZeroSupervisedSamplesFromOpeningRecords, parseOpeningRecordsCsv } from '../src/engine';
import { cleanTrainingRecordsToCsv, convertRawJanggiRecords, normalizeRawChalim, normalizeRawResult } from './convert-raw-janggi-records';

describe('raw Janggi record conversion', () => {
  it('converts compact CSV and JSON fixtures into clean training CSV', () => {
    const tempDir = mkdtempSync(join(tmpdir(), 'oetongsu-raw-records-'));
    try {
      const output = join(tempDir, 'clean.csv');
      const summaryPath = join(tempDir, 'summary.json');
      const summary = convertRawJanggiRecords({
        inputDir: 'data/fixtures/raw-records',
        output,
        summary: summaryPath,
        maxPly: 16,
        group: 'fixture',
        strict: true,
        probeOnly: false
      });

      const csv = readFileSync(output, 'utf8');
      const records = parseOpeningRecordsCsv(csv);
      const exported = exportAlphaZeroSupervisedSamplesFromOpeningRecords(records);

      expect(summary.writtenRecordCount).toBe(3);
      expect(records).toHaveLength(3);
      expect(exported.stats.sampleCount).toBeGreaterThan(0);
      expect(readFileSync(summaryPath, 'utf8')).toContain('"writtenRecordCount": 3');
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('normalizes result and chalim aliases', () => {
    expect(normalizeRawResult('초 완승')).toBe('cho');
    expect(normalizeRawResult('한 완승')).toBe('han');
    expect(normalizeRawResult('무승부')).toBe('draw');
    expect(normalizeRawChalim('inner-elephant')).toBeTruthy();
    expect(normalizeRawChalim('마상상마')).toBe(normalizeRawChalim('inner-elephant'));
  });

  it('writes the exact clean CSV header', () => {
    const csv = cleanTrainingRecordsToCsv([]);
    expect(csv.split('\n')[0]).toBe('source,group,game_index,cho_chalim,han_chalim,result,moves16_ok,first16');
  });
});
