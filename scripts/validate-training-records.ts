import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { exportAlphaZeroSupervisedSamplesFromOpeningRecords, parseOpeningRecordsCsv } from '../src/engine';

interface ValidateOptions {
  input: string;
  summary: string;
  maxPly: number;
  limit?: number;
  strict: boolean;
  maxIllegalRate: number;
  maxUnknownRate: number;
}

interface ValidationSummary {
  input: string;
  createdAt: string;
  recordCount: number;
  validRecordCount: number;
  skippedGameCount: number;
  illegalMoveCount: number;
  parseFailureCount: number;
  unknownResultCount: number;
  unknownResultRate: number;
  illegalMoveRate: number;
  sampleCount: number;
  sampleFirstInvalidRows: Array<Record<string, unknown>>;
  sampleIllegalRows: Array<Record<string, unknown>>;
  strict: boolean;
  passed: boolean;
  errors: string[];
}

const defaultOptions: ValidateOptions = {
  input: 'data/processed/janggi_clean_records.csv',
  summary: 'data/processed/janggi_clean_records.validation.json',
  maxPly: 16,
  strict: false,
  maxIllegalRate: 0.05,
  maxUnknownRate: 0.2
};

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    const inputPath = resolve(options.input);
    if (!existsSync(inputPath)) {
      throw new Error(`Missing input file: ${options.input}`);
    }

    const csv = readFileSync(inputPath, 'utf8');
    const parsedRecords = parseOpeningRecordsCsv(csv);
    const records = options.limit === undefined ? parsedRecords : parsedRecords.slice(0, options.limit);
    const exported = exportAlphaZeroSupervisedSamplesFromOpeningRecords(records, { maxPly: options.maxPly });
    const unknownResultCount = records.filter((record) => record.result === 'unknown').length;
    const unknownResultRate = rate(unknownResultCount, records.length);
    const illegalMoveRate = rate(exported.stats.illegalMoveCount, Math.max(exported.stats.sourceGameCount, 1));
    const validRecordCount = Math.max(0, exported.stats.sourceGameCount - exported.stats.illegalMoveCount);
    const errors = validationErrors({
      recordCount: records.length,
      validRecordCount,
      parseFailureCount: exported.stats.parseFailureCount,
      illegalMoveRate,
      unknownResultRate,
      options
    });
    const summary: ValidationSummary = {
      input: options.input,
      createdAt: new Date().toISOString(),
      recordCount: records.length,
      validRecordCount,
      skippedGameCount: exported.stats.skippedGameCount,
      illegalMoveCount: exported.stats.illegalMoveCount,
      parseFailureCount: exported.stats.parseFailureCount,
      unknownResultCount,
      unknownResultRate,
      illegalMoveRate,
      sampleCount: exported.stats.sampleCount,
      sampleFirstInvalidRows: sampleInvalidRows(records),
      sampleIllegalRows: sampleLikelyIllegalRows(records),
      strict: options.strict,
      passed: errors.length === 0,
      errors
    };

    const summaryPath = resolve(options.summary);
    mkdirSync(dirname(summaryPath), { recursive: true });
    writeFileSync(summaryPath, JSON.stringify(summary, null, 2), 'utf8');

    printSummary(summary);
    if (errors.length > 0) {
      process.exitCode = 1;
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

function validationErrors(input: {
  recordCount: number;
  validRecordCount: number;
  parseFailureCount: number;
  illegalMoveRate: number;
  unknownResultRate: number;
  options: ValidateOptions;
}): string[] {
  const errors: string[] = [];
  if (input.recordCount <= 0) errors.push('recordCount must be greater than 0');
  if (input.validRecordCount <= 0) errors.push('validRecordCount must be greater than 0');
  if (!input.options.strict) return errors;
  if (input.illegalMoveRate > input.options.maxIllegalRate) {
    errors.push(`illegalMoveRate ${input.illegalMoveRate.toFixed(4)} exceeds ${input.options.maxIllegalRate}`);
  }
  if (input.unknownResultRate > input.options.maxUnknownRate) {
    errors.push(`unknownResultRate ${input.unknownResultRate.toFixed(4)} exceeds ${input.options.maxUnknownRate}`);
  }
  if (input.parseFailureCount > input.recordCount) {
    errors.push(`parseFailureCount ${input.parseFailureCount} is larger than recordCount ${input.recordCount}`);
  }
  return errors;
}

function sampleInvalidRows(records: ReturnType<typeof parseOpeningRecordsCsv>): Array<Record<string, unknown>> {
  return records
    .filter((record) => !record.first16 || !record.choChalim || !record.hanChalim || record.result === 'unknown')
    .slice(0, 5)
    .map((record) => ({
      id: record.id,
      source: record.source,
      result: record.result,
      choChalim: record.choChalim,
      hanChalim: record.hanChalim,
      first16: record.first16
    }));
}

function sampleLikelyIllegalRows(records: ReturnType<typeof parseOpeningRecordsCsv>): Array<Record<string, unknown>> {
  return records
    .filter((record) => /(^|\s)\d+\.[^ ]*88|bad-token/.test(record.first16))
    .slice(0, 5)
    .map((record) => ({ id: record.id, source: record.source, first16: record.first16 }));
}

function printSummary(summary: ValidationSummary): void {
  console.log('Training record validation complete');
  console.log(`input: ${summary.input}`);
  console.log(`records: ${summary.recordCount}`);
  console.log(`validRecords: ${summary.validRecordCount}`);
  console.log(`samples: ${summary.sampleCount}`);
  console.log(`illegalMoveCount: ${summary.illegalMoveCount}`);
  console.log(`illegalMoveRate: ${summary.illegalMoveRate.toFixed(4)}`);
  console.log(`parseFailureCount: ${summary.parseFailureCount}`);
  console.log(`unknownResultCount: ${summary.unknownResultCount}`);
  console.log(`unknownResultRate: ${summary.unknownResultRate.toFixed(4)}`);
  console.log(`passed: ${summary.passed}`);
  if (summary.errors.length > 0) {
    console.log(`errors: ${summary.errors.join('; ')}`);
  }
}

function rate(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

function parseArgs(args: string[]): ValidateOptions {
  const options: ValidateOptions = { ...defaultOptions };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--input':
        options.input = requireValue(arg, next);
        i += 1;
        break;
      case '--summary':
        options.summary = requireValue(arg, next);
        i += 1;
        break;
      case '--maxPly':
        options.maxPly = parseInteger(arg, next);
        i += 1;
        break;
      case '--limit':
        options.limit = parseInteger(arg, next);
        i += 1;
        break;
      case '--strict':
        options.strict = true;
        break;
      case '--maxIllegalRate':
        options.maxIllegalRate = parseFloatOption(arg, next);
        i += 1;
        break;
      case '--maxUnknownRate':
        options.maxUnknownRate = parseFloatOption(arg, next);
        i += 1;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

function requireValue(option: string, value: string | undefined): string {
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${option}`);
  return value;
}

function parseInteger(option: string, value: string | undefined): number {
  const parsed = Number(requireValue(option, value));
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`Invalid integer for ${option}: ${value}`);
  return parsed;
}

function parseFloatOption(option: string, value: string | undefined): number {
  const parsed = Number(requireValue(option, value));
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`Invalid number for ${option}: ${value}`);
  return parsed;
}

main();
