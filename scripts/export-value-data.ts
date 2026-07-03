import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { exportValueSamplesFromOpeningRecords, parseOpeningRecordsCsv, valueSamplesToJsonl } from '../src/engine';

interface ExportValueCliOptions {
  input: string;
  output: string;
  summary: string;
  maxPly: number;
  limit?: number;
}

const defaultOptions: ExportValueCliOptions = {
  input: 'data/processed/janggi_clean_records.csv',
  output: 'data/ml/value_samples.jsonl',
  summary: 'data/ml/value_samples_summary.json',
  maxPly: 16
};

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    const inputPath = resolve(options.input);
    const outputPath = resolve(options.output);
    const summaryPath = resolve(options.summary);

    if (!existsSync(inputPath)) {
      throw new Error(`Missing input file: ${options.input}`);
    }

    const csv = readFileSync(inputPath, 'utf8');
    const parsedRecords = parseOpeningRecordsCsv(csv);
    const records = options.limit === undefined ? parsedRecords : parsedRecords.slice(0, options.limit);
    const exported = exportValueSamplesFromOpeningRecords(records, {
      maxPly: options.maxPly,
      includeUnknownResults: false
    });

    mkdirSync(dirname(outputPath), { recursive: true });
    mkdirSync(dirname(summaryPath), { recursive: true });
    writeFileSync(outputPath, valueSamplesToJsonl(exported.samples), 'utf8');
    writeFileSync(
      summaryPath,
      JSON.stringify(
        {
          input: options.input,
          output: options.output,
          createdAt: new Date().toISOString(),
          options: {
            maxPly: options.maxPly,
            limit: options.limit ?? null
          },
          parsedRecordCount: parsedRecords.length,
          usedRecordCount: records.length,
          ...exported.stats
        },
        null,
        2
      ),
      'utf8'
    );

    console.log('Value data export complete');
    console.log(`input: ${options.input}`);
    console.log(`records: ${records.length}`);
    console.log(`samples: ${exported.stats.sampleCount}`);
    console.log(`skippedGameCount: ${exported.stats.skippedGameCount}`);
    console.log(`illegalMoveCount: ${exported.stats.illegalMoveCount}`);
    console.log(`parseFailureCount: ${exported.stats.parseFailureCount}`);
    console.log(`output: ${options.output}`);
    console.log(`summary: ${options.summary}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

function parseArgs(args: string[]): ExportValueCliOptions {
  const options: ExportValueCliOptions = { ...defaultOptions };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--input':
        options.input = requireValue(arg, next);
        i += 1;
        break;
      case '--output':
        options.output = requireValue(arg, next);
        i += 1;
        break;
      case '--summary':
        options.summary = requireValue(arg, next);
        i += 1;
        break;
      case '--maxPly':
        options.maxPly = parseNumber(arg, next);
        i += 1;
        break;
      case '--limit':
        options.limit = parseNumber(arg, next);
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

function parseNumber(option: string, value: string | undefined): number {
  const parsed = Number(requireValue(option, value));
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`Invalid number for ${option}: ${value}`);
  return parsed;
}

main();
