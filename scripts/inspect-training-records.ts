import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseFirst16Moves, parseOpeningRecordsCsv } from '../src/engine';

interface InspectOptions {
  input: string;
  limit: number;
}

const defaultOptions: InspectOptions = {
  input: 'data/processed/janggi_clean_records.csv',
  limit: 10
};

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    const inputPath = resolve(options.input);
    if (!existsSync(inputPath)) throw new Error(`Missing input file: ${options.input}`);

    const records = parseOpeningRecordsCsv(readFileSync(inputPath, 'utf8'));
    console.log(`input: ${options.input}`);
    console.log(`records: ${records.length}`);
    printDistribution('source', records.map((record) => record.source ?? 'unknown'));
    printDistribution('group', records.map((record) => record.group ?? 'unknown'));
    printDistribution('choChalim', records.map((record) => record.choChalim || 'missing'));
    printDistribution('hanChalim', records.map((record) => record.hanChalim || 'missing'));
    printDistribution('result', records.map((record) => record.result));
    printDistribution(
      'first16Length',
      records.map((record) => String(parseFirst16Moves(record.first16).length))
    );
    console.log('first rows:');
    for (const record of records.slice(0, options.limit)) {
      console.log(
        JSON.stringify({
          id: record.id,
          source: record.source,
          group: record.group,
          result: record.result,
          choChalim: record.choChalim,
          hanChalim: record.hanChalim,
          moves16Ok: record.moves16Ok,
          first16Length: parseFirst16Moves(record.first16).length,
          first16: record.first16
        })
      );
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

function printDistribution(label: string, values: string[]): void {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  const top = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 12);
  console.log(`${label}: ${top.map(([value, count]) => `${value}=${count}`).join(', ')}`);
}

function parseArgs(args: string[]): InspectOptions {
  const options = { ...defaultOptions };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--input':
        options.input = requireValue(arg, next);
        i += 1;
        break;
      case '--limit':
        options.limit = parseInteger(arg, next);
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

main();
