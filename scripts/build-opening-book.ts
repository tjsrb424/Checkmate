import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import {
  buildOpeningBookFromRecords,
  openingBookFromJson,
  openingBookToJson,
  parseOpeningRecordsCsv,
  summarizeOpeningBook
} from '../src/engine/openingBook';
import type { OpeningBook, OpeningBookBuildOptions } from '../src/engine/openingBook';

interface BuildCliOptions extends Required<OpeningBookBuildOptions> {
  input: string;
  output: string;
  summary: string;
  pretty: boolean;
}

const defaultOptions: BuildCliOptions = {
  input: 'data/processed/janggi_clean_records.csv',
  output: 'data/opening-book/opening-book.json',
  summary: 'data/opening-book/opening-book-summary.json',
  maxPly: 16,
  minPlayCount: 2,
  maxMovesPerPosition: 5,
  includeUnknownResults: false,
  pretty: true
};

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = resolve(options.input);
  const outputPath = resolve(options.output);
  const summaryPath = resolve(options.summary);

  if (!existsSync(inputPath)) {
    console.error(`Missing input file: ${options.input}`);
    process.exitCode = 1;
    return;
  }

  try {
    const csv = readFileSync(inputPath, 'utf8');
    const records = parseOpeningRecordsCsv(csv);
    const book = buildOpeningBookFromRecords(records, {
      maxPly: options.maxPly,
      minPlayCount: options.minPlayCount,
      maxMovesPerPosition: options.maxMovesPerPosition,
      includeUnknownResults: options.includeUnknownResults
    });
    validateOpeningBook(book);

    mkdirSync(dirname(outputPath), { recursive: true });
    mkdirSync(dirname(summaryPath), { recursive: true });

    writeFileSync(outputPath, options.pretty ? openingBookToJson(book) : JSON.stringify(book), 'utf8');
    const summary = {
      input: options.input,
      output: options.output,
      createdAt: book.createdAt,
      options: {
        maxPly: options.maxPly,
        minPlayCount: options.minPlayCount,
        maxMovesPerPosition: options.maxMovesPerPosition,
        includeUnknownResults: options.includeUnknownResults
      },
      recordCount: records.length,
      ...summarizeOpeningBook(book, { top: 20 })
    };
    writeFileSync(summaryPath, options.pretty ? JSON.stringify(summary, null, 2) : JSON.stringify(summary), 'utf8');

    const restored = openingBookFromJson(readFileSync(outputPath, 'utf8'));
    validateOpeningBook(restored);
    printBuildSummary(options, records.length, book);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

function parseArgs(args: string[]): BuildCliOptions {
  const options = { ...defaultOptions };
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
      case '--minPlayCount':
        options.minPlayCount = parseNumber(arg, next);
        i += 1;
        break;
      case '--maxMovesPerPosition':
        options.maxMovesPerPosition = parseNumber(arg, next);
        i += 1;
        break;
      case '--includeUnknownResults':
        options.includeUnknownResults = true;
        break;
      case '--pretty':
        options.pretty = true;
        break;
      case '--compact':
        options.pretty = false;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

function validateOpeningBook(book: OpeningBook): void {
  if (!book.positions || typeof book.positions !== 'object') throw new Error('Invalid opening book: missing positions');
  for (const position of Object.values(book.positions)) {
    if (!Array.isArray(position.moves)) throw new Error(`Invalid opening book: moves is not an array for ${position.positionKey}`);
    for (const move of position.moves) {
      if (!move.move?.from || !move.move?.to) throw new Error(`Invalid opening book move at ${position.positionKey}`);
    }
  }
}

function printBuildSummary(options: BuildCliOptions, recordCount: number, book: OpeningBook): void {
  console.log('Opening book build complete');
  console.log(`input: ${options.input}`);
  console.log(`records: ${recordCount}`);
  console.log(`sourceGameCount: ${book.sourceGameCount}`);
  console.log(`skippedGameCount: ${book.skippedGameCount}`);
  console.log(`positionCount: ${book.positionCount}`);
  console.log(`moveCount: ${book.moveCount}`);
  console.log(`illegalMoveCount: ${book.illegalMoveCount}`);
  console.log(`parseFailureCount: ${book.parseFailureCount}`);
  console.log(`output: ${options.output}`);
  console.log(`summary: ${options.summary}`);
}

function requireValue(option: string, value: string | undefined): string {
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${option}`);
  return value;
}

function parseNumber(option: string, value: string | undefined): number {
  const parsed = Number(requireValue(option, value));
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`Invalid number for ${option}: ${value}`);
  return parsed;
}

main();
