import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openingBookFromJson, summarizeOpeningBook } from '../src/engine/openingBook';
import { formatMoveWithPiece } from '../src/engine/notation';

interface InspectOptions {
  input: string;
  top: number;
}

const defaultOptions: InspectOptions = {
  input: 'data/opening-book/opening-book.json',
  top: 20
};

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = resolve(options.input);
  if (!existsSync(inputPath)) {
    console.error(`Missing input file: ${options.input}`);
    process.exitCode = 1;
    return;
  }

  const book = openingBookFromJson(readFileSync(inputPath, 'utf8'));
  const summary = summarizeOpeningBook(book, { top: options.top });
  console.log(`positionCount: ${summary.positionCount}`);
  console.log(`moveCount: ${summary.moveCount}`);
  console.log(`sourceGameCount: ${summary.sourceGameCount}`);
  console.log(`skippedGameCount: ${summary.skippedGameCount}`);
  console.log(`illegalMoveCount: ${summary.illegalMoveCount}`);
  console.log(`parseFailureCount: ${summary.parseFailureCount}`);
  console.log('top book moves:');
  for (const item of summary.topOpeningMoves) {
    console.log(
      `ply=${item.ply} turn=${item.turn} move=${formatMoveWithPiece(item.move, item.move.piece)} playCount=${item.playCount} scoreRate=${(
        item.scoreRate * 100
      ).toFixed(1)}% bookScore=${item.bookScore.toFixed(2)}`
    );
  }
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
      case '--top':
        options.top = Number(requireValue(arg, next));
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

main();
