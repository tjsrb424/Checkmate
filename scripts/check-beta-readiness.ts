import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { evaluateBetaReadiness, formatBetaReadinessMarkdown } from '../src/engine/betaReadiness';
import type { ArenaLeagueResult } from '../src/engine/arenaLeague';

interface CliOptions {
  arena: string;
  output: string;
  json: string;
  strict: boolean;
}

const defaultOptions: CliOptions = {
  arena: 'data/arena/arena-league.json',
  output: 'data/arena/beta-readiness.md',
  json: 'data/arena/beta-readiness.json',
  strict: false
};

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (!existsSync(options.arena)) {
      throw new Error(`Missing arena report: ${options.arena}`);
    }

    const league = JSON.parse(readFileSync(resolve(options.arena), 'utf8')) as ArenaLeagueResult;
    const result = evaluateBetaReadiness(league);
    const markdown = formatBetaReadinessMarkdown(result);

    mkdirSync(dirname(resolve(options.output)), { recursive: true });
    mkdirSync(dirname(resolve(options.json)), { recursive: true });
    writeFileSync(resolve(options.output), markdown, 'utf8');
    writeFileSync(resolve(options.json), JSON.stringify(result, null, 2), 'utf8');

    console.log('Beta readiness check complete');
    console.log(`ready: ${result.ready}`);
    console.log(`score: ${(result.score * 100).toFixed(1)}%`);
    console.log(`recommendedLabel: ${result.recommendedLabel}`);
    console.log(`failedChecks: ${result.failedChecks.length}`);
    console.log(`warnings: ${result.warnings.length}`);
    console.log(`output: ${options.output}`);
    console.log(`json: ${options.json}`);

    if (options.strict && !result.ready) process.exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

function parseArgs(args: string[]): CliOptions {
  const options = { ...defaultOptions };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--arena':
        options.arena = requireValue(arg, next);
        i += 1;
        break;
      case '--output':
        options.output = requireValue(arg, next);
        i += 1;
        break;
      case '--json':
        options.json = requireValue(arg, next);
        i += 1;
        break;
      case '--strict':
        options.strict = true;
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
