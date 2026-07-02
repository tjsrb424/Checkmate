import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import {
  arenaLeagueResultToCsv,
  arenaLeagueResultToJson,
  arenaLeagueResultToMarkdown,
  createDefaultLeaguePlayers,
  runArenaLeague
} from '../src/engine/arenaLeague';
import { allFormationPairs, defaultFormationPairs } from '../src/engine/arena';
import type { ArenaLeaguePlayer } from '../src/engine/arenaLeague';

interface CliOptions {
  gamesPerPair: number;
  maxPlies: number;
  formationMode: 'default' | 'all';
  outputDir: string;
  quick: boolean;
}

const defaultOptions: CliOptions = {
  gamesPerPair: 2,
  maxPlies: 120,
  formationMode: 'default',
  outputDir: 'data/arena',
  quick: false
};

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    const effective = options.quick
      ? { ...options, gamesPerPair: 1, maxPlies: 2, formationMode: 'default' as const }
      : options;
    const outputDir = resolve(effective.outputDir);
    const players = effective.quick ? quickPlayers(createDefaultLeaguePlayers()) : createDefaultLeaguePlayers();
    const result = runArenaLeague({
      leagueId: effective.quick ? 'arena-league-quick' : 'arena-league',
      players,
      gamesPerPair: effective.gamesPerPair,
      maxPlies: effective.maxPlies,
      formationPairs: effective.formationMode === 'all' ? allFormationPairs() : defaultFormationPairs(),
      swapSides: true,
      recordMoves: false,
      recordSearchStats: false
    });

    mkdirSync(outputDir, { recursive: true });
    writeFileSync(join(outputDir, 'arena-league.json'), arenaLeagueResultToJson(result), 'utf8');
    writeFileSync(join(outputDir, 'arena-league.csv'), arenaLeagueResultToCsv(result), 'utf8');
    writeFileSync(join(outputDir, 'arena-league.md'), arenaLeagueResultToMarkdown(result), 'utf8');

    console.log('Arena league complete');
    console.log(`players: ${result.report.players.length}`);
    console.log(`matches: ${result.matches.length}`);
    console.log(`games: ${result.report.totalGames}`);
    console.log(`ready: ${result.report.betaReadiness.ready}`);
    console.log(`warnings: ${result.report.warnings.length}`);
    console.log(`outputDir: ${effective.outputDir}`);
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
      case '--gamesPerPair':
        options.gamesPerPair = parsePositiveInt(arg, next);
        i += 1;
        break;
      case '--maxPlies':
        options.maxPlies = parsePositiveInt(arg, next);
        i += 1;
        break;
      case '--formationMode':
        options.formationMode = parseFormationMode(requireValue(arg, next));
        i += 1;
        break;
      case '--outputDir':
        options.outputDir = requireValue(arg, next);
        i += 1;
        break;
      case '--quick':
        options.quick = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

function quickPlayers(players: ArenaLeaguePlayer[]): ArenaLeaguePlayer[] {
  return players.map((player) => ({
    ...player,
    engineConfig: {
      ...player.engineConfig,
      difficulty: undefined,
      limits: { maxDepth: 1, timeMs: 200 },
      options: {
        ...player.engineConfig.options,
        enableQuiescence: false,
        enableTransposition: false
      }
    }
  }));
}

function requireValue(option: string, value: string | undefined): string {
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${option}`);
  return value;
}

function parsePositiveInt(option: string, value: string | undefined): number {
  const parsed = Number(requireValue(option, value));
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`Invalid positive integer for ${option}: ${value}`);
  return parsed;
}

function parseFormationMode(value: string): CliOptions['formationMode'] {
  if (value === 'default' || value === 'all') return value;
  throw new Error(`Invalid formationMode: ${value}`);
}

main();
