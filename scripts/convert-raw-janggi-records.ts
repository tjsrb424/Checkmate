import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { inflateRawSync } from 'node:zlib';
import { TextDecoder } from 'node:util';
import { formationToChalimString, parseFirst16Moves } from '../src/engine/openingBook';

export interface ConvertRawRecordsOptions {
  inputDir?: string;
  input?: string;
  output: string;
  summary: string;
  limit?: number;
  maxPly: number;
  source?: string;
  group: string;
  format?: string;
  strict: boolean;
  probeOnly: boolean;
}

export interface CleanTrainingRecord {
  source: string;
  group: string;
  gameIndex: string;
  choChalim: string;
  hanChalim: string;
  result: 'cho' | 'han' | 'draw' | 'unknown';
  moves16Ok: boolean;
  first16: string;
}

export interface SkippedRawRecord {
  source: string;
  gameIndex?: string;
  reason: string;
  detail?: string;
}

export interface ConversionSummary {
  createdAt: string;
  inputFiles: string[];
  output: string;
  rawGameCount: number;
  convertedRecordCount: number;
  writtenRecordCount: number;
  skippedRecordCount: number;
  moves16OkCount: number;
  moves16FailedCount: number;
  unknownResultCount: number;
  unknownFormationCount: number;
  adapterStats: Record<string, Record<string, number>>;
  resultDistribution: Record<string, number>;
  choChalimDistribution: Record<string, number>;
  hanChalimDistribution: Record<string, number>;
  first16LengthDistribution: Record<string, number>;
  sampleConvertedRecords: CleanTrainingRecord[];
  sampleSkippedRecords: SkippedRawRecord[];
  errors: string[];
}

interface RawInputFile {
  name: string;
  path: string;
  bytes: Buffer;
}

interface TextPayload {
  source: string;
  format: string;
  text: string;
}

interface RawGameCandidate {
  source: string;
  gameIndex: string;
  choChalim?: string;
  hanChalim?: string;
  result?: string;
  first16?: string;
}

interface AdapterResult {
  adapter: string;
  rawGameCount: number;
  records: CleanTrainingRecord[];
  skipped: SkippedRawRecord[];
  errors: string[];
}

interface ProbeReport {
  createdAt: string;
  fileCount: number;
  detectedFormats: Record<string, number>;
  sampleHeaders: Array<{ source: string; headers: string[] }>;
  sampleLines: Array<{ source: string; lines: string[] }>;
  unknownFiles: string[];
  recommendedAdapter: string;
}

const defaultOptions: ConvertRawRecordsOptions = {
  inputDir: 'data/raw/janggi-records',
  output: 'data/processed/janggi_clean_records.csv',
  summary: 'data/processed/janggi_clean_records.conversion.json',
  maxPly: 16,
  group: 'default',
  strict: false,
  probeOnly: false
};

const outputHeader = ['source', 'group', 'game_index', 'cho_chalim', 'han_chalim', 'result', 'moves16_ok', 'first16'];

const formationAliases: Record<string, ReturnType<typeof formationToChalimString>> = {
  innerelephant: formationToChalimString('inner-elephant'),
  'inner-elephant': formationToChalimString('inner-elephant'),
  outerelephant: formationToChalimString('outer-elephant'),
  'outer-elephant': formationToChalimString('outer-elephant'),
  lefteelephant: formationToChalimString('left-elephant'),
  leftephant: formationToChalimString('left-elephant'),
  leftelphant: formationToChalimString('left-elephant'),
  left: formationToChalimString('left-elephant'),
  'left-elephant': formationToChalimString('left-elephant'),
  right: formationToChalimString('right-elephant'),
  rightelephant: formationToChalimString('right-elephant'),
  'right-elephant': formationToChalimString('right-elephant')
};

const koreanFormationAliases: Record<string, ReturnType<typeof formationToChalimString>> = {
  마상상마: formationToChalimString('inner-elephant'),
  상마마상: formationToChalimString('outer-elephant'),
  상마상마: formationToChalimString('left-elephant'),
  마상마상: formationToChalimString('right-elephant')
};

export function convertRawJanggiRecords(options: Partial<ConvertRawRecordsOptions> = {}): ConversionSummary {
  const resolvedOptions = { ...defaultOptions, ...options };
  const inputFiles = collectInputFiles(resolvedOptions);
  const payloads = inputFiles.flatMap((file) => expandPayload(file, resolvedOptions.format));
  const probe = createProbeReport(payloads);
  if (resolvedOptions.probeOnly) {
    const probePath = resolve('data/processed/raw_records_probe_report.json');
    mkdirSync(dirname(probePath), { recursive: true });
    writeFileSync(probePath, JSON.stringify(probe, null, 2), 'utf8');
  }

  const adapterResults = payloads.map((payload) => convertPayload(payload, resolvedOptions));
  const summary = summarizeConversion(inputFiles, adapterResults, resolvedOptions);

  if (!resolvedOptions.probeOnly) {
    writeCleanCsv(resolvedOptions.output, adapterResults.flatMap((result) => result.records));
  }

  mkdirSync(dirname(resolve(resolvedOptions.summary)), { recursive: true });
  writeFileSync(resolve(resolvedOptions.summary), JSON.stringify(summary, null, 2), 'utf8');
  return summary;
}

export function cleanTrainingRecordsToCsv(records: CleanTrainingRecord[]): string {
  return [
    outputHeader.join(','),
    ...records.map((record) =>
      [
        record.source,
        record.group,
        record.gameIndex,
        record.choChalim,
        record.hanChalim,
        record.result,
        String(record.moves16Ok),
        record.first16
      ]
        .map(csvEscape)
        .join(',')
    )
  ].join('\n') + '\n';
}

export function normalizeRawResult(value: string | undefined): CleanTrainingRecord['result'] {
  const normalized = normalizeToken(value ?? '');
  if (['cho', 'red', '10', '1-0', '초', '초승', '초완승', '초승리'].includes(normalized)) return 'cho';
  if (['han', 'blue', '01', '0-1', '한', '한승', '한완승', '한승리'].includes(normalized)) return 'han';
  if (['draw', '12-12', '1/2-1/2', '무', '무승부', '비김'].includes(normalized)) return 'draw';
  if (normalized.includes('초') && normalized.includes('승')) return 'cho';
  if (normalized.includes('한') && normalized.includes('승')) return 'han';
  if (normalized.includes('무승부')) return 'draw';
  return 'unknown';
}

export function normalizeRawChalim(value: string | undefined): string | null {
  if (!value) return null;
  const compact = normalizeToken(value);
  if (koreanFormationAliases[compact]) return koreanFormationAliases[compact];
  if (formationAliases[compact]) return formationAliases[compact];
  for (const formation of ['inner-elephant', 'outer-elephant', 'left-elephant', 'right-elephant'] as const) {
    const chalim = formationToChalimString(formation);
    if (normalizeToken(chalim) === compact) return chalim;
  }
  return null;
}

function collectInputFiles(options: ConvertRawRecordsOptions): RawInputFile[] {
  const paths = options.input ? [resolve(options.input)] : listFiles(resolve(options.inputDir ?? defaultOptions.inputDir));
  const files = paths.map((path) => ({
    name: basename(path),
    path,
    bytes: readFileSync(path)
  }));
  if (files.length === 0) throw new Error('No raw record files found');
  return files;
}

function listFiles(root: string): string[] {
  if (!existsSync(root)) return [];
  const stat = statSync(root);
  if (stat.isFile()) return [root];
  return readdirSync(root)
    .flatMap((entry) => {
      const fullPath = join(root, entry);
      const entryStat = statSync(fullPath);
      if (entryStat.isDirectory()) return listFiles(fullPath);
      return isSupportedInputPath(fullPath) ? [fullPath] : [];
    })
    .sort((a, b) => a.localeCompare(b));
}

function isSupportedInputPath(path: string): boolean {
  return ['.zip', '.gib', '.csv', '.tsv', '.json', '.jsonl', '.txt'].includes(extname(path).toLowerCase());
}

function expandPayload(file: RawInputFile, forcedFormat?: string): TextPayload[] {
  const extension = normalizeExtension(forcedFormat ?? extname(file.name));
  if (extension === 'zip') {
    return readZipEntries(file.bytes).flatMap((entry) => {
      const source = `${file.name}:${entry.name}`;
      const entryExtension = normalizeExtension(extname(entry.name));
      if (!isTextLike(entryExtension)) {
        return [{ source, format: entryExtension || 'unknown', text: '' }];
      }
      return [{ source, format: entryExtension, text: decodeText(entry.bytes) }];
    });
  }

  return [{ source: file.name, format: extension || 'unknown', text: isTextLike(extension) ? decodeText(file.bytes) : '' }];
}

function convertPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  if (!payload.text) {
    return {
      adapter: 'unknown',
      rawGameCount: 0,
      records: [],
      skipped: [{ source: payload.source, reason: 'unsupported-binary-format', detail: payload.format }],
      errors: [`Unsupported binary format: ${payload.source}`]
    };
  }

  if (payload.format === 'csv' || payload.format === 'tsv') return convertDelimitedPayload(payload, options);
  if (payload.format === 'json') return convertJsonPayload(payload, options);
  if (payload.format === 'jsonl') return convertJsonlPayload(payload, options);
  if (payload.format === 'gib' || looksLikeGib(payload.text)) return convertGibPayload(payload, options);
  return convertTextPayload(payload, options);
}

function convertDelimitedPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  const delimiter = payload.format === 'tsv' ? '\t' : ',';
  const rows = parseDelimitedRows(payload.text, delimiter).filter((row) => row.some((cell) => cell.trim()));
  const [header, ...body] = rows;
  if (!header) return emptyAdapterResult('delimited', payload, 'missing-header');
  const indexes = new Map(header.map((cell, index) => [normalizeHeader(cell), index]));
  const hasFirst16 = firstPresent(indexes, ['first16', 'first_16', 'opening16', 'moves16']);
  if (!hasFirst16) return emptyAdapterResult('delimited', payload, 'missing-first16-column');

  return convertCandidates(
    'delimited',
    body.map((row, index) => ({
      source: sourceOverride(payload.source, options),
      gameIndex: getCellAny(row, indexes, ['game_index', 'id', 'gameid']) || String(index + 1),
      choChalim: getCellAny(row, indexes, ['cho_chalim', 'chochalim', 'cho_formation', 'choformation']),
      hanChalim: getCellAny(row, indexes, ['han_chalim', 'hanchalim', 'han_formation', 'hanformation']),
      result: getCellAny(row, indexes, ['result', 'winner', 'outcome', 'raw_result']),
      first16: getCellAny(row, indexes, ['first16', 'first_16', 'opening16', 'moves16'])
    })),
    options
  );
}

function convertJsonPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  try {
    const parsed = JSON.parse(payload.text) as unknown;
    const rawRecords = Array.isArray(parsed) ? parsed : Array.isArray((parsed as { records?: unknown }).records) ? (parsed as { records: unknown[] }).records : [parsed];
    return convertCandidates(
      'json',
      rawRecords.map((record, index) => objectToCandidate(record, payload, options, index)),
      options
    );
  } catch (error) {
    return emptyAdapterResult('json', payload, 'invalid-json', error instanceof Error ? error.message : String(error));
  }
}

function convertJsonlPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  const records: RawGameCandidate[] = [];
  const skipped: SkippedRawRecord[] = [];
  for (const [index, line] of payload.text.split(/\r?\n/).entries()) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      records.push(objectToCandidate(JSON.parse(trimmed), payload, options, index));
    } catch (error) {
      skipped.push({
        source: payload.source,
        gameIndex: String(index + 1),
        reason: 'invalid-jsonl',
        detail: error instanceof Error ? error.message : String(error)
      });
    }
  }
  const result = convertCandidates('jsonl', records, options);
  return { ...result, skipped: [...result.skipped, ...skipped] };
}

function convertGibPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  const chunks = splitGibGames(payload.text);
  return convertCandidates(
    'gib',
    chunks.map((chunk, index) => {
      const tags = parseGibTags(chunk);
      return {
        source: sourceOverride(payload.source, options),
        gameIndex: String(index + 1),
        choChalim: tags.get('초차림'),
        hanChalim: tags.get('한차림'),
        result: tags.get('대국결과'),
        first16: extractGibFirstMoves(chunk, options.maxPly)
      };
    }),
    options
  );
}

function convertTextPayload(payload: TextPayload, options: ConvertRawRecordsOptions): AdapterResult {
  if (looksLikeGib(payload.text)) return convertGibPayload(payload, options);
  return emptyAdapterResult('text', payload, 'unknown-text-format');
}

function convertCandidates(adapter: string, candidates: RawGameCandidate[], options: ConvertRawRecordsOptions): AdapterResult {
  const records: CleanTrainingRecord[] = [];
  const skipped: SkippedRawRecord[] = [];
  const errors: string[] = [];
  const limited = options.limit === undefined ? candidates : candidates.slice(0, options.limit);

  for (const candidate of limited) {
    const choChalim = normalizeRawChalim(candidate.choChalim);
    const hanChalim = normalizeRawChalim(candidate.hanChalim);
    const first16 = normalizeFirst16(candidate.first16 ?? '', options.maxPly);
    if (!choChalim || !hanChalim) {
      skipped.push({
        source: candidate.source,
        gameIndex: candidate.gameIndex,
        reason: 'unknown-formation',
        detail: `cho=${candidate.choChalim ?? ''} han=${candidate.hanChalim ?? ''}`
      });
      continue;
    }
    if (!first16) {
      skipped.push({ source: candidate.source, gameIndex: candidate.gameIndex, reason: 'missing-first16' });
      continue;
    }

    const moves16Ok = parseFirst16Moves(first16).length > 0;
    const record: CleanTrainingRecord = {
      source: candidate.source,
      group: options.group,
      gameIndex: candidate.gameIndex,
      choChalim,
      hanChalim,
      result: normalizeRawResult(candidate.result),
      moves16Ok,
      first16
    };

    if (moves16Ok) records.push(record);
    else {
      skipped.push({ source: candidate.source, gameIndex: candidate.gameIndex, reason: 'first16-parse-failed', detail: first16.slice(0, 120) });
    }
  }

  if (options.strict && skipped.length > 0) {
    errors.push(`${adapter} skipped ${skipped.length} raw records`);
  }

  return { adapter, rawGameCount: candidates.length, records, skipped, errors };
}

function objectToCandidate(value: unknown, payload: TextPayload, options: ConvertRawRecordsOptions, index: number): RawGameCandidate {
  const record = typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};
  return {
    source: sourceOverride(String(record.source ?? payload.source), options),
    gameIndex: String(record.game_index ?? record.gameIndex ?? record.id ?? index + 1),
    choChalim: stringValue(record.cho_chalim ?? record.choChalim ?? record.cho_formation ?? record.choFormation),
    hanChalim: stringValue(record.han_chalim ?? record.hanChalim ?? record.han_formation ?? record.hanFormation),
    result: stringValue(record.result ?? record.winner ?? record.outcome ?? record.raw_result),
    first16: stringValue(record.first16 ?? record.first_16 ?? record.opening16 ?? record.moves16)
  };
}

function summarizeConversion(inputFiles: RawInputFile[], adapterResults: AdapterResult[], options: ConvertRawRecordsOptions): ConversionSummary {
  const records = adapterResults.flatMap((result) => result.records);
  const skipped = adapterResults.flatMap((result) => result.skipped);
  const adapterStats: Record<string, Record<string, number>> = {};
  for (const result of adapterResults) {
    adapterStats[result.adapter] ??= { rawGameCount: 0, writtenRecordCount: 0, skippedRecordCount: 0, payloadCount: 0 };
    adapterStats[result.adapter].rawGameCount += result.rawGameCount;
    adapterStats[result.adapter].writtenRecordCount += result.records.length;
    adapterStats[result.adapter].skippedRecordCount += result.skipped.length;
    adapterStats[result.adapter].payloadCount += 1;
  }

  const summary: ConversionSummary = {
    createdAt: new Date().toISOString(),
    inputFiles: inputFiles.map((file) => relative(process.cwd(), file.path).replace(/\\/g, '/')),
    output: options.output,
    rawGameCount: adapterResults.reduce((sum, result) => sum + result.rawGameCount, 0),
    convertedRecordCount: records.length,
    writtenRecordCount: records.length,
    skippedRecordCount: skipped.length,
    moves16OkCount: records.filter((record) => record.moves16Ok).length,
    moves16FailedCount: skipped.filter((record) => record.reason.includes('first16') || record.reason.includes('missing-first16')).length,
    unknownResultCount: records.filter((record) => record.result === 'unknown').length,
    unknownFormationCount: skipped.filter((record) => record.reason === 'unknown-formation').length,
    adapterStats,
    resultDistribution: countBy(records, (record) => record.result),
    choChalimDistribution: countBy(records, (record) => record.choChalim),
    hanChalimDistribution: countBy(records, (record) => record.hanChalim),
    first16LengthDistribution: countBy(records, (record) => String(parseFirst16Moves(record.first16).length)),
    sampleConvertedRecords: records.slice(0, 5),
    sampleSkippedRecords: skipped.slice(0, 10),
    errors: adapterResults.flatMap((result) => result.errors)
  };

  if (!options.probeOnly && records.length === 0) summary.errors.push('No records were converted');
  if (options.strict && summary.errors.length === 0 && skipped.length > 0) summary.errors.push(`${skipped.length} records were skipped in strict mode`);
  return summary;
}

function createProbeReport(payloads: TextPayload[]): ProbeReport {
  const sampleHeaders: ProbeReport['sampleHeaders'] = [];
  const sampleLines: ProbeReport['sampleLines'] = [];
  const unknownFiles: string[] = [];
  const detectedFormats: Record<string, number> = {};

  for (const payload of payloads) {
    const format = payload.format || 'unknown';
    detectedFormats[format] = (detectedFormats[format] ?? 0) + 1;
    const lines = payload.text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).slice(0, 8);
    sampleLines.push({ source: payload.source, lines });
    if (payload.format === 'csv' || payload.format === 'tsv') {
      const delimiter = payload.format === 'tsv' ? '\t' : ',';
      sampleHeaders.push({ source: payload.source, headers: parseDelimitedRows(payload.text, delimiter)[0] ?? [] });
    }
    if (!payload.text || (!looksLikeGib(payload.text) && !['csv', 'tsv', 'json', 'jsonl', 'gib', 'txt'].includes(payload.format))) {
      unknownFiles.push(payload.source);
    }
  }

  return {
    createdAt: new Date().toISOString(),
    fileCount: payloads.length,
    detectedFormats,
    sampleHeaders: sampleHeaders.slice(0, 20),
    sampleLines: sampleLines.slice(0, 20),
    unknownFiles: unknownFiles.slice(0, 50),
    recommendedAdapter: unknownFiles.length > 0 ? 'Add a source-specific adapter for unknown files.' : 'Existing adapters can probe these files.'
  };
}

function writeCleanCsv(output: string, records: CleanTrainingRecord[]): void {
  const outputPath = resolve(output);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, cleanTrainingRecordsToCsv(records), 'utf8');
}

function readZipEntries(buffer: Buffer): Array<{ name: string; bytes: Buffer }> {
  const eocdOffset = findEndOfCentralDirectory(buffer);
  if (eocdOffset < 0) throw new Error('Invalid ZIP: missing end of central directory');
  const entryCount = buffer.readUInt16LE(eocdOffset + 10);
  let centralOffset = buffer.readUInt32LE(eocdOffset + 16);
  const entries: Array<{ name: string; bytes: Buffer }> = [];

  for (let i = 0; i < entryCount; i += 1) {
    if (buffer.readUInt32LE(centralOffset) !== 0x02014b50) throw new Error('Invalid ZIP: bad central directory header');
    const method = buffer.readUInt16LE(centralOffset + 10);
    const compressedSize = buffer.readUInt32LE(centralOffset + 20);
    const fileNameLength = buffer.readUInt16LE(centralOffset + 28);
    const extraLength = buffer.readUInt16LE(centralOffset + 30);
    const commentLength = buffer.readUInt16LE(centralOffset + 32);
    const localOffset = buffer.readUInt32LE(centralOffset + 42);
    const name = decodeZipFileName(buffer.subarray(centralOffset + 46, centralOffset + 46 + fileNameLength));
    centralOffset += 46 + fileNameLength + extraLength + commentLength;
    if (name.endsWith('/')) continue;

    if (buffer.readUInt32LE(localOffset) !== 0x04034b50) throw new Error(`Invalid ZIP: bad local header for ${name}`);
    const localNameLength = buffer.readUInt16LE(localOffset + 26);
    const localExtraLength = buffer.readUInt16LE(localOffset + 28);
    const dataOffset = localOffset + 30 + localNameLength + localExtraLength;
    const compressed = buffer.subarray(dataOffset, dataOffset + compressedSize);
    if (method === 0) entries.push({ name, bytes: Buffer.from(compressed) });
    else if (method === 8) entries.push({ name, bytes: inflateRawSync(compressed) });
  }

  return entries;
}

function findEndOfCentralDirectory(buffer: Buffer): number {
  for (let offset = buffer.length - 22; offset >= 0; offset -= 1) {
    if (buffer.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  return -1;
}

function decodeZipFileName(bytes: Buffer): string {
  const decoded = decodeText(bytes);
  return decoded.replace(/\0/g, '');
}

function decodeText(bytes: Buffer): string {
  const utf8 = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  const replacementCount = (utf8.match(/\uFFFD/g) ?? []).length;
  if (replacementCount <= 1) return utf8.replace(/^\uFEFF/, '');
  return new TextDecoder('euc-kr', { fatal: false }).decode(bytes).replace(/^\uFEFF/, '');
}

function splitGibGames(text: string): string[] {
  const cleaned = stripBraceComments(text);
  const matches = Array.from(cleaned.matchAll(/^\[대회명\s+"[^"]*"\]/gm));
  if (matches.length === 0) return [cleaned];
  return matches.map((match, index) => {
    const start = match.index ?? 0;
    const end = matches[index + 1]?.index ?? cleaned.length;
    return cleaned.slice(start, end);
  });
}

function stripBraceComments(text: string): string {
  let output = '';
  let depth = 0;
  for (const char of text) {
    if (char === '{') {
      depth += 1;
      output += ' ';
      continue;
    }
    if (char === '}') {
      depth = Math.max(0, depth - 1);
      output += ' ';
      continue;
    }
    output += depth > 0 ? (char === '\n' || char === '\r' ? char : ' ') : char;
  }
  return output;
}

function parseGibTags(text: string): Map<string, string> {
  const tags = new Map<string, string>();
  for (const match of text.matchAll(/^\[([^\s"\]]+)\s+"([^"]*)"\]/gm)) {
    tags.set(match[1], match[2]);
  }
  return tags;
}

function extractGibFirstMoves(text: string, maxPly: number): string {
  const cleaned = stripBraceComments(text);
  const tokens: string[] = [];
  for (const match of cleaned.matchAll(/(\d+)\.\s*([0-9][1-9][^\s\[\]{}]*?[0-9][1-9][^\s\[\]{}]*)/g)) {
    const ply = Number(match[1]);
    if (!Number.isInteger(ply) || ply <= 0) continue;
    tokens.push(`${ply}.${convertGibMoveBody(match[2].trim())}`);
    if (tokens.length >= maxPly) break;
  }
  return tokens.join(' ');
}

function convertGibMoveBody(body: string): string {
  return body.replace(/([0-9])([1-9])/g, (_coord, rowText: string, columnText: string) => {
    const row = Number(rowText);
    const column = Number(columnText);
    const x = column - 1;
    const y = row === 0 ? 9 : row - 1;
    if (x < 0 || x > 8 || y < 0 || y > 9) return `${rowText}${columnText}`;
    return `${x}${y}`;
  });
}

function normalizeFirst16(value: string, maxPly: number): string {
  const compacted = value
    .replace(/(\d+)\.\s+/g, '$1.')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean)
    .slice(0, maxPly);
  return compacted.join(' ');
}

function looksLikeGib(text: string): boolean {
  return /^\[대회명\s+"/m.test(text) || /^\[초차림\s+"/m.test(text) || /\d+\.\s*[0-9][1-9][^\s]*[0-9][1-9]/.test(text);
}

function emptyAdapterResult(adapter: string, payload: TextPayload, reason: string, detail?: string): AdapterResult {
  return {
    adapter,
    rawGameCount: 0,
    records: [],
    skipped: [{ source: payload.source, reason, detail }],
    errors: []
  };
}

function parseDelimitedRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      cell += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === delimiter && !inQuotes) {
      row.push(cell);
      cell = '';
      continue;
    }
    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
      continue;
    }
    cell += char;
  }

  row.push(cell);
  rows.push(row);
  return rows;
}

function getCellAny(row: string[], indexes: Map<string, number>, names: string[]): string {
  for (const name of names) {
    const index = indexes.get(normalizeHeader(name));
    const value = index === undefined ? '' : row[index]?.trim() ?? '';
    if (value) return value;
  }
  return '';
}

function firstPresent(indexes: Map<string, number>, names: string[]): boolean {
  return names.some((name) => indexes.has(normalizeHeader(name)));
}

function normalizeHeader(value: string): string {
  return value.trim().replace(/[\s-]/g, '_').toLowerCase();
}

function normalizeToken(value: string): string {
  return value.trim().toLowerCase().replace(/[\s"'[\],_()-]/g, '');
}

function normalizeExtension(extension: string): string {
  return extension.replace(/^\./, '').toLowerCase();
}

function isTextLike(extension: string): boolean {
  return ['csv', 'tsv', 'json', 'jsonl', 'txt', 'gib', ''].includes(extension);
}

function csvEscape(value: string): string {
  if (!/[",\r\n]/.test(value)) return value;
  return `"${value.replace(/"/g, '""')}"`;
}

function stringValue(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  return String(value);
}

function sourceOverride(source: string, options: ConvertRawRecordsOptions): string {
  return options.source ?? source;
}

function countBy<T>(items: T[], keyFn: (item: T) => string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const item of items) {
    const key = keyFn(item) || 'unknown';
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function parseArgs(args: string[]): ConvertRawRecordsOptions {
  const options: ConvertRawRecordsOptions = { ...defaultOptions };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    switch (arg) {
      case '--inputDir':
        options.inputDir = requireValue(arg, next);
        options.input = undefined;
        i += 1;
        break;
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
      case '--limit':
        options.limit = parseInteger(arg, next);
        i += 1;
        break;
      case '--maxPly':
        options.maxPly = parseInteger(arg, next);
        i += 1;
        break;
      case '--source':
        options.source = requireValue(arg, next);
        i += 1;
        break;
      case '--group':
        options.group = requireValue(arg, next);
        i += 1;
        break;
      case '--format':
        options.format = requireValue(arg, next);
        i += 1;
        break;
      case '--strict':
        options.strict = true;
        break;
      case '--probeOnly':
        options.probeOnly = true;
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

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    const summary = convertRawJanggiRecords(options);
    console.log('Raw Janggi record conversion complete');
    console.log(`inputFiles: ${summary.inputFiles.length}`);
    console.log(`rawGames: ${summary.rawGameCount}`);
    console.log(`writtenRecords: ${summary.writtenRecordCount}`);
    console.log(`skippedRecords: ${summary.skippedRecordCount}`);
    console.log(`unknownResults: ${summary.unknownResultCount}`);
    console.log(`output: ${summary.output}`);
    console.log(`summary: ${options.summary}`);
    if (summary.errors.length > 0) {
      console.log(`errors: ${summary.errors.join('; ')}`);
      process.exitCode = 1;
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
