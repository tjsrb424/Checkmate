# Processed Training Records

Place local supervised Janggi training records here as:

```txt
data/processed/janggi_clean_records.csv
```

This CSV is local training data and is ignored by git by default. Keep only schema documentation and `.gitkeep` in the repository unless a small public fixture is intentionally added under `data/fixtures/`.

## Required Header

Minimum supported columns:

```csv
source,group,game_index,cho_chalim,han_chalim,result,moves16_ok,first16
```

Supported aliases include:

- `game_index` or `id`
- `cho_chalim`, `choChalim`, or `cho_formation`
- `han_chalim`, `hanChalim`, or `han_formation`
- `result`, `winner`, `outcome`, or `raw_result`
- `first16`, `first_16`, `opening16`, or `moves16`
- `moves16_ok`, `moves16Ok`, `first16_ok`, or `opening_ok`

## Values

`first16` tokens use:

```txt
<plyNumber>.<fromX><fromY><pieceLabel><toX><toY>
```

Example:

```txt
1.06졸05 2.03병04
```

Supported result values include `cho`, `han`, `draw`, `1-0`, `0-1`, and `1/2-1/2`; unknown values are treated as `unknown` and skipped by supervised export.

Run validation before export:

```bash
npm run data:validate-records
npm run ml:export-az-supervised
```
