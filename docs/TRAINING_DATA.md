# Training Data Intake

The current blocker for real supervised training is data, not engine wiring. `npm run ml:export-az-supervised` expects:

```txt
data/processed/janggi_clean_records.csv
```

That file is local training data and is ignored by git. Keep source and license notes with the local dataset before running long training jobs.

## Raw Record Conversion

Raw records can be placed under:

```txt
data/raw/janggi-records/
```

This repository currently includes the user-provided raw `.gib` and `.zip` bundles there so another checkout can reproduce the conversion. The stored filenames use stable ASCII slugs, and `data/raw/janggi-records/README.md` maps them back to the original Korean filenames.

Default conversion:

```bash
npm run data:convert-raw-records
```

Defaults:

- input: `data/raw/janggi-records`
- output: `data/processed/janggi_clean_records.csv`
- summary: `data/processed/janggi_clean_records.conversion.json`

Probe only:

```bash
npm run data:convert-raw-records -- --probeOnly
```

Fixture smoke:

```bash
npm run data:convert-raw-records -- --inputDir data/fixtures/raw-records --output data/processed/sample_converted.csv --summary data/processed/sample_conversion.json
npm run data:validate-records -- --input data/processed/sample_converted.csv
```

The converter supports compact CSV/TSV, JSON, JSONL, ZIP-contained text records, and CP949/EUC-KR `.gib` text. GIB move text such as `1. 79졸78` is normalized to the compact token shape expected by the existing opening parser.

## CSV Schema

Minimum header:

```csv
source,group,game_index,cho_chalim,han_chalim,result,moves16_ok,first16
```

Useful aliases:

- `game_index` or `id`
- `cho_chalim`, `choChalim`, or `cho_formation`
- `han_chalim`, `hanChalim`, or `han_formation`
- `result`, `winner`, `outcome`, or `raw_result`
- `first16`, `first_16`, `opening16`, or `moves16`
- `moves16_ok`, `moves16Ok`, `first16_ok`, or `opening_ok`

`first16` uses compact opening tokens:

```txt
<plyNumber>.<fromX><fromY><pieceLabel><toX><toY>
```

Example:

```txt
1.06졸05 2.03병04
```

## Validate

Default local validation:

```bash
npm run data:validate-records
```

Sample fixture smoke:

```bash
npm run data:validate-records -- --input data/fixtures/opening-records.sample.csv --summary data/processed/sample.validation.json
npm run data:inspect-records -- --input data/fixtures/opening-records.sample.csv
```

Strict mode:

```bash
npm run data:validate-records -- --strict --maxIllegalRate 0.05 --maxUnknownRate 0.2
```

Validation summary fields include `recordCount`, `validRecordCount`, `illegalMoveRate`, `unknownResultRate`, `parseFailureCount`, and sample invalid rows.

## Export And Bootstrap

After validation passes on the real CSV:

```bash
npm run ml:export-az-supervised
cd ml
python -m oetongsu_ml.bootstrap_champion --data ../data/ml/az_supervised_samples.jsonl --output ../data/models/checkpoints/supervised_v0001.pt --version supervised_v0001 --epochs 5 --batchSize 64 --channels 64
```

Then run the supervised champion sanity check and staged AutoTrain runs from `docs/TRAINING_UI.md` or the local training execution plan.
