# Opening Book Data

Use `data/processed/janggi_clean_records.csv` for the cleaned game-record CSV.

Run:

```bash
npm run opening:build
```

The generated files are written to:

- `data/opening-book/opening-book.json`
- `data/opening-book/opening-book-summary.json`

`data/raw/` is for original ZIP/GIB files and is not committed. Large generated books are also not committed by default. The app still uses the small built-in seed book until a later lazy-load or worker-preload step.
