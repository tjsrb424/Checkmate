# Oetongsu ML Foundation

This folder contains the first Python-side data foundation for future AlphaZero-style Oetongsu training.

Sprint 17 intentionally does not train a neural network. It defines stable numeric representations that the TypeScript Janggi engine can export and Python code can consume.

## Setup

```bash
cd ml
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

From the repository root, you can also run:

```bash
npm run ml:test
```

## TypeScript to Python JSON Contract

Positions exported from the TypeScript engine should use this structure:

```json
{
  "board": [
    [null, {"side": "HAN", "kind": "HORSE"}]
  ],
  "turn": "CHO",
  "history": [
    {"from": {"x": 0, "y": 6}, "to": {"x": 0, "y": 5}}
  ],
  "winner": null,
  "metadata": {}
}
```

- `board` is 10 rows by 9 columns.
- Each cell is `null` or `{ "side": "CHO" | "HAN", "kind": "GENERAL" | "GUARD" | "ELEPHANT" | "HORSE" | "CHARIOT" | "CANNON" | "SOLDIER" }`.
- Coordinates match the TypeScript engine exactly.
- `x` ranges from `0` to `8`.
- `y` ranges from `0` to `9`.
- `y = 0` is the Han home side.
- `y = 9` is the Cho home side.
- `turn` is `CHO` or `HAN`.
- A move is `{ "from": { "x": number, "y": number }, "to": { "x": number, "y": number } }`.

## Tensor Encoding

`encode_position(position)` returns a NumPy array with shape `(16, 10, 9)`.

Channels:

1. CHO GENERAL
2. CHO GUARD
3. CHO ELEPHANT
4. CHO HORSE
5. CHO CHARIOT
6. CHO CANNON
7. CHO SOLDIER
8. HAN GENERAL
9. HAN GUARD
10. HAN ELEPHANT
11. HAN HORSE
12. HAN CHARIOT
13. HAN CANNON
14. HAN SOLDIER
15. side-to-move CHO plane
16. side-to-move HAN plane

## Policy Index

The first policy index is deliberately simple:

```text
fromX * 10 * 9 * 10 + fromY * 9 * 10 + toX * 10 + toY
```

`POLICY_SIZE = 8100`.

This is not optimized for Janggi-specific move geometry yet. It is a stable bridge format for early supervised learning and self-play data.

