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

## Supervised Policy Smoke Training

Sprint 18 adds a first PyTorch policy network and JSONL training pipeline.

Export sample opening-record data from TypeScript:

```bash
npm run ml:export-policy:sample
```

Train a tiny CPU smoke model:

```bash
cd ml
python -m oetongsu_ml.train_policy --data ../data/ml/policy_samples.sample.jsonl --output ../data/models/policy_net.sample.pt --epochs 1 --batchSize 2 --channels 8
```

Evaluate the checkpoint:

```bash
python -m oetongsu_ml.evaluate_policy --model ../data/models/policy_net.sample.pt --data ../data/ml/policy_samples.sample.jsonl
```

Export and train the first value network:

```bash
npm run ml:export-value:sample
cd ml
python -m oetongsu_ml.train_value --data ../data/ml/value_samples.sample.jsonl --output ../data/models/value_net.sample.pt --epochs 1 --batchSize 2 --channels 8
python -m oetongsu_ml.evaluate_value --model ../data/models/value_net.sample.pt --data ../data/ml/value_samples.sample.jsonl
```

The default `PolicyNet` and `ValueNet` use 64 convolution channels and are intended as simple baselines. CPU smoke runs can lower `--channels`; real training should use a larger dataset and preferably GPU acceleration.

## MCTS Prototype

Sprint 20 adds a first Python-side policy/value MCTS prototype:

```bash
cd ml
python -m oetongsu_ml.mcts_demo
```

The prototype includes a Python rules adapter, random and torch-backed policy/value inference, legal-move prior masking, visit-count policy targets, and smoke tests. It is not yet wired into self-play or the React game UI.

## Self-Play Data

Sprint 21 adds a first JSONL self-play generator:

```bash
npm run ml:selfplay:quick
```

Sprint 27 adds parallel self-play workers that write per-worker shards and merge them into one JSONL file:

```bash
npm run ml:selfplay:parallel:quick
```

Larger direct run:

```bash
cd ml
python -m oetongsu_ml.parallel_self_play --games 1000 --workers 4 --simulations 64 --model ../data/models/checkpoints/supervised_v0001.pt --output ../data/selfplay/parallel_1000.jsonl
```

Direct Python usage is also available:

```bash
cd ml
python -m oetongsu_ml.self_play_runner --games 1 --maxPlies 4 --simulations 4 --randomModel
```

It stores sparse MCTS visit distributions as `policy_target` and fills `value_target` after the game outcome is known. Generated files under `data/selfplay/` are ignored by git.

## AlphaZero Training Loop

Sprint 22 adds a tiny end-to-end AlphaZero-style training iteration:

```bash
npm run ml:az:quick
```

The quick loop generates self-play data, trains a dual-head `AlphaZeroNet`, writes latest model metrics, and stores a versioned checkpoint under `data/models/checkpoints/`.

## Model Arena And Promotion

Sprint 23 adds a checkpoint registry plus candidate-vs-champion arena:

```bash
npm run ml:model-arena:quick
```

Promotion rule: a candidate is marked `promoted` when its arena score rate is at least 55% with zero forfeits and zero illegal moves. Quick mode is only a smoke check; meaningful promotion needs many more games.

## Local AutoTrain Runner

Sprint 24 adds a local orchestration loop that connects self-play, AlphaZero training, candidate-vs-champion arena, and model registry promotion/rejection:

```bash
npm run ml:autotrain:quick
```

Direct Python usage for a larger local run:

```bash
cd ml
python -m oetongsu_ml.autotrain --iterations 10 --gamesPerIteration 100 --simulations 64 --trainEpochs 2 --promotionGames 40 --threshold 0.55 --allowRandomChampion
```

Parallel self-play inside AutoTrain:

```bash
cd ml
python -m oetongsu_ml.autotrain --iterations 10 --gamesPerIteration 1000 --selfplayWorkers 4 --parallelSelfPlay --simulations 64 --trainEpochs 2 --promotionGames 100
```

AutoTrain writes `data/training/autotrain_state.json`, `autotrain_log.jsonl`, and `autotrain_summary.json`. Quick mode is only a smoke test; real strength gains require much larger self-play counts and should be preceded by `npm run ml:rules:quick`.

## Initial Champion Bootstrap

Sprint 25 adds a supervised AlphaZero bootstrap path so AutoTrain can start from an opening-record model instead of a random champion.

Sample smoke:

```bash
npm run ml:export-az-supervised:sample
npm run ml:bootstrap-champion:sample
npm run ml:autotrain:quick
```

Real local start:

```bash
npm run ml:export-az-supervised
cd ml
python -m oetongsu_ml.bootstrap_champion --data ../data/ml/az_supervised_samples.jsonl --epochs 10 --batchSize 64 --channels 64 --version supervised_v0001
python -m oetongsu_ml.autotrain --iterations 10 --gamesPerIteration 100 --simulations 64 --trainEpochs 2 --promotionGames 40 --threshold 0.55
```

`supervised_v0001` is an opening-record bootstrap champion, not proof of playing strength. Strength should be judged only through repeated AutoTrain iterations and arena promotion.

## Local Training Server

Sprint 26 adds a local FastAPI server that exposes AutoTrain controls and training artifacts to the React `훈련` tab.

From the repository root:

```bash
npm run training:server
```

Development reload mode:

```bash
npm run training:server:dev
```

The server binds to `127.0.0.1:8765` and intentionally refuses non-local hosts for the MVP. It exposes health, training status, AutoTrain start/stop, model registry, recent logs, summary, and arena result endpoints.

The AutoTrain subprocess stdout and stderr are captured at:

- `data/training/server_last_stdout.log`
- `data/training/server_last_stderr.log`

See `docs/TRAINING_UI.md` for the UI workflow and endpoint list.

## Kakao-Like Ruleset

Pre-Sprint 24B defines explicit TS/Python rulesets before larger self-play runs. Python self-play and model arena can use the `kakao-like` ruleset, which keeps third-position repetition bans and adjudicates max-ply endings by material score with Han deom 1.5.

The ruleset is documented in `docs/JANGGI_RULESET.md`. Pass and detailed bikjang fault adjudication remain documented TODOs until the exact online-service behavior is verified and the policy index can encode pass safely.

## Rule Parity Hotfix Checks

Python self-play rules must stay aligned with the TypeScript game engine. The Python `TrainingPosition` stores `position_history`, and `generate_legal_moves` bans moves that would create a third occurrence of the same board plus turn.

Run the fast rule checks before AutoTrain work:

```bash
npm run ml:rules:quick
```

These checks include initial-position perft snapshots and repetition-ban behavior.

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
