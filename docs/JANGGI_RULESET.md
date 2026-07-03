# Oetongsu Janggi Ruleset

Oetongsu currently defines explicit rulesets so TypeScript play, Python self-play, arena evaluation, and future training jobs do not drift apart.

## Current Rulesets

- `oetongsu-basic`: compatibility ruleset for the existing engine tests and perft snapshots.
- `kakao-like`: Kakao Janggi-like practical online ruleset used as the ML training target.
- `kja-like`: placeholder for KJA-like adjudication. It currently shares the implemented enforcement behavior with `kakao-like` where exact adjudication is still unverified.

`kakao-like` is intentionally named as an approximation. If an official Kakao Janggi rule text is later verified, this document and the code policy values should be updated together.

## Repetition

Implemented policy:

- `ban-third-position`: a move is illegal if it would create a third occurrence of the same board plus side to move.
- `off`: repetition is not filtered.

Unimplemented policy notes:

- `draw-third-position` is reserved for rulesets that adjudicate the third occurrence as a draw.
- `adjudicate-fault` is reserved for future fault-side detection.

## Bikjang

The current engine treats facing generals as check and filters moves that expose the generals to each other. The ruleset layer records bikjang policy, but claim or fault adjudication is not yet separately implemented.

`kakao-like` and `kja-like` use `score-adjudication` as their documented target policy for unresolved bikjang/end adjudication cases.

## Pass

The ruleset records pass policy:

- `off`
- `allow-when-not-in-check`
- `allow-only-no-legal-move`

Pass move generation is not enabled yet. This avoids changing existing move-index, MCTS, and perft behavior before a pass-aware policy index is designed.

## Material Scoring

Material values:

- Chariot: 13
- Cannon: 7
- Horse: 5
- Elephant: 3
- Guard: 3
- Soldier: 2
- General: 0

Han receives a 1.5 deom. Initial material is therefore CHO 72 and HAN 73.5.

When `maxPlyPolicy` is `score-adjudication`, max-ply termination uses material score:

- higher score wins
- equal score draws

## TS/Python Parity

The TypeScript modules `ruleset.ts` and `scoring.ts` mirror Python `ruleset.py` and `scoring.py`.

Rule changes must be applied to both sides and covered by tests before self-play or AutoTrain data is generated.

## Open TODOs

- Verify official Kakao Janggi wording for bikjang, pass, repetition, and max-ply adjudication.
- Add pass move support only after move indexing and MCTS policy targets can encode pass safely.
- Replace `kja-like` repetition placeholder with adjudicate-fault logic if needed.
- Expand TS/Python parity fixtures for special ruleset behavior.
