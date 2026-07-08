# A3 Local Validation Report

## 1. Executive Summary

- Seed verdict: `high`. Seed sensitivity is meaningful: best seed `7` beats worst seed `4` by val_total_loss `0.492622`.
- Cheap/full consistency: `inconsistent`. Cheap gate does not separate A3 and ablation, so games=4/sims=8/maxPlies=80 is too weak or noisy as a standalone gate.
- Stop/Go decision: **STOP full A4; redesign cheap gate before relying on it**.

## 2. Inputs

- seed: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\seed_probe\seed_sensitivity_summary.json`
- cheap_a3: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\cheap_validation_az_iter_000003.json`
- cheap_ablation: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\cheap_validation_ablation_lr_0_001.json`
- full_a3: `D:\OetongsuArtifacts\a3_ablation_extract\data\models\arena\az_iter_000003_arena.json`
- full_ablation: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\ablation_a3\evaluation_summary.json`

## 3. Seed Sensitivity Probe

| seed | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 | checkpoint |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 6.060398 | 5.164271 | 0.896127 | 0.176471 | `D:\OetongsuArtifacts\a3_ablation_extract\data\training\seed_probe\seed_4.pt` |
| 7 | 5.567775 | 4.584496 | 0.983279 | 0.352941 | `D:\OetongsuArtifacts\a3_ablation_extract\data\training\seed_probe\seed_7.pt` |

Seed sensitivity is meaningful: best seed `7` beats worst seed `4` by val_total_loss `0.492622`.

## 4. Cheap Validation: az_iter_000003

| status | score_rate | games | simulations | maxPlies | wins/losses/draws | warnings |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| pass | 0.750 | 4 | 8 | 80 | 2/0/2 | all games reached maxPlies, score adjudication dominated the gate |

## 5. Cheap Validation: ablation LR 0.001

| status | score_rate | games | simulations | maxPlies | wins/losses/draws | warnings |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| pass | 0.750 | 4 | 8 | 80 | 2/0/2 | all games reached maxPlies, score adjudication dominated the gate |

## 6. Comparison With Full Arena Results

| candidate | cheap_status | cheap_score_rate | full_score_rate | full_w/l/d | avg_plies |
| --- | --- | ---: | ---: | --- | ---: |
| az_iter_000003 | pass | 0.750 | 0.000 | 0/40/0 | 150.0 |
| ablation_a3_lr_0_001 | pass | 0.750 | 0.750 | 10/0/10 | 150.0 |

Cheap gate does not separate A3 and ablation, so games=4/sims=8/maxPlies=80 is too weak or noisy as a standalone gate.

## 7. Stop/Go Decision

- **STOP full A4; redesign cheap gate before relying on it**
- RunPod full A4 remains blocked.
- Champion registry mutation remains blocked.
- Any micro-run must be explicitly bounded and preceded by cheap validation plus documented stop conditions.

## 8. Recommended Next Sprint

- Next sprint should increase paired games/repeats or add margin confidence gates.
- Add repeat-based cheap validation thresholds if single-run gate results are noisy.
- Keep all generated checkpoints, JSONL, and local validation outputs out of Git.
