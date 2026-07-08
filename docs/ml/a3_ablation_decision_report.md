# A3 Ablation Decision Report

## 1. Executive Summary

- Best actionable candidate: `ablation_a3_lr_0_001` (LR `0.0010`), selected by score rate, then lower margin, then lower validation loss.
- Raw top score-rate candidates: `ablation_a3_lr_0_0001`, `ablation_a3_lr_0_001`; this matters because the source JSON's first tied candidate is not necessarily the strongest operational choice.
- Original A3 baseline was `0.0%` (0-40-0). Best ablation is `75.0%` (10-0-10).
- A3 baseline improvement: yes, but every ablation game still reached `maxPlies=150`, so adjudication/draw handling remains a major uncertainty.
- Stop/go decision: **STOP for full RunPod A4 now; GO only for local/cheap confirmation gates.**

## 2. Inputs

- Evaluation summary: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\ablation_a3\evaluation_summary.json`
- Retrain summary: `D:\OetongsuArtifacts\a3_ablation_extract\data\training\ablation_a3\ablation_retrain_summary.json`
- A3 arena baseline: `D:\OetongsuArtifacts\a3_ablation_extract\data\models\arena\az_iter_000003_arena.json`
- Artifact archive: `D:/OetongsuArtifacts/oetongsu_runpod_a3_ablation_artifacts.tgz`
- Extracted artifact root: `D:/OetongsuArtifacts/a3_ablation_extract`

## 3. A3 Failure Recap

- A3 arena result: candidate `0` wins, champion `40` wins, draws `0`, score rate `0.0%`.
- Average plies: `150.0`. The run was dominated by score adjudication at the maximum ply limit.
- Average margin: `12.000`. This is much larger than the `1.5` draw margin and indicates a real practical collapse under that evaluation setup.

## 4. Ablation Retrain Metrics

| LR | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0010 | 3.268623 | 2.789588 | 0.479035 | 0.501193 |
| 0.0003 | 4.467597 | 3.617988 | 0.849609 | 0.396593 |
| 0.0001 | 5.652000 | 4.713879 | 0.938121 | 0.272913 |

## 5. Ablation Evaluation Result

| candidate | LR | score_rate | wins | losses | draws | avg_plies | margin_avg | margin_median | draw_margin_hits | warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ablation_a3_lr_0_0001` | 0.0001 | 75.0% | 10 | 0 | 10 | 150.0 | 4.000 | 4.000 | 10 within / 10 outside | - |
| `ablation_a3_lr_0_001` | 0.0010 | 75.0% | 10 | 0 | 10 | 150.0 | 3.500 | 3.500 | 10 within / 10 outside | - |
| `ablation_a3_lr_0_0003` | 0.0003 | 50.0% | 10 | 10 | 0 | 150.0 | 4.000 | 4.000 | 0 within / 20 outside | 모든 pair가 같은 진영 승리로 갈렸습니다.; 후보/챔피언 강도보다 진영 우세가 pair 결과를 지배할 수 있습니다. |

## 6. Best Candidate

- Choose LR `0.0010` / `ablation_a3_lr_0_001` as the best actionable candidate.
- Reason: it ties the top score rate at `75.0%`, has the lower average/median margin among the tied candidates, and has the strongest validation losses from retraining.
- Treat LR `0.0001` as a raw-score tie, not the operational winner; its validation losses are much worse despite the same short-match score rate.
- Treat LR `0.0003` as rejected for now because it scored only `50.0%` and its paired summary warned that side advantage dominated the result.

## 7. Updated Root Cause Hypothesis

- The failure is probably not a simple learning-rate-only issue. LR changes can recover short-evaluation score rate, but the original A3 full iteration still collapsed to `0.0%`.
- The stronger hypothesis is a self-play target/data loop problem: max-plies-heavy games, score adjudication targets, replay composition, and champion/candidate policy drift can create candidates that look trained but fail promotion.
- Arena/adjudication calibration remains a confounder because every ablation game ended at `150` plies and many wins/draws are margin-threshold decisions, not checkmates.

## 8. Cost Review

- This RunPod artifact reached packaging after about `04:59:05` wall time and already exceeded the intended exploratory budget.
- Continuing with full A4-style RunPod runs before local confirmation risks paying again for another non-promotable candidate.
- The cost-effective next step is analysis plus small deterministic validation: no new self-play, no champion registry mutation, and no long GPU run until gates pass.

## 9. Stop/Go Decision

- Full RunPod A4: **STOP**.
- Local report and expanded evaluation: **GO**.
- Limited RunPod micro-run: **conditional GO** only if local gates confirm LR `0.0010` remains above `60%` score rate with stable margins.
- Registry champion change: **STOP** until promotion criteria pass on a fresh, documented evaluation.

Required gates before any full RunPod run:

- Re-run the best candidate locally or in a capped job with at least `40` paired games.
- Confirm score rate is above `60%`, not just a first-order tie from a 20-game sample.
- Confirm paired summary is not side-dominated and illegal moves/forfeits remain zero.
- Confirm average margin is below the failed A3 baseline and draw-margin hits are explainable.
- Document expected wall time, expected cost, timeout, and autostop path before launch.

## 10. Recommended Next Sprint

- Recommended path: **Option B, self-play target/data redesign**, with LR `0.001` retained as the cheap validation probe.
- Do not run A4 full training as the first action. First inspect value target construction, score-adjudication value labels, maxPlies-heavy samples, replay-buffer composition, and supervised/self-play data mixing.
- If the local gates pass after those checks, run a deliberately capped A4 candidate with `learningRate=0.001`, `gamesPerIteration=100`, `simulations=48`, `arenaSimulations=48`, `maxPlies=150`, `adjudicationDrawMargin=1.5`, `trainEpochs=1`, and `batchSize=64`.
