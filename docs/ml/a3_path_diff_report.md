# A3 Path Difference Report

## 1. Executive Summary

- 결론: A4 full RunPod는 아직 실행하지 않습니다.
- AutoTrain A3와 ablation LR 0.001은 둘 다 `supervised_v0001`에서 resume하는 경로입니다. rejected `az_iter_000002`에서 resume했다는 직접 증거는 코드상 확인되지 않았습니다.
- 큰 차이는 `seed`와 그로 인한 train/validation split 및 shuffle 순서입니다. AutoTrain A3는 `seed=1+3=4`, ablation은 `seed=7`입니다.
- checkpoint 내부 metrics는 매우 비슷하지만 arena 결과는 `0.0%` 대 `75.0%`로 크게 다릅니다. 따라서 단순 학습 성능보다 maxPlies/score-adjudication 평가 불안정성을 우선 의심해야 합니다.

## 2. Artifact Inputs

- artifact root: `D:\OetongsuArtifacts\a3_ablation_extract`
- 누락된 선택 파일: az_iter_000003_metrics.json, autotrain_summary.json, autotrain_state.json

| file | exists | size_bytes |
| --- | ---: | ---: |
| supervised_v0001.pt | True | 189796949 |
| az_iter_000002.pt | True | 189796405 |
| az_iter_000003.pt | True | 189796405 |
| ablation_a3_lr_0_001.pt | True | 189796501 |
| ablation_a3_lr_0_001_metrics.json | True | 903 |
| ablation_retrain_summary.json | True | 2149 |
| evaluation_summary.json | True | 3351 |
| az_iter_000003.jsonl | True | 590940176 |
| az_iter_000003_summary.json | True | 3057085 |
| az_iter_000003_arena.json | True | 16332 |
| az_iter_000003_metrics.json | False | - |
| autotrain_summary.json | False | - |
| autotrain_state.json | False | - |

## 3. Static Code Path Comparison

| 항목 | AutoTrain A3 | ablation LR 0.001 | 판정 |
| --- | --- | --- | --- |
| resume checkpoint source | `get_latest_promoted(registry)`에서 champion path를 찾고 `resumeChampion=True`이면 그 champion으로 resume | RunPod ablation script가 `--resume ../data/models/checkpoints/supervised_v0001.pt`를 명시 | 둘 다 champion resume 경로 |
| rejected candidate resume 가능성 | 코드상 candidate resume은 `champion_path`만 사용. `latestCandidateVersion`/rejected candidate를 resume source로 쓰지 않음 | 해당 없음 | 직접 증거 낮음 |
| data path | `../data/selfplay/az_iter_000003.jsonl` 생성 후 즉시 학습 | 같은 `../data/selfplay/az_iter_000003.jsonl` 재사용 | 동일 계열 |
| sample count | self-play summary 기준 `14674` | metrics 기준 `14674` | 동일 |
| epochs / batch size / channels / lr | `1 / 64 / 64 / 0.001` | `1 / 64 / 64 / 0.001` | 동일 |
| seed | `cfg.seed + iteration`, A3는 `4` | `7` | 중요 차이 |
| train/validation split | `random_split(..., manual_seed(seed))` | 같은 함수지만 seed가 다름 | 중요 차이 |
| shuffle | DataLoader `shuffle=True` | 동일하지만 seed가 다름 | 중요 차이 |
| optimizer | Adam, weight decay 없음 | Adam, weight decay 없음 | 동일 |
| checkpoint format | `model_state`, `channels`, `metrics` | 동일 | 동일 |
| arena | candidate vs champion, `promotion_threshold=0.55`, 40 games | registry-free evaluation, threshold effectively 0.5, 20 games | 평가 표본/판정 차이 |

## 4. Checkpoint Metadata Comparison

| checkpoint | exists | size_bytes | metadata_keys | state_keys | params | has_nan_or_inf | tensor_l2_norm |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| supervised_v0001 | True | 189796949 | `channels, metrics, model_state` | 10 | 47447845 | False | 1457.778881 |
| az_iter_000002 | True | 189796405 | `channels, metrics, model_state` | 10 | 47447845 | False | 1583.649222 |
| az_iter_000003 | True | 189796405 | `channels, metrics, model_state` | 10 | 47447845 | False | 1628.397002 |
| ablation_a3_lr_0_001 | True | 189796501 | `channels, metrics, model_state` | 10 | 47447845 | False | 1631.293612 |

| delta | parameter_delta_norm |
| --- | ---: |
| supervised_v0001 -> az_iter_000003 | 298.856410 |
| supervised_v0001 -> ablation_a3_lr_0_001 | 299.381477 |
| az_iter_000003 -> ablation_a3_lr_0_001 | 82.412350 |

## 5. Metrics Comparison

| model | metrics_source | sample_count | train_count | val_count | lr | epochs | resume | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| az_iter_000003 | checkpoint.metrics | - | - | - | - | - | - | 3.299974 | 2.790627 | 0.509347 | 0.500170 |
| ablation_a3_lr_0_001 | D:\OetongsuArtifacts\a3_ablation_extract\data\training\ablation_a3\ablation_a3_lr_0_001_metrics.json | 14674 | 11739 | 2935 | 0.001000 | 1 | ../data/models/checkpoints/supervised_v0001.pt | 3.268623 | 2.789588 | 0.479035 | 0.501193 |

- self-play summary sample_count: `14674`
- self-play games: `100`, workers: `4`

## 6. Model Output Comparison

Compared positions: 128

| model | exists | parameters | policy_entropy | legal_policy_mass | top1_legal_rate | value_mean | value_std | value_min | value_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised_v0001 | True | 47447845 | 3.368021 | 0.220670 | 0.203125 | 0.112304 | 0.300024 | -0.615991 | 0.674860 |
| az_iter_000003 | True | 47447845 | 2.637925 | 0.719065 | 0.921875 | -0.009957 | 0.609940 | -0.947110 | 0.831464 |
| ablation_a3_lr_0_001 | True | 47447845 | 2.769376 | 0.693860 | 0.906250 | 0.102511 | 0.653917 | -0.935392 | 0.935814 |

| pair | policy_kl_mean | top1_agreement_rate | value_delta_mean | value_abs_delta_mean |
| --- | ---: | ---: | ---: | ---: |
| supervised_v0001 vs az_iter_000003 | 2.495375 | 0.187500 | -0.122261 | 0.415601 |
| supervised_v0001 vs ablation_a3_lr_0_001 | 2.586312 | 0.164062 | -0.009793 | 0.479925 |
| az_iter_000003 vs ablation_a3_lr_0_001 | 0.167270 | 0.671875 | 0.112468 | 0.131857 |

## 7. Arena Result Comparison

| 항목 | AutoTrain az_iter_000003 | ablation_a3_lr_0_001 |
| --- | ---: | ---: |
| scoreRate | 0.000000 | 0.750000 |
| wins/losses/draws | 0/40/0 | 10/0/10 |
| averagePlies | 150.000000 | 150.000000 |
| margin avg/median | 12.000000/15.500000 | 3.500000/3.500000 |
| draw margin hits | 0 within / 40 outside | 10 within / 10 outside |
| paired warnings | [] | [] |

## 8. Most Likely Root Cause

가장 유력한 원인은 AutoTrain resume 버그보다는 학습 호출/분할 차이와 평가 불안정성의 결합입니다. 코드상 AutoTrain A3도 latest promoted champion인 `supervised_v0001`을 resume source로 사용하고, ablation도 `supervised_v0001`을 명시적으로 resume합니다. 다만 AutoTrain은 seed `1 + iteration`, ablation은 seed `7`을 쓰므로 train/validation split과 shuffle 순서가 달라집니다. 두 체크포인트의 checkpoint 내부 metrics는 거의 비슷하지만 arena 결과는 크게 갈렸고, 모든 비교가 `maxPlies=150` score-adjudication에 묶여 있어 평가 변동성/역할 표본 문제가 강하게 남아 있습니다.

## 9. Recommended Fix

- AutoTrain candidate initialization에 guard test를 추가해 candidate resume source가 항상 latest promoted champion인지 검증합니다.
- AutoTrain metrics/checkpoint metadata에 `resume`, `seed`, `train_count`, `val_count`, `sample_count`, `split_seed`, `optimizer`를 명시적으로 남깁니다.
- A4 전에 같은 checkpoint pair를 더 큰 paired local/cheap evaluation으로 재검증합니다.
- promotion arena에 maxPlies/score-adjudication confidence gate를 추가합니다.
- seed 차이만으로 결과가 뒤집히는지 확인하는 작은 local 재현 테스트를 먼저 설계합니다.

## 10. RunPod Decision

## 11. Sprint 41 Follow-up

- Added an AutoTrain candidate-init guard so the candidate resume source resolves to the latest promoted champion, not the latest rejected candidate.
- Added `training_metadata` to AlphaZero metrics JSON and checkpoints.
- Added a cheap validation gate for optional pre-arena checks.
- Added an optional AutoTrain cheap-validation hook, disabled by default unless `--cheapValidationBeforeArena` is set.
- Added a seed sensitivity probe for local small-limit checks before any A4 or RunPod work.
- RunPod A4 remains blocked until guard tests, metadata validation, cheap validation, and seed-sensitivity checks pass.

- Full RunPod A4: **금지**.
- AutoTrain 추가 실행: **금지**.
- 새 self-play 생성: **금지**.
- registry champion 변경: **금지**.
- 허용: 이 path-diff report 기반의 코드 수정, 메타데이터 기록 보강, 작은 로컬/CPU-first 검증.
