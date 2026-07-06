# Codex 전달문: M2 MacBook Air 소규모 학습 검증 / 풀 학습 금지

## 목적

이 작업은 M2 MacBook Air에서 외통수 학습 파이프라인이 정상 동작하는지만 확인하는 소규모 검증 작업이다.

중요:

- 이 MacBook에서는 full supervised 학습을 돌리지 않는다.
- `282,102 samples × 5 epochs` 같은 장시간 학습은 절대 실행하지 않는다.
- 풀 학습은 별도의 NVIDIA RTX 3070 Ti PC에서 진행한다.
- MacBook에서는 환경 세팅, 데이터 변환, 제한 validation/export, smoke champion, 웹 훈련 탭 확인까지만 한다.

---

## 1. 작업 위치

먼저 git clone 또는 git pull 받은 repository root에서 시작한다.

repository root는 다음 파일/폴더가 보이는 위치다.

```txt
package.json
src/
ml/
data/
scripts/
```

예시:

```bash
cd ~/dev/Checkmate
git pull
```

---

## 2. Node 환경 세팅

repository root에서 실행한다.

```bash
npm install
```

가벼운 확인:

```bash
npm run build
```

가능하면 테스트도 실행한다.

```bash
npm run test
```

---

## 3. Python venv 세팅

`ml` 폴더로 이동해서 venv를 만든다.

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

만약 `httpx2` 관련 설치 문제가 나면, `ml/requirements.txt`에서 다음을 수정한다.

```txt
httpx2>=0.28
```

를

```txt
httpx>=0.28
```

로 변경한 뒤 다시 설치한다.

---

## 4. PyTorch / MPS 확인

`ml` 폴더에서 실행한다.

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("mps_available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("mps_built:", torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False)
PY
```

기대:

```txt
cuda_available: False
mps_available: True 또는 False
```

M2 MacBook에는 CUDA가 없으므로 `cuda_available=False`는 정상이다.

---

## 5. ML 테스트

repository root 기준으로 실행한다.

```bash
cd ..
npm run ml:test
npm run ml:selfplay:parallel:quick
npm run ml:autotrain:quick
```

목표:

- 테스트 통과
- parallel self-play quick 정상
- autotrain quick 완료
- illegalMoves = 0
- forfeits = 0

---

## 6. 원본 기보 → processed CSV 변환

repository root에서 실행한다.

```bash
npm run data:convert-raw-records
npm run data:inspect-records
```

기대:

```txt
data/processed/janggi_clean_records.csv 생성
대략 19,000 records 확인
```

---

## 7. MacBook용 제한 validation

전체 validation은 오래 걸릴 수 있으므로 MacBook에서는 우선 limit 기반으로만 확인한다.

```bash
npm run data:validate-records -- --limit 500 --strict --maxIllegalRate 0.05 --maxUnknownRate 0.2
```

통과하면 충분하다.

주의:

- 전체 19,000건 validation은 MacBook에서 필수 아님.
- full validation은 RTX 3070 Ti PC에서 진행한다.

---

## 8. MacBook용 제한 supervised export

전체 export도 오래 걸릴 수 있으므로 MacBook에서는 제한 export만 수행한다.

```bash
npm run ml:export-az-supervised -- --limit 5000 --output data/ml/az_supervised_samples.mac_smoke.jsonl --summary data/ml/az_supervised_summary.mac_smoke.json
```

확인:

```bash
ls -lh data/ml/az_supervised_samples.mac_smoke.jsonl
cat data/ml/az_supervised_summary.mac_smoke.json
```

목표:

- sampleCount > 0
- export 파일 생성
- summary 생성

---

## 9. MacBook용 smoke champion 생성

`ml` 폴더로 이동한다.

```bash
cd ml
source .venv/bin/activate
```

아주 작은 smoke champion만 만든다.

```bash
python -m oetongsu_ml.bootstrap_champion   --data ../data/ml/az_supervised_samples.mac_smoke.jsonl   --output ../data/models/checkpoints/supervised_v0001_mac_smoke.pt   --version supervised_v0001_mac_smoke   --epochs 1   --batchSize 64   --channels 32   --overwrite
```

확인:

```bash
ls -lh ../data/models/checkpoints/supervised_v0001_mac_smoke.pt
cat ../data/models/registry.json
```

목표:

- checkpoint 생성
- registry에 `supervised_v0001_mac_smoke`가 promoted로 등록

주의:

- 이 모델은 실전용 champion이 아니다.
- smoke test용이다.

---

## 10. 선택: 아주 작은 lite champion

MacBook이 너무 뜨겁지 않고 시간이 허용되면 선택적으로만 실행한다.

```bash
python -m oetongsu_ml.bootstrap_champion   --data ../data/ml/az_supervised_samples.mac_smoke.jsonl   --output ../data/models/checkpoints/supervised_v0001_mac_lite.pt   --version supervised_v0001_mac_lite   --epochs 2   --batchSize 64   --channels 32   --overwrite
```

주의:

- 이 역시 실전 full champion이 아니다.
- MacBook에서는 여기까지만 한다.

---

## 11. 웹 훈련 탭 확인

터미널 1, repository root:

```bash
cd ~/dev/Checkmate
npm run training:server
```

터미널 2, repository root:

```bash
cd ~/dev/Checkmate
npm run dev
```

브라우저:

```txt
http://127.0.0.1:5173
```

훈련 탭에서 확인:

- Training Server 연결
- registry에 `supervised_v0001_mac_smoke` 표시
- summary/log/arena 표시

---

## 12. MacBook에서 실행할 작은 AutoTrain 값

훈련 탭에서 아래 값으로만 실행한다.

```txt
Iterations: 1
Games: 4
Simulations: 8
Max plies: 40
Epochs: 1
Batch: 8
Arena games: 2
Workers: 1
Parallel self-play: OFF 또는 ON
Threshold: 0.55
Ruleset: kakao-like
```

목표:

- 웹 UI에서 AutoTrain 시작/종료 확인
- registry/log/summary/arena 갱신 확인
- illegalMoves = 0
- forfeits = 0

주의:

- MacBook에서 `Workers 4`, `Games 50+`, `Simulations 32+`는 실행하지 않는다.
- MacBook Air는 팬이 없으므로 장시간 훈련 금지.

---

## 13. 절대 실행하지 말 것

MacBook에서는 아래 full 학습을 실행하지 않는다.

```bash
python -m oetongsu_ml.bootstrap_champion   --data ../data/ml/az_supervised_samples.jsonl   --output ../data/models/checkpoints/supervised_v0001.pt   --version supervised_v0001   --epochs 5   --batchSize 64   --channels 64
```

또한 아래 full export도 MacBook에서는 필수로 실행하지 않는다.

```bash
npm run ml:export-az-supervised
```

full export/full training은 RTX 3070 Ti PC에서 할 예정이다.

---

## 14. RTX 3070 Ti PC에서 나중에 할 작업

MacBook에서는 smoke만 확인한다.

RTX 3070 Ti PC에서는 다음을 진행한다.

1. git pull
2. npm install
3. Python venv 생성
4. CUDA-enabled PyTorch 설치
5. `torch.cuda.is_available() == True` 확인
6. full validation
7. full supervised export
8. full `supervised_v0001` 생성
9. 웹 훈련 탭에서 본격 AutoTrain

---

## 15. 생성 파일 커밋 주의

MacBook에서 생성되는 다음 파일들은 일반 git에 커밋하지 않는다.

```txt
data/processed/*.csv
data/processed/*.json
data/ml/*.jsonl
data/ml/*.json
data/models/
data/selfplay/
data/training/
```

commit 대상은 코드, 문서, 작은 fixture만이다.

---

## 16. 완료 보고 형식

작업이 끝나면 다음 형식으로 보고한다.

```txt
M2 MacBook Air 소규모 검증 완료

환경:
- node version:
- python version:
- torch version:
- cuda_available:
- mps_available:

실행 결과:
- npm install:
- npm run build:
- npm run test:
- npm run ml:test:
- npm run data:convert-raw-records:
- data:validate-records limit 500:
- ml:export-az-supervised limit 5000:
- supervised_v0001_mac_smoke 생성:
- training server:
- vite app:
- 훈련 탭 small AutoTrain:

생성 파일:
- data/processed/janggi_clean_records.csv
- data/ml/az_supervised_samples.mac_smoke.jsonl
- data/models/checkpoints/supervised_v0001_mac_smoke.pt

문제/주의:
-
```

---

## 핵심 한 줄

M2 MacBook Air에서는 전체 학습을 하지 않는다. 데이터 변환, 제한 export, smoke champion, 웹 훈련 탭 확인까지만 수행하고, full supervised_v0001 학습은 RTX 3070 Ti PC에서 진행한다.
