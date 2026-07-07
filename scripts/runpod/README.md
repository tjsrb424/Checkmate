# RunPod AutoTrain scripts

## A3 calibrated resume

Run from a RunPod instance after placing the A2 artifact at the repository root:

```bash
cd /workspace/Checkmate
bash scripts/runpod/run_autotrain_a3_calibrated_autostop.sh
```

Required before launch:

```txt
/workspace/Checkmate/oetongsu_runpod_a2_artifacts.tgz exists
RUNPOD_API_KEY is set
RUNPOD_POD_ID is set
```

The A3 script resumes from `completedIterations >= 2`, expects `supervised_v0001` as the current champion and `az_iter_000002` as the latest rejected candidate, then runs only up to iteration 3 to create `az_iter_000003`.

## A3 full ablation

Run from a RunPod instance after placing the A3 artifact at the repository root:

```bash
cd /workspace/Checkmate
bash scripts/runpod/run_ablation_a3_full_autostop.sh
```

Detached run:

```bash
cd /workspace/Checkmate
nohup bash scripts/runpod/run_ablation_a3_full_autostop.sh > data/training/ablation_a3_launcher.nohup.log 2>&1 &
echo $! > data/training/ablation_a3_launcher.pid
disown
```

Required before launch:

```txt
/workspace/Checkmate/oetongsu_runpod_a3_artifacts.tgz exists
RUNPOD_API_KEY is set
RUNPOD_POD_ID is set
```

Useful checks:

```bash
cd /workspace/Checkmate
cat data/training/runpod_a3_ablation_progress.json
tail -100 data/training/ablation_a3_launcher.nohup.log
tail -100 data/training/ablation_a3_runpod.log
ls -lh oetongsu_runpod_a3_ablation_artifacts.tgz
```

The ablation script restores the A3 artifact, verifies `completedIterations >= 3`, retrains LR candidates from the fixed A3 self-play data, evaluates them against `supervised_v0001`, packages `oetongsu_runpod_a3_ablation_artifacts.tgz`, then requests RunPod stop.
