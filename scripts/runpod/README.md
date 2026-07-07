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
