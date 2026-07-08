# RunPod Cost Control Policy

## Purpose

Keep Oetongsu ML work from turning failed experiments into repeated GPU spend. RunPod is for bounded confirmation runs after local analysis has already shown a plausible path to improvement.

## Principles

- Do artifact analysis locally before launching another remote job.
- Do not run A4, AutoTrain, self-play generation, or registry promotion just because the previous run produced files.
- Treat score-rate collapse near `0.0` as a stop signal until root cause evidence changes.
- Prefer small, reproducible validation over long exploratory runs.
- Keep `data/models`, `data/training`, `data/selfplay`, checkpoints, archives, and JSONL outputs out of commits unless explicitly requested.

## RunPod Usage Gate

A RunPod job is allowed only when all of these are documented:

- AutoTrain candidate-init guard tests pass.
- A cheap validation gate result exists before any full promotion arena for disputed candidates.
- A seed sensitivity probe has been run or explicitly waived with rationale before A4.
- A current path-diff report exists when AutoTrain and ablation results disagree.
- Any AutoTrain/ablation path mismatch has a documented explanation or a code fix before A4.
- Objective, expected output files, and expected decision after the run.
- Maximum wall time, expected cost, timeout, and autostop method.
- Local smoke test or small evaluation result that justifies remote GPU use.
- Stop condition for low score rate, side-dominated arena pairs, repeated max-plies adjudication, missing outputs, or no measurable improvement.
- Confirmation that the job will not mutate the champion registry unless promotion gates pass.

## Work Allowed On RunPod

- Capped retraining or evaluation that is too slow locally but already has a specific hypothesis.
- Short confirmation of a locally selected candidate.
- Packaging artifacts for local analysis.

## Work To Do Locally First

- Read evaluation, retrain, arena, progress, and autostop artifacts.
- Generate decision reports.
- Generate a path-diff report before any full run when AutoTrain and ablation disagree.
- Run unit tests and CLI smoke tests.
- Prefer local/CPU-first evaluation when the check is CPU-bound or only reads existing checkpoints.
- Inspect whether arena results are side-dominated, draw-threshold dominated, max-plies-heavy, or affected by illegal moves/forfeits.
- Decide whether the next change is hyperparameter tuning, self-play target/data repair, or a pause in AlphaZero full training.

## Stop Criteria

- Candidate score rate is near `0.0`.
- Candidate does not improve over the failed baseline.
- Candidate resume source cannot be proven to be the latest promoted champion.
- Cheap validation gate is missing, failed, or side-dominated for a disputed candidate.
- Seed sensitivity has not been checked before A4 after a seed/split-related discrepancy.
- AutoTrain and ablation paths disagree and the difference has not been explained.
- Arena pairs are side-dominated or all games reach max plies without useful separation.
- A required artifact is missing or malformed.
- Autostop, timeout, or budget cap is not configured.
- Two consecutive expensive runs show no promotion-quality improvement.

## Single-Run Budget Standard

Before launch, record:

- Target maximum wall time.
- Expected dollar cost.
- Hard timeout.
- Autostop command or watchdog.
- Artifact package path.
- The exact command to retrieve artifacts.

If the estimate is uncertain, run a smaller confirmation job first.

## Repeated No-Result Policy

After two expensive runs without a promotion-quality result, stop full training and redesign the training strategy. For the current A3 evidence, that means investigating self-play value targets, score-adjudication labels, replay-buffer composition, supervised/self-play mixing, and champion policy regularization before any full A4 run.

## Champion Registry Policy

Champion changes are allowed only after documented promotion criteria pass. Ablation checkpoints, quick experiments, and candidates from score-rate ties must not update the registry.
