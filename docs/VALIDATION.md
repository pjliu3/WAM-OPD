# Validation

## CPU checks

The public test suite passes **131 CPU checks** using synthetic fixtures for sampling, weighting, asset
validation, preprocessing, checkpoint loading and evaluation orchestration.
It does not require the original trajectory pool, evaluation panel, score snapshots
or recorded success outcomes.

```bash
python -m pip install -e '.[eval,test]'
wam-pwtr verify-release
python -m pytest -q
```

The lightweight `.[test]` installation supports dependency-free checks. Install
`.[eval,test]` to include tensor preprocessing and checkpoint tests. The current
validation status and completed test count are recorded in `VALIDATION.json`.

## User-supplied inputs

`wam-pwtr preview --training-root /path/to/training-bundle` validates the local
training schedule without model inference. `test.py --check-only` validates a
supplied evaluation bundle and runtime without creating a model or environment
instance. See [TRAINING.md](TRAINING.md) and [EVALUATION.md](EVALUATION.md).

CPU tests and preflight checks do not measure closed-loop task success. A new
training run or environment evaluation requires the corresponding local assets
and an explicit execution command. No real model training or environment evaluation
has been executed as part of preparing this code-only package.
