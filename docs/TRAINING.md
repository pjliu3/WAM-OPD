# Training mixed-pool PWTR

## Prerequisites

Install the compatible FastWAM model dependencies and converted backbone assets.
Source fingerprints and environment versions are documented in `provenance/`.
Prepare a local [training bundle and binary assets](DATA.md#training-bundle).
The converted VAE must match the runtime identity described in
[EVALUATION.md](EVALUATION.md#runtime-and-model-inputs).

The training recipe uses two GPUs; its reference hardware configuration is two
A100 GPUs, 16 CPU cores and 220 GB host memory. Reserve 288 GiB filesystem headroom
for full checkpoint validation and export, and check the local storage quota.
Use a new output directory for each run.

## Command

Run from the source repository root:

```bash
python -m pip install -e '.[train]'
wam-pwtr doctor \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets --purpose train --hash-files
CUDA_VISIBLE_DEVICES=0,1 FASTWAM_OPD_FP32_ROPE=0 \
  wam-pwtr train \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets \
  --fastwam-root /path/to/FastWAM \
  --output /path/to/runs/pwtr-u96
```

## Fixed configuration

| Setting | Value |
|---|---|
| Initial state | Shared pretrained student checkpoint |
| Trajectory pool | 80 initial-student + 40 task-teacher trajectories |
| Selected decisions | Equally spaced, at most 16 per trajectory |
| Short trajectories | All decisions, without duplication or padding |
| Prefix | Earlier selected decisions only; current decision excluded |
| Weight strength / reference mixture / cap | 0.1 / 2:1 / 2 |
| Weight normalization | Within task × condition, under the sampling measure q |
| Weight refresh | Updates 0, 24, 48, 72 |
| Training | 96 updates, seed 0 |
| Trainable modules | Student MoT and proprioception encoder |
| Optimizer | AdamW, lr 1e-6, betas (0.9, 0.95), weight decay 0 |
| Gradient clipping / precision | 1.0 / BF16 with FP32 optimizer masters |
| Action / video loss coefficients | 10 / 0 |

The fixed recipe validates a schedule of 1309 teacher-supervision queries and
6148 student-path score evaluations: 6136 during weight refreshes and 12 during
startup checks. The local bundle must satisfy this accounting. Fixed reference
scores are validated before reuse; scoring and initialization preserve the query
RNG state. Training starts from initialization and does not resume interrupted runs.

## Checkpoints

Model parameters, optimizer state, RNG state and the query ledger are saved
together. Initial and final saves are checked by reading them back. The u96
inference export is `exports/update_000096/model_weights.pt`; use that file with
`test.py --checkpoint`. Each run records its own input and source identities.

## CPU validation

```bash
python -m pip install -e '.[eval,test]'
wam-pwtr verify-release
python -m pytest -q
```

The public tests use synthetic fixtures. To check a supplied training schedule
without loading a model, run `wam-pwtr preview --training-root /path/to/training-bundle`.
See [VALIDATION.md](VALIDATION.md) for validation scope.
