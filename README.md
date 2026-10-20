<div align="center">

<picture>
  <source media="(max-width: 600px)" srcset="assets/title-mobile.svg">
  <img src="assets/title.svg" alt="WAM-OPD: Sharpening World Action Models via On-Policy Distillation" width="900">
</picture>

Panjun&nbsp;Liu · Xiaohan&nbsp;Lei · Shiqi&nbsp;Zhang · Yikun&nbsp;Wang<br>
Yongxin&nbsp;Zhang · Mingyi&nbsp;Hu · Shida&nbsp;Sun · Jiateng&nbsp;Shou<br>
Wengang&nbsp;Zhou · Jiajun&nbsp;Deng<sup>✉</sup> · Zhiwei&nbsp;Xiong

**University of Science and Technology of China**<br>
<sub>✉ Corresponding author</sub>

<a href="https://arxiv.org/abs/2609.34250v1"><img src="assets/paper.svg" alt="Paper · arXiv" height="32"></a>&nbsp;
<a href="https://pjliu3.github.io/wam-opd-project/"><img src="assets/project.svg" alt="Project page" height="32"></a>&nbsp;
<a href="#citation"><img src="assets/citation.svg" alt="BibTeX" height="32"></a>&nbsp;
<a href="LICENSE"><img src="assets/license.svg" alt="MIT License" height="32"></a>

[Overview](#overview) · [Method](#method) · [Results](#results) · [Quick start](#quick-start)<br>
[English](README.md) / [简体中文](README.zh-CN.md)

**Task adaptation with a fixed trajectory pool and fresh teacher supervision.**<br>
No additional environment rollouts during distillation.

</div>

<p align="center">
  <img src="assets/teaser.png" alt="Figure 1: WAM-OPD transfers task-specific knowledge through dense velocity supervision on student denoising paths, using a fixed trajectory pool." width="100%">
</p>

## Overview

**WAM-OPD** sharpens pretrained world action models on target tasks while retaining
their broader manipulation skills. Its core method, **Prefix-Weighted Trajectory
Replay (PWTR)**, combines stored observation histories with fresh student queries
and frozen task-teacher supervision.

This repository provides the **mixed-pool PWTR** implementation: training,
checkpoint export, model inference, and evaluation utilities. The source release
contains code, method documentation and paper figures. Training and evaluation
bundles, model weights, trajectory records and feature caches are supplied locally.

| Fixed trajectory pool | Fresh student queries | Prefix-weighted updates |
|:---|:---|:---|
| Reuse initial-student and task-teacher rollouts. | Generate current-student denoising paths at stored histories. | Weight each decision using compatibility scores from its preceding sampled prefix. |

## Method

<p align="center">
  <img src="assets/method.png" alt="Figure 2: The mixed trajectory pool, prefix compatibility scoring, and teacher-supervised student update in WAM-OPD." width="100%">
</p>
<p align="center"><sub>WAM-OPD framework · Figure 2 of the paper</sub></p>

1. **Collect once, replay throughout training.** Build a fixed pool from the initial
   student and task teachers. Select up to 16 equally spaced decisions per
   trajectory; use every decision when a trajectory is shorter.
2. **Estimate prefix compatibility.** Score stored denoising paths under a
   periodically refreshed student snapshot and two fixed references. Accumulate scores over earlier
   selected decisions, then normalize, clip and renormalize the resulting weights.
3. **Distill on fresh queries.** At each replayed history, query the frozen task
   teacher along a fresh student denoising path. Apply the weighted velocity loss
   and update the student once per trajectory.

The current decision is excluded from its own prefix. Teacher targets and weights
are stop-gradient. See the [method specification](docs/METHOD.md) for the score,
weighting formula and implementation details.

## Results

**Simulation results reported in the paper (Table 1).** Success rates are in %.
The same adapted model is evaluated on two target tasks and six retention tasks.

| Method | Move Stapler Pad | Open Microwave | Target average | Retention average |
|:---|---:|---:|---:|---:|
| **WAM-OPD (PWTR)** | **80.50** | **59.00** | **69.75** | **86.92** |

The paper also evaluates transfer to real-world manipulation. See the
[project page](https://pjliu3.github.io/wam-opd-project/) for robot
demonstrations and the [paper](https://arxiv.org/abs/2609.34250v1) for the full
experimental results.

## Quick start

### 1. Install

Run from the repository root. CPU utilities support Python 3.10+; the model
runtime uses Python 3.11 and PyTorch 2.7.1.

```bash
python -m pip install -e '.[eval,test]'
wam-pwtr verify-release
python -m pytest -q
```

Install compatible FastWAM and RoboTwin dependencies and their external assets
for model training or environment evaluation. See the [data guide](docs/DATA.md)
for the required local bundle layouts.

### 2. Check a local training bundle

`--training-root` contains `assets.json`, `pool.json`, `reference_scores.json` and
`schedule.json`. `--asset-root` contains the model and trajectory binaries named
by those manifests.

```bash
wam-pwtr preview --training-root /path/to/training-bundle
wam-pwtr doctor \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets --purpose train --hash-files
```

`preview` validates the fixed sampling schedule without loading a model.
Inputs and caches must already be available locally; commands do not download them.

### 3. Train PWTR

```bash
CUDA_VISIBLE_DEVICES=0,1 FASTWAM_OPD_FP32_ROPE=0 \
  wam-pwtr train \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets \
  --fastwam-root /path/to/FastWAM \
  --output /path/to/runs/pwtr-u96
```

The fixed recipe uses **120 mixed trajectories**, **96 updates**, and a weight
refresh every **24 updates**. It requires two GPUs. Hyperparameters, query
accounting and resource requirements are in the [training guide](docs/TRAINING.md).

### 4. Run saved-history inference

```bash
CUDA_VISIBLE_DEVICES=0 wam-pwtr infer-record \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets \
  --fastwam-root /path/to/FastWAM \
  --record-id RECORD_ID \
  --seed 0 \
  --output /path/to/prediction.pt
```

This uses the `pwtr_u96` checkpoint identified by the local asset manifest and
returns normalized action latents for a saved history. Robot execution additionally
requires action denormalization, observation feedback and replanning.

### 5. Test a checkpoint in RoboTwin

Provide a local [evaluation bundle](docs/EVALUATION.md) and its text contexts:

```bash
python test.py \
  --evaluation-root /path/to/evaluation-bundle \
  --checkpoint /path/to/model_weights.pt \
  --fastwam-root /path/to/FastWAM \
  --robotwin-root /path/to/RoboTwin \
  --text-cache /path/to/text_cache \
  --gpu 0 \
  --output /path/to/evaluation/pwtr-u96
```

The command loads the specified checkpoint and evaluates the local 400-case
bundle (two tasks × two conditions × 100 scenes). Add `--check-only` to validate dependencies and assets without creating a
model or simulation scene. Task and condition selection are described in the
[evaluation guide](docs/EVALUATION.md).

New results are written to `summary.json`, `summary.csv` and `outcomes.jsonl` in
the chosen output directory. The source release does not include evaluation scenes,
recorded outcomes or text-feature caches.

## Documentation

| Guide | Contents |
|:---|:---|
| [Method](docs/METHOD.md) | Sampled prefixes, compatibility scores and distillation loss |
| [Training](docs/TRAINING.md) | Fixed recipe, resource requirements and checkpoints |
| [Local bundles](docs/DATA.md) | Metadata, model and trajectory inputs |
| [Evaluation](docs/EVALUATION.md) | Run a checkpoint on a supplied RoboTwin panel |
| [Validation](docs/VALIDATION.md) | CPU tests using synthetic fixtures |

<details>
<summary><b>Repository structure</b></summary>

```text
test.py            Closed-loop RoboTwin task evaluation
src/wam_pwtr/      PWTR commands, training, inference and evaluation
src/wam_opd_v2/    Training loop, optimizer and checkpoint validation
src/fastwam_opd/   FastWAM model and record interfaces
configs/           Method and environment configuration
assets/            Paper figures and README graphics
tests/             CPU tests with synthetic fixtures
tools/             Local asset utilities
provenance/        Source and external-runtime fingerprints
```

</details>

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026wamopd,
  title = {{WAM-OPD}: Sharpening World Action Models via On-Policy Distillation},
  author = {Liu, Panjun and Lei, Xiaohan and Zhang, Shiqi and Wang, Yikun
            and Zhang, Yongxin and Hu, Mingyi and Sun, Shida and Shou, Jiateng
            and Zhou, Wengang and Deng, Jiajun and Xiong, Zhiwei},
  year = {2026},
  eprint = {2609.34250},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.34250}
}
```

## License

Project code is released under the [MIT License](LICENSE). External libraries,
pretrained checkpoints and task assets retain their respective licenses; see
[Third-party notices](THIRD_PARTY_NOTICES.md).
