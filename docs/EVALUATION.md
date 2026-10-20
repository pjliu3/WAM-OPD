# Test a checkpoint in RoboTwin

`test.py` loads a complete student checkpoint and runs the scenes defined by a
local evaluation bundle. It writes new per-episode outcomes and success-rate
summaries to a new output directory.

## Runtime and model inputs

Use Linux, a CUDA-capable NVIDIA GPU, and a working SAPIEN/Vulkan rendering setup.
Install compatible FastWAM and RoboTwin 2.0 dependencies in the same Python
environment. Python 3.11 and PyTorch 2.7.1 are the reference versions.

```bash
python -m pip install -e '.[eval]'
```

The external runtime supplies:

- FastWAM model code, `sim_robotwin.yaml` and its converted Wan VAE.
- RoboTwin task code, the Aloha-AgileX embodiment, task objects, textures,
  motion-planning libraries and camera configuration.
- A complete student checkpoint, supplied with `--checkpoint`.

For a model trained with this package, use
`exports/update_000096/model_weights.pt`. The loader checks the complete MoT and
proprioception-encoder keys and tensor shapes. Runtime source identities are
specified in `provenance/evaluation_runtime.json`.

The required converted VAE is
`DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`,
1,409,401,152 bytes, SHA256
`0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996`.
The identity matches the [upstream ModelScope file](https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/files)
at revision `150f75d811d51f6c7760154aa7fec371dccda529`. Provide it under FastWAM's
`checkpoints/` directory or `DIFFSYNTH_MODEL_BASE_PATH`. Its content is checked
before model construction.

Each task/condition group runs in a separate process, sequentially on the selected
GPU, with one model load per group. The model and simulator share that GPU.
Configure driver, Vulkan ICD and native-library paths for your installation.

## Local evaluation bundle

The source package includes no scene lists, per-scene instructions, expert caches,
normalization statistics, text features or recorded evaluation outcomes. Provide
these inputs locally using the following layout:

```text
evaluation-bundle/
├── evaluation_scenes.json
├── evaluation_instructions.json
├── evaluation_assets.json
├── evaluation_dataset_stats.json
└── expert_cache/<task>/demo_<condition>/<seed>.json
```

The full bundle must provide 100 scenes per task and condition: two tasks × two
conditions × 100 scenes, with noise ID 0. Scene IDs and instructions come from
your local bundle. Pass its directory directly as `--evaluation-root`, without
an additional `data/` level. Pass the separate T5 context directory as `--text-cache`. The bundle's
asset manifest binds each required file by size and SHA256. See
[DATA.md](DATA.md#evaluation-bundle) for the file roles.

## Check inputs

Run from the source repository root:

```bash
python test.py \
  --evaluation-root /path/to/evaluation-bundle \
  --checkpoint /path/to/model_weights.pt \
  --fastwam-root /path/to/FastWAM \
  --robotwin-root /path/to/RoboTwin \
  --text-cache /path/to/text_cache \
  --gpu 0 \
  --output /path/to/evaluation/pwtr-u96 \
  --check-only
```

Preflight checks the selected cases, asset and source fingerprints, VAE identity,
normalization, model configuration and runtime imports. Hashing the complete
checkpoint may take several minutes on network storage. Preflight does not
construct a model or simulator scene; GPU inference and rendering are exercised
by the actual evaluation command.

The default statistics and expert-cache paths resolve within `--evaluation-root`.
`--dataset-stats` and `--expert-cache` can select matching local copies. Missing
assets stop the command; it does not download files or generate replacement scenes
or text embeddings.

## Run and inspect

Remove `--check-only` to run the complete supplied panel. The supported tasks are
`move_stapler_pad` and `open_microwave`; conditions are `clean` and `randomized`.
For a short run, add:

```bash
--task move_stapler_pad --condition clean --episodes-per-condition 1
```

A subset is labelled separately in the output. `wam-pwtr-test` provides the same
interface as `test.py`. `--gpu` selects one GPU index or UUID. The finite, positive
`--worker-timeout` defaults to 43,200 seconds per task/condition process. Existing
output directories are rejected.

| Output | Contents |
|---|---|
| `summary.json` | Successes, trials and aggregate/per-condition rates |
| `summary.csv` | Per-task/condition rows and an aggregate row |
| `outcomes.jsonl` | Scene identity, instruction, outcome and action count |
| `preflight.json` | Checkpoint, input and dependency identities |
| `cases.json` | Selected scene/instruction panel |
| `status.json` | Execution state and any failure information |
| `<task>/<condition>/worker.log` | Model and simulator logs |

Success rates are fractions between 0 and 1. Summaries require complete, unique
coverage of the selected cases. Infrastructure errors stop the run and are not
counted as policy failures. No automatic retries, resampling or backfill occur.

## Control protocol

| Setting | Value |
|---|---|
| Full panel | Two tasks × two conditions × 100 scenes; noise ID 0 |
| Scene IDs and instructions | Supplied by the local evaluation bundle |
| Action seed | `environment_seed*1000+replan_index` |
| Transition noise | 0 |
| Action horizon / executed chunk | 32 / up to 24 |
| Denoising steps / precision | 10 / BF16 |
| Action interface | 14-dimensional denormalized joint positions (`qpos`) |
| Robot | Aloha-AgileX |

A new observation is requested at each replanning boundary. The task environment
checks success after each executed action. Method and environment configuration
are in `configs/`; scene selection belongs to the supplied bundle.

The README displays the paper's reported results. Running `test.py` produces
results for the checkpoint, bundle and runtime actually supplied to that run.
