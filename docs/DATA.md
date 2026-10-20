# Local training and evaluation bundles

This source release contains implementation code, configurations and paper figures.
Provide metadata, model checkpoints, trajectory records, evaluation scenes and
feature caches in local directories. The commands use explicit paths and do not
download or generate missing input assets.

## Training bundle

Pass this directory with `--training-root`:

```text
training-bundle/
├── assets.json
├── pool.json
├── reference_scores.json
└── schedule.json
```

- `assets.json`: model paths, byte sizes and SHA256 fingerprints, plus hashes of
  the pool manifest and fixed reference scores.
- `pool.json`: initial/teacher checkpoint identities and ordered trajectory
  records, including record IDs, content hashes and executed action lengths.
- `reference_scores.json`: fixed initial-student and task-teacher score maps,
  indexed by record ID.
- `schedule.json`: the fixed 96-update trajectory and decision selection.

Model and trajectory paths in these manifests are relative to `--asset-root`:

```text
assets/
├── models/
│   ├── initial.pt
│   ├── move_stapler_pad.pt
│   ├── open_microwave.pt
│   └── pwtr_u96.pt
└── records/
    └── <record-sha256>.pt
```

Training uses the initial student, the two task teachers and the stored records.
Saved-history inference uses the `pwtr_u96` entry in the local asset manifest.
Supply a bundle matching the [fixed training recipe](TRAINING.md); its schedule,
reference scores and hashes are validated before use.

```bash
wam-pwtr preview --training-root /path/to/training-bundle
wam-pwtr doctor \
  --training-root /path/to/training-bundle \
  --asset-root /path/to/assets --purpose train --hash-files
```

Use `--purpose infer` for saved-history inference inputs, or `--purpose all` for
the full local asset inventory. FastWAM's converted backbone assets belong to its
external runtime and are checked separately.

## Evaluation bundle

Pass this directory with `--evaluation-root`:

```text
evaluation-bundle/
├── evaluation_scenes.json
├── evaluation_instructions.json
├── evaluation_assets.json
├── evaluation_dataset_stats.json
└── expert_cache/
    ├── move_stapler_pad/{demo_clean,demo_randomized}/<seed>.json
    └── open_microwave/{demo_clean,demo_randomized}/<seed>.json
```

The supported full panel contains 100 scenes for each of the two tasks and two
conditions (400 cases), with noise ID 0. The scene manifest supplies those
task/condition seed lists; the instruction manifest supplies the exact text for
each scene. The asset
manifest records expert-cache, text-context and normalization-file identities.
Use matching expert-valid scene caches and dataset statistics.

Text contexts are supplied separately with `--text-cache`. Each file is named
`<prompt-sha256>.t5_len128.wan22ti2v5b.pt` and contains the fixed-length context
and mask expected by FastWAM. These files are verified against the local
`evaluation_assets.json`; missing or mismatched inputs stop evaluation.

See [EVALUATION.md](EVALUATION.md) for preflight and execution commands.
