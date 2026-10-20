"""Full-checkpoint PWTR inference for the RoboTwin evaluation loop.

Call ``reset(environment_seed, instruction)`` once per scene. Each ``infer``
returns 32 denormalized joint-position actions; the environment runner executes
at most the first 24, checking task success and the action limit after each step.
"""
from pathlib import Path
from typing import Any, Mapping


class RoboTwinPolicy:
    def __init__(self, adapter, builder, text_loader):
        self.adapter = adapter
        self.builder = builder
        self.text_loader = text_loader
        self.environment_seed = None
        self.instruction = None
        self.replan_index = 0

    def reset(self, environment_seed: int, instruction: str) -> None:
        if type(environment_seed) is not int or environment_seed < 0:
            raise ValueError("environment_seed must be a non-negative integer")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("a non-empty per-scene instruction is required")
        self.environment_seed = environment_seed
        self.instruction = instruction
        self.replan_index = 0

    def infer(self, observation: Mapping[str, Any]):
        import numpy as np

        if self.environment_seed is None:
            raise RuntimeError("reset the policy with the scene seed and instruction before inference")
        condition = self.builder.build(observation, self.instruction)
        condition = self.text_loader.apply(condition)
        # The packaged deterministic Euler sampler has zero transition noise.
        trajectory = self.adapter.collect_student_trajectory(
            condition,
            num_inference_steps=10,
            sigma_shift=None,
            seed=self.environment_seed * 1000 + self.replan_index,
            rand_device="cpu",
            storage_device="cpu",
        )
        actions = np.asarray(self.builder.denormalize_action(trajectory.latents[:, -1]), dtype=np.float32)
        if actions.ndim != 3 or actions.shape[:2] != (1, 32) or not actions.shape[2]:
            raise ValueError("expected one 32-step action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("action chunk must contain only finite values")
        self.replan_index += 1
        return actions[0]


def _load_processor(config, dataset_stats: Path, dataset_stats_bytes=None):
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    processor = instantiate(config.data.train.processor).eval()
    if dataset_stats_bytes is None:
        stats = load_dataset_stats_from_json(str(dataset_stats))
    else:
        # Preserve the external loader's exact conversion rules, reading only
        # our private snapshot of bytes already verified against the manifest.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json') as handle:
            handle.write(dataset_stats_bytes)
            handle.flush()
            stats = load_dataset_stats_from_json(handle.name)
    processor.set_normalizer_from_stats(stats)
    return processor


def _runtime_spec(fastwam_root: Path):
    """Resolve the local VAE using FastWAM's configuration, without downloads."""
    import importlib
    from fastwam_opd.model_loader import compose_fastwam_config, bind_fastwam_source, local_vae_path

    config = compose_fastwam_config(fastwam_root, "sim_robotwin.yaml")
    module_name, _, name = str(config.model._target_).rpartition(".")
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise ValueError("configured FastWAM model factory is not callable")
    # Import the actual backbone dependencies without calling its constructor.
    from fastwam.models.wan22.fastwam import FastWAM
    bind_fastwam_source(fastwam_root)

    if not callable(FastWAM):
        raise ValueError("FastWAM backbone is unavailable")
    return config, local_vae_path(fastwam_root, config.model)


def preflight_policy(fastwam_root, dataset_stats, expected_dataset_stats=None) -> dict[str, Any]:
    """Check backbone imports, VAE content identity and CPU normalization only."""
    import numpy as np
    import torch
    from .robotwin_bridge import RoboTwinConditionBuilder

    root = Path(fastwam_root).expanduser().resolve()
    stats = Path(dataset_stats).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not stats.is_file():
        raise FileNotFoundError(stats)
    config, vae = _runtime_spec(root)
    from fastwam_opd.model_loader import verify_vae_file
    vae_identity = verify_vae_file(vae)
    action_dim = int(config.model.action_dit_config.action_dim)
    proprio_dim = int(config.model.proprio_dim)
    if (action_dim, proprio_dim) != (14, 14):
        raise ValueError("the supplied RoboTwin panel requires 14-dimensional action and state")
    if expected_dataset_stats is None:
        processor = _load_processor(config, stats)
    else:
        from .robotwin_text import verified_asset_bytes
        processor = _load_processor(config, stats, verified_asset_bytes(stats, expected_dataset_stats))
    builder = RoboTwinConditionBuilder(processor)
    state = builder._normalize_state(np.zeros(proprio_dim, dtype=np.float32))
    actions = builder.denormalize_action(torch.zeros(1, 32, action_dim))
    if (tuple(state.shape) != (1, proprio_dim) or actions.shape != (1, 32, action_dim)
            or not bool(torch.isfinite(state).all()) or not np.isfinite(actions).all()):
        raise ValueError("dataset normalizer returned an invalid shape or nonfinite values")
    return dict(status="READY", model_loads=0, environment_episodes=0,
                action_dim=action_dim, proprio_dim=proprio_dim,
                vae_path=str(vae), vae_bytes=vae_identity['size_bytes'],
                vae_sha256=vae_identity['sha256'])


def _load_runtime(*, checkpoint: Path, fastwam_root: Path, dataset_stats: Path, device: str, dataset_stats_bytes=None):
    # External model construction happens only in this factory.
    from fastwam_opd.model_loader import compose_fastwam_config, load_fastwam_model
    from fastwam_opd.fastwam_adapter import FastWAMActionAdapter

    model = load_fastwam_model(
        fastwam_root=fastwam_root,
        config_name="sim_robotwin.yaml",
        checkpoint=checkpoint,
        device=device,
        mixed_precision="bf16",
        load_text_encoder=False,
        strict_full_checkpoint=True,
    ).eval()
    config = compose_fastwam_config(fastwam_root, "sim_robotwin.yaml")
    processor = (_load_processor(config, dataset_stats) if dataset_stats_bytes is None else
                 _load_processor(config, dataset_stats, dataset_stats_bytes))
    return FastWAMActionAdapter(model), processor


def create_policy(*, checkpoint, fastwam_root, dataset_stats, text_cache, device="cuda:0",
                  expected_dataset_stats=None, expected_text_cache=None) -> RoboTwinPolicy:
    """Load the specified model and preprocessing assets, without any teachers."""
    paths = {key: Path(value).expanduser().resolve() for key, value in (
        ("checkpoint", checkpoint), ("fastwam_root", fastwam_root),
        ("dataset_stats", dataset_stats), ("text_cache", text_cache),
    )}
    for key in ("checkpoint", "dataset_stats"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    for key in ("fastwam_root", "text_cache"):
        if not paths[key].is_dir():
            raise FileNotFoundError(paths[key])
    from .robotwin_bridge import RoboTwinConditionBuilder
    from .robotwin_text import CachedTextContextLoader, verified_asset_bytes

    extra = {}
    if expected_dataset_stats is not None:
        extra['dataset_stats_bytes'] = verified_asset_bytes(paths['dataset_stats'], expected_dataset_stats)
    text_loader = CachedTextContextLoader(paths['text_cache'], expected_files=expected_text_cache)

    adapter, processor = _load_runtime(
        checkpoint=paths["checkpoint"], fastwam_root=paths["fastwam_root"],
        dataset_stats=paths["dataset_stats"], device=device, **extra,
    )
    return RoboTwinPolicy(adapter, RoboTwinConditionBuilder(processor), text_loader)


def get_model(usr_args: Mapping[str, Any]) -> RoboTwinPolicy:
    """Factory for callers supplying the same explicit paths as ``create_policy``."""
    return create_policy(
        checkpoint=usr_args["checkpoint"], fastwam_root=usr_args["fastwam_root"],
        dataset_stats=usr_args["dataset_stats"], text_cache=usr_args["text_cache"],
        device=usr_args.get("device", "cuda:0"),
        expected_dataset_stats=usr_args.get('expected_dataset_stats'),
        expected_text_cache=usr_args.get('expected_text_cache'),
    )
