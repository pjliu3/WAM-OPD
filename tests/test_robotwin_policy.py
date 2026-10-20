"""CPU contracts for observation conversion and closed-loop action generation."""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch", minversion="2.7")
pytest.importorskip("PIL")


PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def modules():
    return (
        importlib.import_module("wam_pwtr.robotwin_policy"),
        importlib.import_module("wam_pwtr.robotwin_bridge"),
        importlib.import_module("wam_pwtr.robotwin_text"),
    )


class Processor:
    shape_meta = {"state": [{"key": "qpos"}], "action": [{"key": "qpos"}]}

    def __init__(self):
        self.normalizer = SimpleNamespace(
            forward=self.normalize,
            normalizers={"action": {"qpos": SimpleNamespace(backward=lambda x: x * 2 + 5)}},
        )

    def action_state_transform(self, batch):
        return {"state": {"qpos": batch["state"]["qpos"] + 2}}

    def normalize(self, batch):
        return {"state": {"qpos": batch["state"]["qpos"] * 3}}


def observation():
    return {
        "observation": {
            "head_camera": {"rgb": np.full((19, 25, 3), [255, 0, 0], dtype=np.uint8)},
            "left_camera": {"rgb": np.full((9, 11, 3), [0, 128, 0], dtype=np.uint8)},
            "right_camera": {"rgb": np.full((8, 13, 3), [0, 0, 64], dtype=np.uint8)},
        },
        "joint_action": {"vector": np.array([1, 2], dtype=np.float32)},
    }


def write_context(root, instruction="move the stapler"):
    prompt = PROMPT.format(task=instruction)
    name = hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
    context = torch.arange(384, dtype=torch.float32).reshape(128, 3)
    mask = torch.arange(128) < 2
    path = root / name
    torch.save({"context": context, "mask": mask}, path)
    return path


def test_three_views_and_state_use_deployment_preprocessing():
    _, bridge, _ = modules()
    builder = bridge.RoboTwinConditionBuilder(Processor())
    value = builder.build(observation(), "move the stapler")
    assert value.action_horizon == 32
    assert value.input_image.shape == (3, 384, 320)
    assert value.input_image.dtype == torch.float32
    torch.testing.assert_close(value.input_image[:, 0, 0], torch.tensor([1., -1., -1.]))
    torch.testing.assert_close(value.input_image[:, 300, 0], torch.tensor([-1., 1 / 255., -1.]))
    torch.testing.assert_close(value.input_image[:, 300, 200], torch.tensor([-1., -1., -127 / 255.]))
    torch.testing.assert_close(value.proprio, torch.tensor([[9., 12.]]))
    assert value.prompt == PROMPT.format(task="move the stapler")


def test_action_denormalization_preserves_all_32_predictions():
    _, bridge, _ = modules()
    builder = bridge.RoboTwinConditionBuilder(Processor())
    actions = builder.denormalize_action(torch.arange(64).reshape(1, 32, 2))
    assert actions.shape == (1, 32, 2)
    np.testing.assert_array_equal(actions[0, 0], [5., 7.])
    np.testing.assert_array_equal(actions[0, -1], [129., 131.])


def test_text_context_zeros_padding_then_uses_all_positions(tmp_path):
    _, _, text = modules()
    path = write_context(tmp_path)
    before = path.read_bytes()
    loader = text.CachedTextContextLoader(tmp_path)
    context, mask = loader.encode(PROMPT.format(task="move the stapler"))
    torch.testing.assert_close(context[:2], torch.tensor([[0., 1., 2.], [3., 4., 5.]]))
    assert torch.count_nonzero(context[2:]) == 0
    assert mask.dtype == torch.bool and bool(mask.all())
    context.fill_(123.)
    second, _ = loader.encode(PROMPT.format(task="move the stapler"))
    assert second[0, 0] == 0
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        loader.encode(PROMPT.format(task="unavailable instruction"))


def asset_spec(path):
    data = path.read_bytes()
    return dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_verified_asset_reads_are_bounded_and_size_is_nonnegative_int(tmp_path, monkeypatch):
    _, _, text = modules()
    path = tmp_path / 'asset.bin'
    path.write_bytes(b'abc')
    expected = asset_spec(path)
    original_open = Path.open
    reads = []
    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def read(self, count=-1):
            assert count == 4, 'verified asset reads must be capped at expected size + 1'
            reads.append(count)
            return self.handle.read(count)
    def tracked_open(self, mode='r', *args, **kwargs):
        handle = original_open(self, mode, *args, **kwargs)
        return BoundedReader(handle) if self == path and mode == 'rb' else handle
    monkeypatch.setattr(Path, 'open', tracked_open)
    assert text.verified_asset_bytes(path, expected) == b'abc'
    path.write_bytes(b'abc' + b'x' * 100)
    with pytest.raises(ValueError, match='asset identity'):
        text.verified_asset_bytes(path, expected)
    assert reads == [4, 4]
    for size in [-2, -1, True, 3.0, None]:
        with pytest.raises(ValueError, match='non-negative integer'):
            text.verified_asset_bytes(path, dict(expected, size_bytes=size))
    assert reads == [4, 4]


def test_bound_text_cache_rejects_replacement_before_read(tmp_path):
    _, _, text = modules()
    path = write_context(tmp_path)
    expected = asset_spec(path)
    loader = text.CachedTextContextLoader(tmp_path, expected_files={path.name: expected})
    torch.save({'context': torch.ones(128, 3), 'mask': torch.ones(128, dtype=torch.bool)}, path)
    with pytest.raises(ValueError, match='asset identity'):
        loader.encode(PROMPT.format(task='move the stapler'))


def test_bound_text_load_uses_verified_bytes_when_path_changes(tmp_path, monkeypatch):
    _, _, text = modules()
    path = write_context(tmp_path)
    loader = text.CachedTextContextLoader(tmp_path, expected_files={path.name: asset_spec(path)})
    original_load = torch.load
    def replace_during_load(source, **kwargs):
        torch.save({'context': torch.full((128, 3), 99.), 'mask': torch.ones(128, dtype=torch.bool)}, path)
        return original_load(source, **kwargs)
    monkeypatch.setattr(torch, 'load', replace_during_load)
    context, _ = loader.encode(PROMPT.format(task='move the stapler'))
    torch.testing.assert_close(context[:2], torch.tensor([[0., 1., 2.], [3., 4., 5.]]))


def test_factory_pins_stats_bytes_before_loading_model(tmp_path, monkeypatch):
    policy, _, _ = modules()
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'CPU fixture')
    stats = tmp_path / 'stats.json'; stats.write_text('{"scale": [1, 2]}')
    expected = asset_spec(stats)
    seen = []
    def load(**kwargs):
        stats.write_text('{"scale": [99, 99]}')
        seen.append(json.loads(kwargs['dataset_stats_bytes']))
        return Adapter(), Processor()
    monkeypatch.setattr(policy, '_load_runtime', load)
    policy.create_policy(checkpoint=checkpoint, fastwam_root=tmp_path, dataset_stats=stats,
                         text_cache=tmp_path, expected_dataset_stats=expected)
    assert seen == [{'scale': [1, 2]}]
    with pytest.raises(ValueError, match='asset identity'):
        policy.create_policy(checkpoint=checkpoint, fastwam_root=tmp_path, dataset_stats=stats,
                             text_cache=tmp_path, expected_dataset_stats=expected)
    assert len(seen) == 1


def test_fastwam_binding_promotes_requested_tree_and_rejects_preloaded_other_tree(tmp_path, monkeypatch):
    pytest.importorskip('hydra')
    from fastwam_opd.model_loader import bind_fastwam_source
    from types import ModuleType
    root = tmp_path / 'requested'
    source = root / 'src'
    (source / 'fastwam').mkdir(parents=True)
    (source / 'fastwam/__init__.py').write_text("SOURCE = 'requested'\n")
    other = tmp_path / 'other'
    (other / 'fastwam').mkdir(parents=True)
    (other / 'fastwam/__init__.py').write_text("SOURCE = 'other'\n")
    # Keep pre-existing imports isolated from this synthetic source-tree test.
    for name in list(sys.modules):
        if name == 'fastwam' or name.startswith('fastwam.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, 'path', [str(other), str(source), *sys.path])
    bind_fastwam_source(root)
    assert sys.path[:2] == [str(source), str(root)]
    try:
        loaded = importlib.import_module('fastwam')
        assert loaded.SOURCE == 'requested'
        assert Path(loaded.__file__).resolve().is_relative_to(source)
    finally:
        sys.modules.pop('fastwam', None)  # Remove only this test's temporary package.
    stale = ModuleType('fastwam.old')
    stale.__file__ = str(other / 'fastwam/old.py')
    monkeypatch.setitem(sys.modules, 'fastwam.old', stale)
    with pytest.raises(ValueError, match='already loaded from'):
        bind_fastwam_source(root)
    assert sys.modules['fastwam.old'] is stale


class Adapter:
    """Replace only the expensive neural network in the policy behavior tests."""
    def __init__(self):
        self.calls = []

    def collect_student_trajectory(self, condition, **kwargs):
        self.calls.append((condition, kwargs))
        # Different path positions catch selecting an intermediate state.
        latents = torch.zeros((1, 11, 32, 2))
        latents[:, -1] = torch.arange(64).reshape(1, 32, 2) + kwargs["seed"] % 10
        return SimpleNamespace(latents=latents)


def make_policy(tmp_path):
    policy, bridge, text = modules()
    write_context(tmp_path)
    adapter = Adapter()
    obj = policy.RoboTwinPolicy(adapter, bridge.RoboTwinConditionBuilder(Processor()), text.CachedTextContextLoader(tmp_path))
    return obj, adapter


def test_policy_seeds_follow_scene_and_replan_and_reset(tmp_path):
    obj, adapter = make_policy(tmp_path)
    with pytest.raises(RuntimeError, match="reset"):
        obj.infer(observation())
    obj.reset(100000, "move the stapler")
    first = obj.infer(observation())
    second = obj.infer(observation())
    obj.reset(100002, "move the stapler")
    third = obj.infer(observation())
    assert first.dtype == np.float32 and first.shape == (32, 2)
    np.testing.assert_array_equal(first[0], [5., 7.])
    np.testing.assert_array_equal(first[-1], [129., 131.])
    np.testing.assert_array_equal(second[0], [7., 9.])
    np.testing.assert_array_equal(third, first)
    assert [call[1]["seed"] for call in adapter.calls] == [100000000, 100000001, 100002000]
    assert adapter.calls[0][1] == {
        "num_inference_steps": 10, "sigma_shift": None,
        "seed": 100000000, "rand_device": "cpu", "storage_device": "cpu",
    }
    condition = adapter.calls[0][0]
    assert condition.prompt is None and condition.context.shape == (128, 3)
    assert bool(condition.context_mask.all())


@pytest.mark.parametrize("seed,instruction", [(True, "move the stapler"), (-1, "move the stapler"), (1, "")])
def test_reset_rejects_ambiguous_scene_or_instruction(tmp_path, seed, instruction):
    obj, _ = make_policy(tmp_path)
    with pytest.raises(ValueError):
        obj.reset(seed, instruction)


def test_nonfinite_action_does_not_advance_replan(tmp_path):
    obj, adapter = make_policy(tmp_path)
    obj.reset(100000, "move the stapler")
    original = adapter.collect_student_trajectory
    def invalid(condition, **kwargs):
        result = original(condition, **kwargs)
        result.latents[:, -1, 0, 0] = float("nan")
        return result
    adapter.collect_student_trajectory = invalid
    with pytest.raises(ValueError, match="finite"):
        obj.infer(observation())
    adapter.collect_student_trajectory = original
    obj.infer(observation())
    assert [call[1]["seed"] for call in adapter.calls] == [100000000, 100000000]


def test_factory_requires_explicit_assets_before_model_load(tmp_path, monkeypatch):
    policy, _, _ = modules()
    def forbidden(**kwargs):
        pytest.fail("model load must not happen for missing assets")
    monkeypatch.setattr(policy, "_load_runtime", forbidden)
    with pytest.raises(FileNotFoundError):
        policy.create_policy(checkpoint=tmp_path / "missing.pt", fastwam_root=tmp_path,
                             dataset_stats=tmp_path / "stats.json", text_cache=tmp_path)


def test_factory_uses_only_explicit_checkpoint_and_normalizer(tmp_path, monkeypatch):
    policy, _, _ = modules()
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"test only")
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    write_context(tmp_path)
    observed = []
    def load(**kwargs):
        observed.append(kwargs)
        return Adapter(), Processor()
    monkeypatch.setattr(policy, "_load_runtime", load)
    obj = policy.get_model(dict(checkpoint=str(checkpoint), fastwam_root=str(tmp_path),
                                dataset_stats=str(stats), text_cache=str(tmp_path), device="cpu"))
    obj.reset(100000, "move the stapler")
    assert obj.infer(observation()).shape == (32, 2)
    assert observed == [dict(checkpoint=checkpoint, fastwam_root=tmp_path,
                             dataset_stats=stats, device="cpu")]


def test_preflight_checks_vae_and_normalizer_without_loading_model(tmp_path, monkeypatch):
    policy, _, _ = modules()
    from fastwam_opd import model_loader
    contents = b"verified VAE fixture"
    expected_sha = hashlib.sha256(contents).hexdigest()
    monkeypatch.setattr(model_loader, "VAE_ASSET",
                        {"size_bytes": len(contents), "sha256": expected_sha}, raising=False)
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    vae = tmp_path / "Wan2.2_VAE.safetensors"
    config = SimpleNamespace(model=SimpleNamespace(proprio_dim=14,
                             action_dit_config=SimpleNamespace(action_dim=14)))
    monkeypatch.setattr(policy, "_runtime_spec", lambda root: (config, vae), raising=False)
    monkeypatch.setattr(policy, "_load_processor", lambda cfg, path: Processor(), raising=False)
    def forbidden(**kwargs):
        pytest.fail("preflight must not construct a model")
    monkeypatch.setattr(policy, "_load_runtime", forbidden)
    with pytest.raises(FileNotFoundError, match="VAE"):
        policy.preflight_policy(tmp_path, stats)
    vae.write_bytes(contents)
    result = policy.preflight_policy(tmp_path, stats)
    assert result["model_loads"] == 0 and result["environment_episodes"] == 0
    assert result["action_dim"] == result["proprio_dim"] == 14
    assert result["vae_path"] == str(vae)
    assert result["vae_sha256"] == expected_sha


def test_preflight_rejects_changed_vae_content_before_processor_loading(tmp_path, monkeypatch):
    policy, _, _ = modules()
    from fastwam_opd import model_loader
    original = b"known VAE content"
    monkeypatch.setattr(model_loader, "VAE_ASSET",
                        {"size_bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()},
                        raising=False)
    vae = tmp_path / "Wan2.2_VAE.safetensors"
    vae.write_bytes(b"x" * len(original))
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    config = SimpleNamespace(model=SimpleNamespace(proprio_dim=14,
                             action_dit_config=SimpleNamespace(action_dim=14)))
    monkeypatch.setattr(policy, "_runtime_spec", lambda root: (config, vae))
    calls = []
    monkeypatch.setattr(policy, "_load_processor",
                        lambda *args: calls.append("processor") or Processor())
    with pytest.raises(ValueError, match="VAE.*SHA256"):
        policy.preflight_policy(tmp_path, stats)
    assert not calls


def test_preflight_rejects_invalid_normalized_state(tmp_path, monkeypatch):
    policy, _, _ = modules()
    from fastwam_opd import model_loader
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    vae = tmp_path / "Wan2.2_VAE.safetensors"
    vae.write_bytes(b"preflight fixture")
    monkeypatch.setattr(model_loader, "VAE_ASSET",
                        {"size_bytes": vae.stat().st_size,
                         "sha256": hashlib.sha256(vae.read_bytes()).hexdigest()}, raising=False)
    config = SimpleNamespace(model=SimpleNamespace(proprio_dim=14,
                             action_dit_config=SimpleNamespace(action_dim=14)))
    monkeypatch.setattr(policy, "_runtime_spec", lambda root: (config, vae), raising=False)
    processor = Processor()
    processor.normalizer.forward = lambda batch: {"state": {"qpos": torch.full((1, 14), float("nan"))}}
    monkeypatch.setattr(policy, "_load_processor", lambda cfg, path: processor, raising=False)
    with pytest.raises(ValueError, match="normalizer"):
        policy.preflight_policy(tmp_path, stats)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
        self.proprio_encoder = torch.nn.Linear(2, 3)
        self.legacy_loads = []

    def load_checkpoint(self, path):
        self.legacy_loads.append(path)


def full_payload(model):
    return {
        "mot": {k: torch.full_like(v, 3) for k, v in model.mot.state_dict().items()},
        "proprio_encoder": {k: torch.full_like(v, 7) for k, v in model.proprio_encoder.state_dict().items()},
        "step": 96,
        "torch_dtype": "torch.float32",
    }


def load_tiny(tmp_path, monkeypatch, payload, *, strict=True):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from fastwam_opd import model_loader
    model = TinyModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint = tmp_path / "weights.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(model_loader, "compose_fastwam_config", lambda *_: OmegaConf.create({"model": {}}))
    monkeypatch.setattr(model_loader, "verify_fastwam_vae", lambda *args: {}, raising=False)
    monkeypatch.setattr(model_loader, "instantiate", lambda *a, **kw: model)
    kwargs = dict(fastwam_root=tmp_path, config_name="sim_robotwin.yaml", checkpoint=checkpoint,
                  device="cpu", mixed_precision="bf16", load_text_encoder=False)
    if strict:
        kwargs["strict_full_checkpoint"] = True
    return model, before, lambda: model_loader.load_fastwam_model(**kwargs)


def test_strict_full_checkpoint_loads_both_complete_modules(tmp_path, monkeypatch):
    payload = full_payload(TinyModel())
    model, _, load = load_tiny(tmp_path, monkeypatch, payload)
    assert load() is model
    assert not model.legacy_loads
    for key, tensor in model.mot.state_dict().items():
        assert torch.equal(tensor, payload["mot"][key])
    for key, tensor in model.proprio_encoder.state_dict().items():
        assert torch.equal(tensor, payload["proprio_encoder"][key])


@pytest.mark.parametrize("damage", ["mot_missing", "mot_extra", "mot_shape", "proprio_missing",
                                    "proprio_shape", "missing_proprio", "non_tensor", "dit_only",
                                    "base_checkpoint", "action_lora", "joint_lora", "sparse_delta"])
def test_strict_full_checkpoint_rejects_partial_and_redirected_payloads(tmp_path, monkeypatch, damage):
    payload = full_payload(TinyModel())
    if damage == "mot_missing":
        del payload["mot"]["0.weight"]
    elif damage == "mot_extra":
        payload["mot"]["unexpected.weight"] = torch.zeros(2)
    elif damage == "mot_shape":
        payload["mot"]["0.weight"] = torch.zeros(2, 4)
    elif damage == "proprio_missing":
        del payload["proprio_encoder"]["weight"]
    elif damage == "proprio_shape":
        payload["proprio_encoder"]["bias"] = torch.zeros(4)
    elif damage == "missing_proprio":
        del payload["proprio_encoder"]
    elif damage == "non_tensor":
        payload["mot"]["0.weight"] = "not a tensor"
    elif damage == "dit_only":
        payload = {"dit": payload["mot"]}
    else:
        payload[damage] = "another-checkpoint.pt"
    model, before, load = load_tiny(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError):
        load()
    assert not model.legacy_loads
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_training_loader_default_keeps_external_load_behavior(tmp_path, monkeypatch):
    model, before, load = load_tiny(tmp_path, monkeypatch, {"legacy_format": "external loader owns this"}, strict=False)
    assert load() is model
    assert model.legacy_loads == [str(tmp_path / "weights.pt")]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_evaluation_factory_requires_strict_full_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    from fastwam_opd import model_loader, fastwam_adapter
    policy, _, _ = modules()
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return TinyModel()
    monkeypatch.setattr(model_loader, "load_fastwam_model", load)
    monkeypatch.setattr(model_loader, "compose_fastwam_config", lambda *_: object())
    monkeypatch.setattr(fastwam_adapter, "FastWAMActionAdapter", lambda _: Adapter())
    monkeypatch.setattr(policy, "_load_processor", lambda *a: Processor())
    policy._load_runtime(checkpoint=tmp_path / "weights.pt", fastwam_root=tmp_path,
                         dataset_stats=tmp_path / "stats.json", device="cpu")
    assert calls[0]["strict_full_checkpoint"] is True
    assert calls[0]["mixed_precision"] == "bf16" and calls[0]["load_text_encoder"] is False
