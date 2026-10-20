"""CPU checks for the immutable VAE required by the FastWAM backbone."""
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch", minversion="2.7")
from omegaconf import OmegaConf

from fastwam_opd import model_loader


@pytest.fixture
def vae_asset(tmp_path, monkeypatch):
    path = tmp_path / "Wan2.2_VAE.safetensors"
    data = b"small fixed VAE asset"
    path.write_bytes(data)
    spec = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(model_loader, "VAE_ASSET", spec, raising=False)
    return path, spec


def test_verified_vae_returns_exact_byte_identity(vae_asset):
    path, spec = vae_asset
    assert model_loader.verify_vae_file(path) == {"path": str(path.resolve()), **spec}


def test_missing_vae_is_rejected(vae_asset):
    path, _ = vae_asset
    path.unlink()
    with pytest.raises(FileNotFoundError, match="VAE"):
        model_loader.verify_vae_file(path)


def test_truncated_vae_is_rejected_before_hashing(vae_asset, monkeypatch):
    path, _ = vae_asset
    path.write_bytes(b"truncated")
    def no_open(*args, **kwargs):
        pytest.fail("a size mismatch should be rejected before reading file contents")
    monkeypatch.setattr(Path, "open", no_open)
    with pytest.raises(ValueError, match="VAE.*size"):
        model_loader.verify_vae_file(path)


def test_same_size_changed_vae_is_rejected(vae_asset):
    path, spec = vae_asset
    path.write_bytes(b"x" * spec["size_bytes"])
    with pytest.raises(ValueError, match="VAE.*SHA256"):
        model_loader.verify_vae_file(path)


@pytest.mark.parametrize("cache_kind", ["relative", "absolute", "literal_tilde"])
def test_vae_path_uses_external_resolver_and_cache_reset(tmp_path, monkeypatch, cache_kind):
    root = tmp_path / "FastWAM"
    cache = {"relative": Path("checkpoints"), "absolute": tmp_path / "assets",
             "literal_tilde": Path("~/assets")}[cache_kind]
    resolved = SimpleNamespace(model_id="converted/wan", origin_file_pattern="Wan2.2_VAE.safetensors")
    calls = []
    def reset_local_model_path():
        calls.append("reset")
        resolved.local_model_path = str(cache)
    resolved.reset_local_model_path = reset_local_model_path
    def resolve_configs(**kwargs):
        calls.append(kwargs)
        return None, None, resolved, None
    external = ModuleType("fastwam.models.wan22.helpers.loader")
    external._resolve_configs = resolve_configs
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setattr(model_loader, "bind_fastwam_source", lambda value: calls.append(("bind", value)))
    config = OmegaConf.create({"model_id": "wan", "tokenizer_model_id": "tokenizer"})
    expected_base = cache if cache.is_absolute() else root / cache
    assert model_loader.local_vae_path(root, config) == (
        expected_base / "converted/wan/Wan2.2_VAE.safetensors"
    ).resolve()
    assert calls == [
        ("bind", root.resolve()),
        {"model_id": "wan", "tokenizer_model_id": "tokenizer", "redirect_common_files": True},
        "reset",
    ]


def test_fastwam_vae_combines_path_resolution_and_full_verification(vae_asset, monkeypatch):
    path, spec = vae_asset
    root, config = path.parent, object()
    calls = []
    def local_vae_path(actual_root, actual_config):
        calls.append((actual_root, actual_config))
        return path
    monkeypatch.setattr(model_loader, "local_vae_path", local_vae_path, raising=False)
    assert model_loader.verify_fastwam_vae(root, config) == {"path": str(path.resolve()), **spec}
    assert calls == [(root, config)]


@pytest.mark.parametrize("failure", ["missing", "size", "content"])
def test_model_construction_cannot_start_with_invalid_vae(vae_asset, monkeypatch, failure):
    path, spec = vae_asset
    checkpoint = path.parent / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint is not loaded in this pre-construction test")
    if failure == "missing":
        path.unlink()
    elif failure == "size":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * spec["size_bytes"])
    monkeypatch.setenv("FASTWAM_OPD_FP32_ROPE", "0")
    monkeypatch.setattr(model_loader, "compose_fastwam_config", lambda *args: OmegaConf.create({"model": {}}))
    monkeypatch.setattr(model_loader, "local_vae_path", lambda *args: path, raising=False)
    def forbidden_construction(*args, **kwargs):
        pytest.fail("model construction must not start with a missing or damaged VAE")
    monkeypatch.setattr(model_loader, "instantiate", forbidden_construction)
    expected_error = FileNotFoundError if failure == "missing" else ValueError
    with pytest.raises(expected_error, match="VAE"):
        model_loader.load_fastwam_model(
            fastwam_root=path.parent, config_name="sim_robotwin.yaml",
            checkpoint=checkpoint, device="cpu", mixed_precision="bf16",
            load_text_encoder=False,
        )
