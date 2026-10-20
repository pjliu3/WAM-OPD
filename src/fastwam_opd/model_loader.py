"""Construct the FastWAM backbone and load full model weights."""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


# Converted VAE bytes verified in the reference runtime snapshot.
VAE_ASSET = {
    "filename": "Wan2.2_VAE.safetensors",
    "size_bytes": 1409401152,
    "sha256": "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996",
}


def precision_dtype(name: str) -> torch.dtype:
    values = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported mixed precision {name!r}") from exc


def bind_fastwam_source(fastwam_root: str | Path):
    """Select one source tree; never mix previously imported FastWAM modules."""
    root = Path(fastwam_root).resolve()
    src = root / "src"
    package = (src / 'fastwam').resolve()
    for name, module in list(sys.modules.items()):
        if name != 'fastwam' and not name.startswith('fastwam.'):
            continue
        origin = getattr(module, '__file__', None)
        paths = [origin] if origin else list(getattr(module, '__path__', ()))
        if not paths or any(not Path(path).resolve().is_relative_to(package) for path in paths):
            raise ValueError(f'FastWAM module {name} already loaded from {origin or paths}; requested {package}')
    selected = (str(src), str(root))
    sys.path[:] = [*selected, *(entry for entry in sys.path if entry not in selected)]


def compose_fastwam_config(fastwam_root: str | Path, config_name: str):
    root = Path(fastwam_root).resolve()
    bind_fastwam_source(root)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        return compose(config_name=config_name)


def local_vae_path(fastwam_root: str | Path, model_config) -> Path:
    """Resolve the VAE cache path exactly as FastWAM does, without downloading."""
    root = Path(fastwam_root).resolve()
    bind_fastwam_source(root)
    from fastwam.models.wan22.helpers.loader import _resolve_configs

    _, _, vae, _ = _resolve_configs(
        model_id=model_config.model_id,
        tokenizer_model_id=model_config.tokenizer_model_id,
        redirect_common_files=model_config.get("redirect_common_files", True),
    )
    vae.reset_local_model_path()
    # Preserve the external loader's literal path semantics, including "~".
    base = Path(vae.local_model_path)
    if not base.is_absolute():
        base = root / base
    return (base / vae.model_id / vae.origin_file_pattern).resolve()


def verify_vae_file(path: str | Path) -> dict:
    """Require the complete reference VAE bytes before constructing a model."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VAE asset is missing: {path}")
    size = path.stat().st_size
    if size != VAE_ASSET["size_bytes"]:
        raise ValueError(
            f"VAE size mismatch: expected {VAE_ASSET['size_bytes']}, got {size}: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != VAE_ASSET["sha256"]:
        raise ValueError(
            f"VAE SHA256 mismatch: expected {VAE_ASSET['sha256']}, got {actual_sha256}: {path}"
        )
    return {"path": str(path), "size_bytes": size, "sha256": actual_sha256}


def verify_fastwam_vae(fastwam_root: str | Path, model_config) -> dict:
    """Resolve and verify the local VAE required by a FastWAM configuration."""
    return verify_vae_file(local_vae_path(fastwam_root, model_config))


def read_full_checkpoint(path: str | Path):
    """Read a self-contained inference export without following checkpoint links.

    Tensor storage is memory mapped on CPU. Only the two complete module states
    and the export's optional step/dtype metadata are accepted.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"mot", "proprio_encoder"}
    allowed = required | {"step", "torch_dtype"}
    if not isinstance(payload, Mapping) or not required <= payload.keys():
        raise ValueError("full checkpoint requires mot and proprio_encoder state dictionaries")
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f"unsupported full-checkpoint fields: {sorted(map(str, unexpected))}")
    for name in required:
        state = payload[name]
        if (not isinstance(state, Mapping) or not state
                or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                       for key, value in state.items())):
            raise ValueError(f"full checkpoint {name} must be a non-empty tensor state dictionary")
    return payload


def _load_full_checkpoint(model, payload) -> None:
    """Validate both modules completely before copying any checkpoint tensors."""
    modules = {}
    for name in ("mot", "proprio_encoder"):
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"model lacks required full-checkpoint module {name}")
        expected = module.state_dict()
        supplied = payload[name]
        if set(expected) != set(supplied):
            missing = sorted(set(expected) - set(supplied))
            extra = sorted(set(supplied) - set(expected))
            raise ValueError(f"full checkpoint {name} keys differ: missing={missing[:5]}, extra={extra[:5]}")
        for key, tensor in expected.items():
            if tensor.shape != supplied[key].shape:
                raise ValueError(
                    f"full checkpoint {name}.{key} shape differs: "
                    f"expected {tuple(tensor.shape)}, got {tuple(supplied[key].shape)}"
                )
        modules[name] = module
    for name, module in modules.items():
        module.load_state_dict(payload[name], strict=True)


def load_fastwam_model(
    *,
    fastwam_root: str | Path,
    config_name: str,
    checkpoint: str | Path,
    device: str,
    mixed_precision: str,
    load_text_encoder: bool,
    action_lora_rank: int = 0,
    strict_full_checkpoint: bool = False,
):
    """Load a full FastWAM checkpoint with the PWTR model configuration."""
    if type(action_lora_rank) is not int or action_lora_rank != 0:
        raise ValueError("PWTR requires action_lora_rank=0")
    if os.environ.get("FASTWAM_OPD_FP32_ROPE", "0") != "0":
        raise ValueError("PWTR requires FASTWAM_OPD_FP32_ROPE=0")
    load_path = Path(checkpoint).expanduser().resolve()
    if not load_path.is_file():
        raise FileNotFoundError(load_path)
    payload = read_full_checkpoint(load_path) if strict_full_checkpoint else None

    root = Path(fastwam_root).resolve()
    cfg = compose_fastwam_config(root, config_name)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    model_cfg.mot_checkpoint_mixed_attn = False
    model_cfg.action_lora_rank = 0
    model_cfg.action_lora_alpha = None
    model_cfg.action_lora_dropout = 0.0
    model_cfg.joint_lora_rank = 0
    verify_fastwam_vae(root, model_cfg)
    previous_cwd = Path.cwd()
    try:
        # FastWAM resolves converted Wan assets through ./checkpoints.
        os.chdir(root)
        model = instantiate(
            model_cfg,
            model_dtype=precision_dtype(mixed_precision),
            device=device,
        )
        if strict_full_checkpoint:
            _load_full_checkpoint(model, payload)
        else:
            model.load_checkpoint(str(load_path))
    finally:
        os.chdir(previous_cwd)
    return model.to(device)
