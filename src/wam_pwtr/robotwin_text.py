"""Read frozen T5 contexts with the FastWAM deployment padding convention."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import io
from pathlib import Path

import torch


def verified_asset_bytes(path, expected):
    """Hash the exact immutable bytes later supplied to the decoder."""
    size = expected['size_bytes']
    if type(size) is not int or size < 0:
        raise ValueError('evaluation asset size must be a non-negative integer')
    path = Path(path)
    with path.open('rb') as handle:
        data = handle.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError(f'evaluation asset identity changed: {path}')
    return data


class CachedTextContextLoader:
    def __init__(self, cache_dir: str | Path, *, cache_size: int = 8, expected_files=None):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(self.cache_dir)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.expected_files = None if expected_files is None else {name: dict(spec) for name, spec in expected_files.items()}

    def path_for_prompt(self, prompt: str) -> Path:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path_for_prompt(prompt)
        if path in self._cache:
            self._cache.move_to_end(path)
            context, mask = self._cache[path]
            return context.clone(), mask.clone()
        if self.expected_files is None:
            source = path
        else:
            if path.name not in self.expected_files:
                raise ValueError(f'text context is outside the bound asset inventory: {path}')
            source = io.BytesIO(verified_asset_bytes(path, self.expected_files[path.name]))
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not {"context", "mask"} <= payload.keys():
            raise ValueError(f"invalid text context payload: {path}")
        context = payload["context"].detach().clone().cpu()
        mask = payload["mask"].detach().to(device="cpu", dtype=torch.bool).clone()
        if context.ndim != 2 or context.shape[0] != 128 or mask.shape != (128,):
            raise ValueError(f"expected context [128,D] and mask [128]: {path}")
        # Match the reference training and deployment: zero padded vectors,
        # then attend to all positions in the fixed-length context.
        context[~mask] = 0.0
        mask = torch.ones_like(mask)
        if self.cache_size:
            self._cache[path] = (context.clone(), mask.clone())
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return context, mask

    def apply(self, condition):
        if condition.prompt is None:
            raise ValueError("a raw prompt is required to load the frozen text context")
        context, mask = self.encode(condition.prompt)
        return replace(condition, prompt=None, context=context, context_mask=mask)
