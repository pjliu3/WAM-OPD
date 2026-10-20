"""Deterministic Euler integration for the deployed action-flow sampler."""
from __future__ import annotations

import torch


def _broadcast_step(delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    delta = torch.as_tensor(delta, device=sample.device, dtype=sample.dtype)
    if delta.ndim == 0:
        return delta
    if delta.ndim != 1:
        raise ValueError(f"delta must be scalar or [B], got shape {tuple(delta.shape)}")
    if delta.shape[0] not in {1, sample.shape[0]}:
        raise ValueError(
            f"delta batch {delta.shape[0]} is incompatible with sample batch {sample.shape[0]}"
        )
    return delta.reshape(-1, *([1] * (sample.ndim - 1)))


def transition_mean(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Deterministic flow-ODE transition mean used by FastWAM's scheduler."""

    if sample.shape != velocity.shape:
        raise ValueError(
            f"sample and velocity must have identical shapes, got {tuple(sample.shape)} "
            f"and {tuple(velocity.shape)}"
        )
    return sample + velocity * _broadcast_step(delta, sample)
