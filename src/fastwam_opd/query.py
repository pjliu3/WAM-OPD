"""Select a low-noise query from the current student's denoising path."""
from __future__ import annotations

import random


def sample_query_index(
    num_steps: int,
    *,
    distribution: str,
    seed: int,
    beta_alpha: float = 5.0,
    beta_beta: float = 2.0,
) -> int:
    """Select one diffusion state without changing the caller's RNG state."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if distribution != "low_noise":
        raise ValueError("PWTR uses the low_noise query distribution")
    if beta_alpha <= 0 or beta_beta <= 0:
        raise ValueError("beta query parameters must be positive")
    generator = random.Random(int(seed))
    coordinate = generator.betavariate(beta_alpha, beta_beta)
    return min(int(coordinate * num_steps), num_steps - 1)
