from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DenoisingTrajectory:
    """Action denoising path from a stored collector or the current student.

    `latents[:, j]` is the input to denoising step `j`; the final element is the
    sampled action latent after the last transition.
    """

    latents: torch.Tensor
    timesteps: torch.Tensor
    deltas: torch.Tensor
    transition_stds: torch.Tensor | None = None
    behavior_log_probs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.latents.ndim != 4:
            raise ValueError(
                "latents must be [B, S+1, horizon, action_dim], got "
                f"{tuple(self.latents.shape)}"
            )
        if self.timesteps.ndim != 1 or self.deltas.ndim != 1:
            raise ValueError("timesteps and deltas must both be one-dimensional")
        if self.timesteps.shape != self.deltas.shape:
            raise ValueError("timesteps and deltas must have the same shape")
        if self.latents.shape[1] != self.timesteps.numel() + 1:
            raise ValueError(
                "latents must contain one more state than denoising steps: "
                f"got {self.latents.shape[1]} states and {self.timesteps.numel()} steps"
            )
        if not torch.isfinite(self.latents).all():
            raise ValueError("latents contain non-finite values")
        if not torch.isfinite(self.timesteps).all() or not torch.isfinite(self.deltas).all():
            raise ValueError("schedule contains non-finite values")
        if not torch.all(self.deltas < 0):
            raise ValueError("FastWAM reverse-flow deltas must all be negative")
        stochastic_fields = (
            self.transition_stds is not None,
            self.behavior_log_probs is not None,
        )
        if any(stochastic_fields) and not all(stochastic_fields):
            raise ValueError(
                "stochastic trajectories require both transition_stds and "
                "behavior_log_probs"
            )
        if self.transition_stds is not None:
            if self.transition_stds.ndim != 1:
                raise ValueError("transition_stds must be one-dimensional")
            if self.transition_stds.shape != self.timesteps.shape:
                raise ValueError(
                    "transition_stds must have one value per denoising step"
                )
            if not torch.isfinite(self.transition_stds).all() or not torch.all(
                self.transition_stds > 0
            ):
                raise ValueError("transition_stds must be positive and finite")
            expected_log_probs = (
                self.latents.shape[0],
                self.num_steps,
                *self.latents.shape[2:],
            )
            assert self.behavior_log_probs is not None
            if tuple(self.behavior_log_probs.shape) != expected_log_probs:
                raise ValueError(
                    f"behavior_log_probs must have shape {expected_log_probs}, got "
                    f"{tuple(self.behavior_log_probs.shape)}"
                )
            if not torch.isfinite(self.behavior_log_probs).all():
                raise ValueError("behavior_log_probs contain non-finite values")

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.numel())

    @property
    def is_stochastic(self) -> bool:
        return self.transition_stds is not None


@dataclass(frozen=True)
class TeacherTargets:
    """Serialized transition-mean payload used by the fixed-record format."""

    means: torch.Tensor

    def __post_init__(self) -> None:
        if self.means.ndim != 4:
            raise ValueError(
                "means must be [B, S, horizon, action_dim], got "
                f"{tuple(self.means.shape)}"
            )
        if not torch.isfinite(self.means).all():
            raise ValueError("teacher means contain non-finite values")


@dataclass(frozen=True)
class VelocityTargets:
    """Serialized velocity payload used by the fixed-record format."""

    query_indices: torch.Tensor
    velocities: torch.Tensor

    def __post_init__(self) -> None:
        if self.query_indices.ndim != 1:
            raise ValueError("query_indices must be one-dimensional")
        if self.query_indices.dtype != torch.long:
            raise ValueError("query_indices must use torch.long dtype")
        if self.query_indices.numel() == 0:
            raise ValueError("query_indices must not be empty")
        if self.velocities.ndim != 4:
            raise ValueError(
                "velocities must be [B, K, horizon, action_dim], got "
                f"{tuple(self.velocities.shape)}"
            )
        if self.velocities.shape[1] != self.query_indices.numel():
            raise ValueError(
                "velocity query dimension must match query_indices: "
                f"got {self.velocities.shape[1]} and {self.query_indices.numel()}"
            )
        if not torch.isfinite(self.velocities).all():
            raise ValueError("teacher velocities contain non-finite values")


@dataclass(frozen=True)
class TeacherPolicyTargets:
    """Serialized Gaussian-target payload used by the fixed-record format.

    PWTR obtains fresh teacher targets during training; these stored fields are
    read and validated only to preserve the original record format.
    """

    log_probs: torch.Tensor
    means: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.log_probs.ndim != 4:
            raise ValueError(
                "log_probs must be [B, S, horizon, action_dim], got "
                f"{tuple(self.log_probs.shape)}"
            )
        if not torch.isfinite(self.log_probs).all():
            raise ValueError("teacher log_probs contain non-finite values")
        if self.means is not None:
            if self.means.shape != self.log_probs.shape:
                raise ValueError(
                    "teacher means and log_probs must have identical shapes, got "
                    f"{tuple(self.means.shape)} and {tuple(self.log_probs.shape)}"
                )
            if not torch.isfinite(self.means).all():
                raise ValueError("teacher means contain non-finite values")


@dataclass(frozen=True)
class OPDRecord:
    """A fixed-pool observation, stored denoising path, and archival targets."""

    teacher_name: str
    task_name: str
    condition: Any
    trajectory: DenoisingTrajectory
    targets: TeacherTargets | VelocityTargets | TeacherPolicyTargets

    @property
    def num_supervision_steps(self) -> int:
        if isinstance(self.targets, TeacherTargets):
            return self.trajectory.num_steps
        if isinstance(self.targets, VelocityTargets):
            return int(self.targets.query_indices.numel())
        if isinstance(self.targets, TeacherPolicyTargets):
            return self.trajectory.num_steps
        raise TypeError(f"unsupported teacher targets: {type(self.targets).__name__}")

    def __post_init__(self) -> None:
        if not self.teacher_name.strip():
            raise ValueError("teacher_name must be non-empty")
        if not self.task_name.strip():
            raise ValueError("task_name must be non-empty")
        if isinstance(self.targets, TeacherTargets):
            expected = (
                self.trajectory.latents.shape[0],
                self.trajectory.num_steps,
                *self.trajectory.latents.shape[2:],
            )
            if tuple(self.targets.means.shape) != expected:
                raise ValueError(
                    f"teacher target shape must be {expected}, "
                    f"got {tuple(self.targets.means.shape)}"
                )
            return
        if isinstance(self.targets, VelocityTargets):
            expected = (
                self.trajectory.latents.shape[0],
                self.targets.query_indices.numel(),
                *self.trajectory.latents.shape[2:],
            )
            if tuple(self.targets.velocities.shape) != expected:
                raise ValueError(
                    f"teacher velocity shape must be {expected}, "
                    f"got {tuple(self.targets.velocities.shape)}"
                )
            indices = self.targets.query_indices
            if bool((indices < 0).any()) or bool(
                (indices >= self.trajectory.num_steps).any()
            ):
                raise ValueError(
                    "query_indices must refer to student denoising transitions"
                )
            if torch.unique(indices).numel() != indices.numel():
                raise ValueError("query_indices must not contain duplicates")
            return
        if isinstance(self.targets, TeacherPolicyTargets):
            if not self.trajectory.is_stochastic:
                raise ValueError(
                    "teacher policy targets require a stochastic student trajectory"
                )
            expected = (
                self.trajectory.latents.shape[0],
                self.trajectory.num_steps,
                *self.trajectory.latents.shape[2:],
            )
            if tuple(self.targets.log_probs.shape) != expected:
                raise ValueError(
                    f"teacher log-prob shape must be {expected}, got "
                    f"{tuple(self.targets.log_probs.shape)}"
                )
            if self.targets.means is not None and tuple(self.targets.means.shape) != expected:
                raise ValueError(
                    f"teacher mean shape must be {expected}, got "
                    f"{tuple(self.targets.means.shape)}"
                )
            return
        raise TypeError(f"unsupported teacher targets: {type(self.targets).__name__}")
