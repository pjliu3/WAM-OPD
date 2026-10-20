"""Read fixed-pool records and their serialized target payloads."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .fastwam_adapter import FastWAMConditionInput
from .records import (
    DenoisingTrajectory,
    OPDRecord,
    TeacherPolicyTargets,
    TeacherTargets,
    VelocityTargets,
)


FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = {1, 2, FORMAT_VERSION}


def _condition_from_payload(value: dict[str, Any]) -> FastWAMConditionInput:
    return FastWAMConditionInput(**value)


def _targets_from_payload(
    value: dict[str, Any],
) -> TeacherTargets | VelocityTargets | TeacherPolicyTargets:
    kind = value.get("kind")
    if kind in {None, "transition_means"} and "means" in value:
        return TeacherTargets(means=value["means"])
    if kind == "velocity":
        return VelocityTargets(
            query_indices=value["query_indices"],
            velocities=value["velocities"],
        )
    if kind == "teacher_log_probs":
        return TeacherPolicyTargets(log_probs=value["log_probs"])
    if kind == "teacher_gaussian_policy":
        return TeacherPolicyTargets(
            log_probs=value["log_probs"],
            means=value["means"],
        )
    raise ValueError(f"unsupported OPD target payload kind: {kind!r}")


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_opd_record(path: str | Path) -> OPDRecord:
    payload = _torch_load(path)
    if (
        payload.get("format") != "fastwam_opd_target"
        or payload.get("version") not in SUPPORTED_FORMAT_VERSIONS
    ):
        raise ValueError(f"unsupported OPD target format in {path}")
    return OPDRecord(
        teacher_name=payload["teacher_name"],
        task_name=payload["task_name"],
        condition=_condition_from_payload(payload["condition"]),
        trajectory=DenoisingTrajectory(**payload["trajectory"]),
        targets=_targets_from_payload(payload["targets"]),
    )


def load_opd_metadata(path: str | Path) -> dict[str, Any]:
    payload = _torch_load(path)
    if (
        payload.get("format") != "fastwam_opd_target"
        or payload.get("version") not in SUPPORTED_FORMAT_VERSIONS
    ):
        raise ValueError(f"unsupported OPD target format in {path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"OPD metadata must be a mapping in {path}")
    return dict(metadata)
