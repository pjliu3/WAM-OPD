"""Score stored denoising paths on their executed action coordinates.

The caller owns model eval/train mode, snapshot version and RNG preservation.
Fresh teacher supervision is evaluated separately by the training backend.
"""
import math
from numbers import Integral


def score_record_paths(adapter, records, executed_prefix_steps, *, sigma_scale=0.025, min_sigma=1.e-4, return_details=False):
    """Return ordered scalar scores or per-record diagnostic dictionaries.

    Each record must contain one [1, S, H, D] stored trajectory, with a common
    S/H/D across this batch. Its timesteps, negative deltas and actual executed
    action length can differ from every other record. A scalar integer prefix
    broadcasts; otherwise provide one strict integer per record.

    Conditions are prepared once under no-grad. Every diffusion transition uses
    one [B,H,D] prediction. Residual arithmetic uses FP32;
    FP64 reductions and validity flags stay on adapter.device until one final
    small [B,4] host transfer. Nonfinite inputs/predictions fail the whole batch.
    """
    import torch

    if (type(sigma_scale) not in (int, float) or not math.isfinite(float(sigma_scale)) or float(sigma_scale) <= 0
            or type(min_sigma) not in (int, float) or not math.isfinite(float(min_sigma)) or float(min_sigma) <= 0):
        raise ValueError('history scorer kernel must have finite positive sigma_scale and min_sigma')
    records = list(records)
    if isinstance(executed_prefix_steps, Integral) and not isinstance(executed_prefix_steps, bool):
        prefixes = [int(executed_prefix_steps)] * len(records)
    else:
        try:
            prefixes = list(executed_prefix_steps)
        except TypeError as exc:
            raise ValueError('executed prefixes must be integers, one per record') from exc
    if len(prefixes) != len(records) or any(
        not isinstance(p, Integral) or isinstance(p, bool) for p in prefixes
    ):
        raise ValueError('executed prefixes must be integers, one per record')
    if not records:
        return []

    latents_parts, delta_parts, time_parts = [], [], []
    expected_shape = None
    for record, prefix in zip(records, prefixes):
        trajectory = record.trajectory
        tensors = (trajectory.latents, trajectory.deltas, trajectory.timesteps)
        if any(not isinstance(t, torch.Tensor) or t.is_complex() for t in tensors):
            raise ValueError('stored paths and schedules must be real tensors')
        latents, deltas, timesteps = tensors
        shape = tuple(latents.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] < 2 or min(shape[2:]) <= 0:
            raise ValueError('history scoring requires nonempty single-record [1,S,H,D] paths')
        if expected_shape is not None and shape != expected_shape:
            raise ValueError('history scoring requires a same-shape path batch')
        expected_shape = shape
        steps = shape[1] - 1
        if not 0 < prefix <= shape[2]:
            raise ValueError('invalid actual executed action prefix')
        if deltas.numel() != steps or timesteps.numel() < steps:
            raise ValueError('history trajectory schedule shape mismatch')
        # Compute stored-path residuals and deltas in FP32.
        latents_parts.append(latents.detach().to(device=adapter.device, dtype=torch.float32))
        delta_parts.append(deltas.detach().reshape(-1).to(device=adapter.device, dtype=torch.float32))
        time_parts.append(timesteps.detach().reshape(-1)[:steps].to(
            device=adapter.device, dtype=adapter.dtype))

    with torch.no_grad():
        latents = torch.cat(latents_parts, dim=0)
        deltas = torch.stack(delta_parts)
        timesteps = torch.stack(time_parts)
        batch, _, horizon, action_dim = latents.shape
        prefix_tensor = torch.tensor(prefixes, device=adapter.device, dtype=torch.long)
        mask = torch.arange(horizon, device=adapter.device)[None, :, None] < prefix_tensor[:, None, None]
        dimensions = prefix_tensor.to(dtype=torch.float64) * action_dim
        valid = (torch.isfinite(latents).flatten(1).all(1)
                 & torch.isfinite(deltas).all(1) & (deltas < 0).all(1)
                 & torch.isfinite(timesteps).all(1))
        score = torch.zeros(batch, dtype=torch.float64, device=adapter.device)
        residual_squares = torch.zeros_like(score)
        velocity_squares = torch.zeros_like(score)
        prepared = adapter.prepare_conditions([record.condition for record in records])
        for index in range(steps):
            velocity = adapter.predict_velocity(
                latents_action=latents[:, index].to(dtype=adapter.dtype),
                timestep_action=timesteps[:, index], condition=prepared,
            )
            if not isinstance(velocity, torch.Tensor) or tuple(velocity.shape) != (batch, horizon, action_dim):
                raise ValueError('history student velocity shape mismatch')
            if velocity.is_complex() or velocity.device != latents.device:
                raise ValueError('history student velocity dtype/device mismatch')
            velocity = velocity.detach().float()
            valid = valid & torch.isfinite(velocity).flatten(1).all(1)
            dt = deltas[:, index]
            residual = latents[:, index + 1] - latents[:, index] - dt[:, None, None] * velocity
            # Mask before squaring, so finite unused large coordinates cannot
            # overflow a reduction or contribute to an actually executed path.
            squared_sum = residual.masked_fill(~mask, 0).double().square().sum((1, 2))
            dt64 = dt.double()
            # Invalid deltas remain marked invalid; safe sqrt postpones their
            # rejection to the same final host transfer without a step sync.
            sigma = (float(sigma_scale) * (-dt64).clamp_min(0).sqrt()).clamp_min(float(min_sigma))
            score = score - (squared_sum / dimensions) / (2 * sigma.square())
            residual_squares = residual_squares + squared_sum
            velocity_squares = velocity_squares + squared_sum / dt64.square()
        statistics = torch.stack((score,
            (residual_squares / (dimensions * steps)).sqrt(),
            (velocity_squares / (dimensions * steps)).sqrt(),
            valid.to(dtype=torch.float64)), dim=1).cpu().tolist()

    if any(row[3] != 1.0 or not all(math.isfinite(value) for value in row[:3]) for row in statistics):
        raise ValueError('nonfinite history path score/input or nonnegative trajectory delta')
    if return_details:
        return [dict(log_score=row[0], transition_residual_rms=row[1],
                     velocity_residual_rms=row[2], executed_action_steps=int(prefix),
                     diffusion_steps=steps) for row, prefix in zip(statistics, prefixes)]
    return [row[0] for row in statistics]
