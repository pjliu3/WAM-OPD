"""FastWAM action-flow sampling and differentiable PWTR conditioning."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Sequence

import torch

from .loss import transition_mean
from .records import DenoisingTrajectory


@dataclass(frozen=True)
class FastWAMConditionInput:
    """Serializable policy input before model-specific condition encoding."""

    input_image: torch.Tensor
    action_horizon: int
    prompt: str | None = None
    proprio: torch.Tensor | None = None
    context: torch.Tensor | None = None
    context_mask: torch.Tensor | None = None
    tiled: bool = False

    def __post_init__(self) -> None:
        image = self.input_image
        if image.ndim == 3:
            channels, height, width = image.shape
        elif image.ndim == 4 and image.shape[0] == 1:
            _, channels, height, width = image.shape
        else:
            raise ValueError(
                "input_image must be [3,H,W] or [1,3,H,W], got " f"{tuple(image.shape)}"
            )
        if channels != 3:
            raise ValueError(f"input_image must have three channels, got {channels}")
        if height % 16 or width % 16:
            raise ValueError(
                f"image size must be divisible by 16, got {height}x{width}"
            )
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        uses_prompt = self.prompt is not None
        uses_context = self.context is not None or self.context_mask is not None
        if uses_prompt == uses_context:
            raise ValueError(
                "provide exactly one of prompt or the context/context_mask pair"
            )
        if uses_context and (self.context is None or self.context_mask is None):
            raise ValueError("context and context_mask must be provided together")


@dataclass(frozen=True)
class PreparedFastWAMCondition:
    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, torch.Tensor]]
    attention_mask: torch.Tensor
    video_seq_len: int
    action_horizon: int


@dataclass(frozen=True)
class _EncodedCondition:
    context: torch.Tensor
    context_mask: torch.Tensor
    first_frame_latents: torch.Tensor | None
    action_horizon: int
    fuse_vae_embedding_in_latents: bool


@dataclass(frozen=True)
class _StaticCondition:
    """Frozen condition state reusable across independent diffusion losses."""

    context: torch.Tensor
    context_mask: torch.Tensor
    proprio: torch.Tensor | None
    first_frame_latents: torch.Tensor | None
    action_horizon: int
    fuse_vae_embedding_in_latents: bool


@dataclass(frozen=True)
class OnlineFlowQuery:
    """Detached query states from one deployed action-flow rollout.

    Query rows may contain one or several diffusion points per context. The
    intermediate solver graph is never retained; every state is stop-gradient.
    """

    query_latents: torch.Tensor
    query_timesteps: torch.Tensor
    query_indices: torch.Tensor
    final_latents: torch.Tensor
    query_sample_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.query_latents.ndim != 3:
            raise ValueError("query_latents must be [Q,H,D]")
        if self.final_latents.ndim != 3:
            raise ValueError("final_latents must be [B,H,D]")
        if self.final_latents.shape[1:] != self.query_latents.shape[1:]:
            raise ValueError("final and query action latent shapes must align")
        query_count = self.query_latents.shape[0]
        batch = self.final_latents.shape[0]
        if batch == 0 or query_count == 0:
            raise ValueError("online action query batches must not be empty")
        if self.query_timesteps.shape != (query_count,):
            raise ValueError("query_timesteps must be [Q]")
        if self.query_indices.shape != (query_count,):
            raise ValueError("query_indices must be [Q]")
        sample_indices = self.query_sample_indices
        if sample_indices is None:
            if query_count != batch:
                raise ValueError(
                    "query_sample_indices are required when Q differs from B"
                )
            sample_indices = torch.arange(query_count, dtype=torch.long)
            object.__setattr__(self, "query_sample_indices", sample_indices)
        if sample_indices.shape != (query_count,):
            raise ValueError("query_sample_indices must be [Q]")
        if bool(((sample_indices < 0) | (sample_indices >= batch)).any()):
            raise ValueError("query_sample_indices contains an invalid batch row")
        if set(sample_indices.tolist()) != set(range(batch)):
            raise ValueError("every final action row must have at least one query")


def configure_trainable_parameters(
    model: torch.nn.Module,
    train_mode: str,
) -> tuple[torch.nn.Parameter, ...]:
    """Train the complete MoT and proprio encoder used by PWTR."""
    if train_mode != "full_mot":
        raise ValueError("PWTR requires full_mot training")
    if getattr(model, "action_lora_config", None) is not None:
        raise ValueError("full_mot mode requires a student without action LoRA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.mot.train()
    parameters = tuple(model.mot.parameters())
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        parameters += tuple(proprio_encoder.parameters())
    unique_parameters = tuple(dict.fromkeys(parameters))
    for parameter in unique_parameters:
        parameter.requires_grad_(True)
    return unique_parameters


class FastWAMActionAdapter:
    """Action velocity field with fixed image/text encoders and a MoT condition cache.

    Student loss predictions rebuild a differentiable video/proprio condition
    cache. Fixed-reference scoring and query generation run without gradients.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        train_video_condition: bool = False,
    ) -> None:
        self.model = model
        self.train_video_condition = bool(train_video_condition)
        for name in (
            "action_expert",
            "video_expert",
            "mot",
            "infer_action_scheduler",
        ):
            if not hasattr(model, name):
                raise TypeError(f"model is missing required FastWAM attribute {name!r}")

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.torch_dtype

    def _prepare_common_conditions(
        self,
        values: Sequence[FastWAMConditionInput],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Combine fixed image/text encodings with the proprio condition graph."""

        static = self._prepare_static_conditions(values)
        prepared = self._encode_static_conditions(static)
        if prepared.first_frame_latents is None:
            raise RuntimeError("common Fast-WAM conditions require image latents")
        return (
            prepared.first_frame_latents,
            prepared.context,
            prepared.context_mask,
            prepared.action_horizon,
        )

    def _prepare_static_conditions(
        self,
        values: Sequence[FastWAMConditionInput],
        *,
        include_first_frame: bool = True,
    ) -> _StaticCondition:
        """Encode fixed image/text inputs and detach stored proprio values."""

        values = tuple(values)
        if not values:
            raise ValueError("values must not be empty")
        action_horizon = values[0].action_horizon
        if any(value.action_horizon != action_horizon for value in values):
            raise ValueError("all conditions must use the same action horizon")
        tiled = values[0].tiled
        if any(value.tiled != tiled for value in values):
            raise ValueError("all conditions must use the same tiled setting")

        first_frame_latents = None
        if include_first_frame:
            images = []
            for value in values:
                image = value.input_image
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                images.append(image)
            with torch.no_grad():
                first_frame_latents = torch.cat(
                    [
                        self.model._encode_input_image_latents_tensor(
                            input_image=image.to(
                                device=self.device,
                                dtype=self.dtype,
                            ),
                            tiled=tiled,
                        ).detach()
                        for image in images
                    ],
                    dim=0,
                )

        contexts = []
        context_masks = []
        for value in values:
            if value.prompt is not None:
                with torch.no_grad():
                    context, context_mask = self.model.encode_prompt(value.prompt)
            else:
                assert value.context is not None and value.context_mask is not None
                context = value.context
                context_mask = value.context_mask
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "context/context_mask must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            if context.shape[0] != 1 or context_mask.shape[0] != 1:
                raise ValueError("each condition must contain exactly one context row")
            contexts.append(context)
            context_masks.append(context_mask)
        if len({tuple(context.shape[1:]) for context in contexts}) != 1:
            raise ValueError("all contexts in a batch must have the same shape")
        if len({tuple(mask.shape[1:]) for mask in context_masks}) != 1:
            raise ValueError("all context masks in a batch must have the same shape")
        context = torch.cat(contexts, dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        context_mask = torch.cat(context_masks, dim=0).to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        context = context.detach()
        context_mask = context_mask.detach()

        proprio = None
        proprio_values = [value.proprio for value in values]
        if any(proprio is not None for proprio in proprio_values):
            if any(proprio is None for proprio in proprio_values):
                raise ValueError("proprio must be present for every condition or none")
            proprio_rows = []
            for proprio in proprio_values:
                assert proprio is not None
                if proprio.ndim == 1:
                    proprio = proprio.unsqueeze(0)
                if proprio.ndim != 2 or proprio.shape[0] != 1:
                    raise ValueError(
                        f"proprio must be [D] or [1,D], got {tuple(proprio.shape)}"
                    )
                proprio_rows.append(proprio)
            if len({tuple(row.shape[1:]) for row in proprio_rows}) != 1:
                raise ValueError("all proprio rows in a batch must have the same shape")
            proprio = (
                torch.cat(proprio_rows, dim=0)
                .to(device=self.device, dtype=self.dtype)
                .detach()
            )

        return _StaticCondition(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            first_frame_latents=first_frame_latents,
            action_horizon=action_horizon,
            fuse_vae_embedding_in_latents=bool(
                getattr(
                    self.model.video_expert,
                    "fuse_vae_embedding_in_latents",
                    False,
                )
            ),
        )

    def _encode_static_conditions(
        self,
        static: _StaticCondition,
    ) -> _EncodedCondition:
        """Materialize a fresh proprio graph over cached frozen conditions."""

        context = static.context
        context_mask = static.context_mask
        if static.proprio is not None:
            grad_context = (
                contextlib.nullcontext()
                if self.train_video_condition
                else torch.no_grad()
            )
            with grad_context, autocast_for(self.model):
                context, context_mask = self.model._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=static.proprio,
                )
            if not self.train_video_condition:
                context = context.detach()
                context_mask = context_mask.detach()

        return _EncodedCondition(
            context=context,
            context_mask=context_mask,
            first_frame_latents=static.first_frame_latents,
            action_horizon=static.action_horizon,
            fuse_vae_embedding_in_latents=(static.fuse_vae_embedding_in_latents),
        )

    def prepare_condition(
        self, value: FastWAMConditionInput
    ) -> PreparedFastWAMCondition:
        return self.prepare_conditions((value,))

    def prepare_conditions(
        self,
        values: Sequence[FastWAMConditionInput],
    ) -> PreparedFastWAMCondition:
        """Prepare a same-shape condition batch in one video/MoT pass."""

        (
            first_frame_latents,
            context,
            context_mask,
            action_horizon,
        ) = self._prepare_common_conditions(values)

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        fuse_flag = bool(
            getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        grad_video = (
            contextlib.nullcontext() if self.train_video_condition else torch.no_grad()
        )
        with grad_video, autocast_for(self.model):
            video_pre = self.model.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self.model._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        if not self.train_video_condition:
            context = context.detach()
            context_mask = context_mask.detach()
            video_kv_cache = [
                {key: tensor.detach() for key, tensor in layer.items()}
                for layer in video_kv_cache
            ]
        return PreparedFastWAMCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask.detach(),
            video_seq_len=video_seq_len,
            action_horizon=action_horizon,
        )

    @torch.no_grad()
    def collect_online_queries(
        self,
        condition_inputs: Sequence[FastWAMConditionInput],
        *,
        num_inference_steps: int,
        sigma_shift: float | None,
        seeds: Sequence[int],
        query_indices: Sequence[int | Sequence[int]],
        rand_device: str = "cpu",
        storage_device: str = "cpu",
    ) -> OnlineFlowQuery:
        """Roll out the current student without retaining a replay trajectory."""

        condition_inputs = tuple(condition_inputs)
        seeds = tuple(int(seed) for seed in seeds)
        raw_query_indices = tuple(query_indices)
        batch = len(condition_inputs)
        if batch == 0:
            raise ValueError("condition_inputs must not be empty")
        if len(seeds) != batch or len(raw_query_indices) != batch:
            raise ValueError("conditions, seeds, and query indices must align")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        query_indices_by_sample = []
        for raw_indices in raw_query_indices:
            if isinstance(raw_indices, int):
                indices = (int(raw_indices),)
            else:
                indices = tuple(int(index) for index in raw_indices)
            if not indices:
                raise ValueError("every sample needs at least one query index")
            if len(indices) != len(set(indices)):
                raise ValueError("query indices for one sample must be unique")
            if any(index < 0 or index >= num_inference_steps for index in indices):
                raise ValueError("query index is outside the inference schedule")
            query_indices_by_sample.append(indices)

        flat_sample_indices = tuple(
            row for row, indices in enumerate(query_indices_by_sample) for _ in indices
        )
        flat_query_indices = tuple(
            index for indices in query_indices_by_sample for index in indices
        )

        condition = self.prepare_conditions(condition_inputs)
        latent_rows = []
        for seed in seeds:
            generator = torch.Generator(device=rand_device).manual_seed(seed)
            latent_rows.append(
                torch.randn(
                    (
                        1,
                        condition.action_horizon,
                        int(self.model.action_expert.action_dim),
                    ),
                    generator=generator,
                    device=rand_device,
                    dtype=torch.float32,
                )
            )
        latents = torch.cat(latent_rows, dim=0).to(device=self.device, dtype=self.dtype)
        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        query_count = len(flat_query_indices)
        query_latents = torch.empty(
            (query_count, *latents.shape[1:]),
            device=self.device,
            dtype=latents.dtype,
        )
        captured = torch.zeros(query_count, device=self.device, dtype=torch.bool)
        for step_index, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            query_rows = [
                query_row
                for query_row, query_index in enumerate(flat_query_indices)
                if query_index == step_index
            ]
            if query_rows:
                source_rows = [flat_sample_indices[row] for row in query_rows]
                query_latents[query_rows] = latents[source_rows]
                captured[query_rows] = True
            timestep = step_t.expand(batch).to(device=self.device, dtype=latents.dtype)
            velocity = self.predict_velocity(
                latents_action=latents,
                timestep_action=timestep,
                condition=condition,
            )
            latents = transition_mean(latents, velocity, step_delta)
        if not bool(captured.all()):
            raise RuntimeError("not every online query state was captured")
        query_index_tensor = torch.tensor(
            flat_query_indices, device=timesteps.device, dtype=torch.long
        )
        query_sample_index_tensor = torch.tensor(
            flat_sample_indices, device=timesteps.device, dtype=torch.long
        )
        return OnlineFlowQuery(
            query_latents=query_latents.detach().to(
                storage_device, dtype=torch.float32
            ),
            query_timesteps=timesteps.index_select(0, query_index_tensor)
            .detach()
            .to(storage_device, dtype=torch.float32),
            query_indices=query_index_tensor.detach().to(storage_device),
            final_latents=latents.detach().to(storage_device, dtype=torch.float32),
            query_sample_indices=query_sample_index_tensor.detach().to(
                storage_device
            ),
        )

    def predict_velocity(
        self,
        *,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        condition: PreparedFastWAMCondition,
    ) -> torch.Tensor:
        action_pre = self.model.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=condition.context,
            context_mask=condition.context_mask,
        )
        action_tokens = self.model.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=condition.video_kv_cache,
            attention_mask=condition.attention_mask,
            video_seq_len=condition.video_seq_len,
        )
        return self.model.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def collect_student_trajectory(
        self,
        condition_input: FastWAMConditionInput,
        *,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int,
        rand_device: str = "cpu",
        storage_device: str = "cpu",
    ) -> DenoisingTrajectory:
        """Generate a deterministic flow path from fresh seeded Gaussian noise."""
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        condition = self.prepare_condition(condition_input)
        generator = torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (
                1,
                condition.action_horizon,
                int(self.model.action_expert.action_dim),
            ),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        states = [latents.detach().to(storage_device, dtype=torch.float32)]
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.reshape(1).to(device=self.device, dtype=latents.dtype)
            velocity = self.predict_velocity(
                latents_action=latents,
                timestep_action=timestep,
                condition=condition,
            )
            mean = transition_mean(latents, velocity, step_delta)
            latents = mean
            states.append(latents.detach().to(storage_device, dtype=torch.float32))
        return DenoisingTrajectory(
            latents=torch.stack(states, dim=1),
            timesteps=timesteps.detach().to(storage_device, dtype=torch.float32),
            deltas=deltas.detach().to(storage_device, dtype=torch.float32),
        )


def autocast_for(model: torch.nn.Module):
    device = torch.device(model.device)
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = model.torch_dtype
    if dtype not in {torch.float16, torch.bfloat16}:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)
