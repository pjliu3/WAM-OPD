"""FastWAM's three-view RoboTwin observation and action transforms."""
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from fastwam_opd.fastwam_adapter import FastWAMConditionInput


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize(size_wh, resample=Image.Resampling.BILINEAR), dtype=np.uint8)


@dataclass
class RoboTwinConditionBuilder:
    processor: Any
    action_horizon: int = 32
    prompt_template: str = DEFAULT_PROMPT

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        metadata = self.processor.shape_meta["state"]
        if len(metadata) != 1:
            raise ValueError("expected one merged state key")
        key = metadata[0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key].cpu()

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError("action must have shape [B,T,D]")
        metadata = self.processor.shape_meta["action"]
        if len(metadata) != 1:
            raise ValueError("expected one merged action key")
        normalizer = self.processor.normalizer.normalizers["action"][metadata[0]["key"]]
        return normalizer.backward(action.float().cpu()).numpy()

    def build(self, observation: Mapping[str, Any], instruction: str) -> FastWAMConditionInput:
        cameras = observation["observation"]
        head = _resize_rgb(cameras["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(cameras["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(cameras["right_camera"]["rgb"], (160, 128))
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        image = image * (2.0 / 255.0) - 1.0
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return FastWAMConditionInput(
            input_image=image,
            action_horizon=self.action_horizon,
            prompt=self.prompt_template.format(task=instruction),
            proprio=self._normalize_state(state),
        )
