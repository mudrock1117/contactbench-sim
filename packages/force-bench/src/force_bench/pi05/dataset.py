"""ContactBench windows and OpenPI transforms for pi0.5."""
from __future__ import annotations

import dataclasses
import numpy as np

from force_bench.data.dataset import EpisodeWindowDataset, WindowSpec
from force_bench.data.episode import discover_episodes, load_episode
from force_bench.data.timebase import PolicyTimeConfig, causal_indices
from force_bench.dp.official_contract import validate_eef_episode_arrays
from force_bench.dp.official_dataset import _pose_matrix
from force_bench.robot.eef_action import encode_relative_eef, relative_eef_history
from .official import activate_openpi

PROMPT = "pick up the thin object on the table"
ACTION_DIM = 10
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 32
POLICY_HZ = 24
PHYSICS_HZ = 60


class ContactBenchPi05Dataset:
    """One causal wrist observation and 32 fixed-anchor 10D EEF targets."""

    def __init__(self, data_path, *, success_only=True, prompt=PROMPT):
        timing = PolicyTimeConfig(
            physics_hz=PHYSICS_HZ, policy_hz=POLICY_HZ,
            observation_horizon=1, action_horizon=ACTION_HORIZON)
        self.base = EpisodeWindowDataset(
            discover_episodes(data_path, success_only=success_only),
            WindowSpec(
                observation_horizon=1, action_horizon=ACTION_HORIZON,
                camera_name="wrist", include_privileged=False,
                policy_time_config=timing))
        if not len(self.base):
            raise ValueError(f"no pi0.5 windows under {data_path}")
        self.prompt = str(prompt)
        self.episodes = {}
        for item in self.base.episodes:
            ep = item["episode"]
            validate_eef_episode_arrays(ep.arrays, len(ep["actions"]))
            self.episodes[str(ep.metadata["episode_id"])] = ep

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        ep = self.episodes[str(sample["episode_id"])]
        anchor_s = float(np.asarray(sample["timestamp"])[-1])
        query = anchor_s + np.arange(ACTION_HORIZON, dtype=np.float64) / POLICY_HZ
        action_times = np.asarray(ep["timestamps"][:-1], dtype=np.float64)
        indices = causal_indices(action_times, query)
        if np.any(indices < 0) or np.any(indices >= len(ep["actions"])):
            raise IndexError("pi0.5 action chunk crosses episode boundary")
        anchor_index = int(indices[0])
        anchor = _pose_matrix(
            ep["action_anchor_tcp_pose_world_xyzw"][anchor_index])
        actions = np.stack([
            encode_relative_eef(
                anchor,
                _pose_matrix(ep["action_tcp_target_world_xyzw"][i]),
                ep["action_gripper_width_m"][i])
            for i in indices]).astype(np.float32)
        anchor_width = float(ep["action_anchor_gripper_width_m"][anchor_index])
        state = relative_eef_history(
            anchor[None], np.asarray([anchor_width]), anchor)[0].astype(np.float32)
        rgb = np.asarray(sample["rgb"], dtype=np.uint8)
        if rgb.shape != (1, 3, 240, 320):
            raise ValueError(f"expected causal wrist RGB [1,3,240,320], got {rgb.shape}")
        image = rgb[0].transpose(1, 2, 0).copy()
        rgb_timestamp = float(np.asarray(sample["rgb_timestamp"])[-1])
        if rgb_timestamp > anchor_s + 1e-12:
            raise ValueError("future wrist image selected for pi0.5")
        return {
            "wrist_rgb": image,
            "state": state,
            "actions": actions,
            "prompt": self.prompt,
            "timestamp": np.float64(anchor_s),
            "rgb_timestamp": np.float64(rgb_timestamp),
            "rgb_age_s": np.float64(anchor_s - rgb_timestamp),
            "policy_hz": np.float32(POLICY_HZ),
            "rgb_hz": np.float32(12.0),
        }


activate_openpi()
from openpi import transforms
from openpi.models import model as openpi_model


@dataclasses.dataclass(frozen=True)
class ContactBenchInputs(transforms.DataTransformFn):
    """Map wrist-only ContactBench samples to the official three-camera schema."""

    model_type: openpi_model.ModelType

    def __call__(self, data):
        image = np.asarray(data["wrist_rgb"], dtype=np.uint8)
        if image.shape != (240, 320, 3):
            raise ValueError(f"wrist_rgb must be HWC uint8 [240,320,3], got {image.shape}")
        blank = np.zeros_like(image)
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": blank,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": blank.copy(),
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": data.get("prompt", PROMPT),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-2:] != (ACTION_HORIZON, ACTION_DIM):
                raise ValueError(f"actions must end in [32,10], got {actions.shape}")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class ContactBenchOutputs(transforms.DataTransformFn):
    """Remove official 32D padding without altering the 10D action semantics."""

    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.shape[-1] != MODEL_ACTION_DIM:
            raise ValueError(f"OpenPI output must have {MODEL_ACTION_DIM} dimensions")
        result = dict(data)
        result["actions"] = actions[..., :ACTION_DIM].copy()
        return result
