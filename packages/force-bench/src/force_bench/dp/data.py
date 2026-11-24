from __future__ import annotations

from pathlib import Path

import numpy as np

from force_bench.data.dataset import (EpisodeWindowDataset, WindowSpec,
                                      split_episode_paths)
from force_bench.data.episode import discover_episodes, load_episode
from force_bench.data.timebase import PolicyTimeConfig


def window_spec(config):
    timing = PolicyTimeConfig(physics_hz=config.physics_hz, policy_hz=config.policy_hz,
                              observation_horizon=config.observation_horizon,
                              action_horizon=config.action_horizon)
    camera = tuple(config.camera_names) if config.camera == "both" else config.camera
    return WindowSpec(observation_horizon=config.observation_horizon,
                      action_horizon=config.action_horizon, camera_name=camera,
                      include_tactile=False, include_privileged=False,
                      policy_time_config=timing)


def episode_splits(config):
    paths = discover_episodes(config.data_path, success_only=True)
    if not paths:
        raise ValueError(f"no successful episodes found under {config.data_path}")
    if config.overfit_single_episode:
        if len(paths) != 1:
            raise ValueError("overfit_single_episode requires exactly one successful episode")
        return {"train": paths, "validation": [], "test": []}
    return split_episode_paths(
        paths, level=config.split_level,
        fractions=(config.train_fraction, config.validation_fraction,
                   config.test_fraction), seed=config.seed)


def build_dataset(paths, config, *, as_torch=False):
    return EpisodeWindowDataset(paths, window_spec(config), as_torch=as_torch)


def compute_statistics(dataset):
    """Compute normalization exclusively over the selected training windows."""
    if not len(dataset):
        raise ValueError("cannot compute statistics from an empty training set")
    proprio_sum = np.zeros(18, dtype=np.float64)
    proprio_sq = np.zeros(18, dtype=np.float64)
    action_min = np.full(9, np.inf, dtype=np.float64)
    action_max = np.full(9, -np.inf, dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        proprio = np.asarray(sample["proprio"], dtype=np.float64)
        actions = np.asarray(sample["actions"], dtype=np.float64)
        proprio_sum += proprio.sum(axis=0)
        proprio_sq += np.square(proprio).sum(axis=0)
        count += len(proprio)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
    mean = proprio_sum / count
    variance = np.maximum(proprio_sq / count - mean * mean, 1.0e-12)
    return {"proprio_mean": mean.astype(np.float32),
            "proprio_std": np.sqrt(variance).astype(np.float32),
            "action_min": action_min.astype(np.float32),
            "action_max": action_max.astype(np.float32)}


def auxiliary_action_audit(paths):
    """Report recorded controls that the 9D deployment interface omits."""
    maxima = {"action_target_velocity": 0.0, "action_feedforward": 0.0}
    missing = []
    for path in map(Path, paths):
        episode = load_episode(path)
        for key in maxima:
            if key not in episode.arrays:
                missing.append(f"{path.name}:{key}")
            else:
                maxima[key] = max(maxima[key], float(np.abs(episode[key]).max(initial=0.0)))
    return {"max_abs": maxima, "missing": missing,
            "joint_position_only_equivalent": all(value <= 1.0e-8 for value in maxima.values())
                                              and not missing}


class TorchWindowDataset:
    """Drop metadata/privileged fields before PyTorch collation."""
    def __init__(self, dataset, camera_names):
        self.dataset, self.camera_names = dataset, tuple(camera_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if len(self.camera_names) == 1:
            images = {self.camera_names[0]: sample["rgb"]}
        else:
            images = {name: sample["cameras"][name]["rgb"] for name in self.camera_names}
        # Only deployable causal fields are exposed to the model.
        return {"proprio": sample["proprio"], "images": images,
                "actions": sample["actions"]}
