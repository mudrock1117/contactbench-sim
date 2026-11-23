"""Timestamp-aware learning windows over ContactBench demonstration episodes.

The dataset is intentionally model-agnostic.  NumPy arrays are returned by
default (PyTorch's default collator converts them to tensors); ``as_torch``
performs an explicit lazy conversion without making torch a core dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .episode import discover_episodes, load_episode


@dataclass(frozen=True)
class WindowSpec:
    observation_horizon: int = 2
    action_horizon: int = 16
    camera_name: str | None = "fixed"
    rgb_alignment: str = "previous"
    include_tactile: bool = False
    include_privileged: bool = False
    rgb_channels_first: bool = True

    def __post_init__(self):
        if self.observation_horizon < 1 or self.action_horizon < 1:
            raise ValueError("window horizons must be positive")
        if self.rgb_alignment != "previous":
            raise ValueError("only causal 'previous' RGB alignment is supported")


def policy_observation_indices(metadata, *, include_privileged=False):
    """Indices visible to a policy, derived from field metadata.

    Privileged object poses are excluded unless the caller explicitly opts in.
    Old episodes without field metadata fail closed rather than guessing that
    every recorded scalar is deployable.
    """
    fields = metadata.get("observation_fields")
    if not isinstance(fields, list):
        raise ValueError("episode has no observation_fields; policy inputs cannot be inferred safely")
    indices, selected = [], []
    for field in fields:
        if field.get("privileged", False) and not include_privileged:
            continue
        start, stop = int(field["start"]), int(field["stop"])
        if start < 0 or stop <= start:
            raise ValueError("invalid observation field range")
        indices.extend(range(start, stop))
        selected.append(dict(field))
    return np.asarray(indices, dtype=np.int64), selected


def _episode_key(path, metadata, level):
    if level == "task":
        return str(metadata.get("task", "unknown"))
    if level == "seed":
        return str(metadata.get("seed", "unknown"))
    if level == "episode":
        return str(metadata.get("episode_id", Path(path).name))
    raise ValueError("split level must be task, episode, or seed")


def split_episode_paths(paths, *, level="episode", fractions=(0.8, 0.1, 0.1), seed=0):
    """Deterministically split whole groups; no task/episode/seed crosses splits."""
    names = ("train", "validation", "test")
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("fractions must be three nonnegative values summing to one")
    boundaries = np.cumsum(fractions)
    result = {name: [] for name in names}
    cache = {}
    for path in map(Path, paths):
        metadata = load_episode(path).metadata
        key = _episode_key(path, metadata, level)
        if key not in cache:
            digest = hashlib.sha256(f"{int(seed)}:{level}:{key}".encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            cache[key] = names[min(int(np.searchsorted(boundaries, value, side="right")), 2)]
        result[cache[key]].append(path)
    return result


class EpisodeWindowDataset:
    """Map-style dataset anchored at a 60 Hz action/state index.

    For anchor ``k`` the history contains state indices
    ``k-H+1 .. k`` and future actions contain ``k .. k+A-1``.  Action ``i``
    is effective on ``[timestamps[i], timestamps[i+1]]``.  Each state time is
    paired with the newest RGB time not later than it; repeated RGB indices
    are expected when images are 12 Hz and control is 60 Hz.
    """

    def __init__(self, root_or_paths, spec=WindowSpec(), *, success_only=False, as_torch=False):
        self.spec, self.as_torch = spec, bool(as_torch)
        if isinstance(root_or_paths, (str, Path)):
            paths = discover_episodes(root_or_paths, success_only=success_only)
        else:
            paths = [Path(path) for path in root_or_paths]
        self.episodes, self.samples = [], []
        for path in paths:
            episode = load_episode(path)  # validates episode and RGB checksums/timestamps
            if episode["actions"].shape[1] != 9:
                raise ValueError(f"{path}: expected recorded 9D joint action")
            obs_indices, fields = policy_observation_indices(
                episode.metadata, include_privileged=spec.include_privileged)
            if len(obs_indices) == 0 or obs_indices.max() >= episode["observations"].shape[1]:
                raise ValueError(f"{path}: observation field metadata is inconsistent")
            item = {"episode": episode, "observation_indices": obs_indices,
                    "observation_fields": fields}
            if spec.camera_name is not None:
                item.update(self._camera_index(path, episode.metadata, spec.camera_name))
            episode_index = len(self.episodes)
            self.episodes.append(item)
            first = spec.observation_horizon - 1
            last = len(episode["actions"]) - spec.action_horizon
            self.samples.extend((episode_index, anchor) for anchor in range(first, last + 1))

    @staticmethod
    def _camera_index(path, metadata, camera_name):
        rgb = metadata.get("rgb")
        if rgb is None or camera_name not in rgb.get("camera_names", []):
            return {"camera_dir": None, "rgb_timestamps": None}
        directory = Path(path) / "rgb" / camera_name
        return {"camera_dir": directory,
                "rgb_timestamps": np.load(directory / "timestamps.npy", allow_pickle=False)}

    def __len__(self):
        return len(self.samples)

    def _convert(self, value):
        if not self.as_torch:
            return value
        try:
            import torch
        except ImportError as error:
            raise ImportError("as_torch=True requires PyTorch") from error
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, dict):
            return {key: self._convert(item) for key, item in value.items()}
        return value

    def __getitem__(self, index):
        episode_index, anchor = self.samples[index]
        item = self.episodes[episode_index]
        episode = item["episode"]
        h, a = self.spec.observation_horizon, self.spec.action_horizon
        obs_slice = slice(anchor - h + 1, anchor + 1)
        act_slice = slice(anchor, anchor + a)
        state_t = episode["timestamps"][obs_slice]
        action_start = episode["timestamps"][anchor:anchor + a]
        action_end = episode["timestamps"][anchor + 1:anchor + a + 1]
        sample = {
            "proprio": episode["observations"][obs_slice][:, item["observation_indices"]].copy(),
            "proprio_timestamps": state_t.copy(),
            "actions": episode["actions"][act_slice].copy(),
            "action_start_timestamps": action_start.copy(),
            "action_end_timestamps": action_end.copy(),
            "episode_id": episode.metadata["episode_id"],
            "task": episode.metadata.get("task"),
            "seed": episode.metadata.get("seed"),
            "anchor_index": np.int64(anchor),
        }
        camera_dir, rgb_t = item["camera_dir"], item["rgb_timestamps"]
        if camera_dir is not None:
            rgb_indices = np.searchsorted(rgb_t, state_t, side="right") - 1
            valid = rgb_indices >= 0
            safe = np.maximum(rgb_indices, 0)
            frames = [iio.imread(camera_dir / "images" / f"{int(i):06d}.png") for i in safe]
            frames = np.stack(frames)
            if self.spec.rgb_channels_first:
                frames = frames.transpose(0, 3, 1, 2)
            sample.update(rgb=frames, rgb_valid=valid,
                          rgb_indices=rgb_indices.astype(np.int64),
                          rgb_timestamps=np.where(valid, rgb_t[safe], np.nan),
                          rgb_age_s=np.where(valid, state_t - rgb_t[safe], np.nan))
        if self.spec.include_tactile and "tactile" in episode.arrays:
            sample["tactile"] = episode["tactile"][obs_slice].copy()
            sample["tactile_valid"] = episode["tactile_valid"][obs_slice].copy()
        language = episode.metadata.get("language_instruction")
        if language is not None:
            sample["language_instruction"] = str(language)
        return self._convert(sample)

    @property
    def schema(self):
        h, a = self.spec.observation_horizon, self.spec.action_horizon
        return {
            "proprio": {"shape": [h, "D_policy"], "dtype": "float32",
                         "units": "per observation_fields; normally q[rad,m] + qd[rad/s,m/s]"},
            "rgb": {"shape": ([h, 3, "H", "W"] if self.spec.rgb_channels_first
                               else [h, "H", "W", 3]), "dtype": "uint8", "optional": True,
                    "alignment": "latest image timestamp <= proprio timestamp; may repeat at 12 vs 60 Hz"},
            "actions": {"shape": [a, 9], "dtype": "float32",
                        "units": "joint position targets: 7 rad + 2 m",
                        "time": "action[i] applies from action_start_timestamps[i] to action_end_timestamps[i]"},
            "tactile": {"shape": [h, 2, "Ht", "Wt"], "dtype": "float32", "optional": True},
            "language_instruction": {"shape": [], "dtype": "string", "optional": True},
        }

