"""Shared causal timebase rules for offline windows and online policies."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyTimeConfig:
    physics_hz: float = 60.0
    policy_hz: float = 12.0
    observation_horizon: int = 2
    action_horizon: int = 16

    def __post_init__(self):
        ratio = self.physics_hz / self.policy_hz
        if self.physics_hz <= 0 or self.policy_hz <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("physics_hz/policy_hz must be a positive integer")
        if self.observation_horizon < 1 or self.action_horizon < 1:
            raise ValueError("horizons must be positive")

    @property
    def policy_dt(self):
        return 1.0 / self.policy_hz

    @property
    def physics_frames_per_action(self):
        return int(round(self.physics_hz / self.policy_hz))


def causal_indices(source_timestamps, query_timestamps):
    """Newest source at or before each query; -1 means no causal sample."""
    source = np.asarray(source_timestamps, dtype=np.float64)
    query = np.asarray(query_timestamps, dtype=np.float64)
    if source.ndim != 1 or query.ndim != 1 or len(source) == 0:
        raise ValueError("timestamps must be vectors and source must be nonempty")
    return np.searchsorted(source, query, side="right") - 1


def policy_action_ticks(state_timestamps, config):
    """Policy starts whose full hold interval remains inside the episode."""
    timestamps = np.asarray(state_timestamps, dtype=np.float64)
    duration = timestamps[-1] - timestamps[0]
    count = int(np.floor(duration / config.policy_dt + 1e-10))
    ticks = timestamps[0] + np.arange(count, dtype=np.float64) * config.policy_dt
    return ticks[ticks + config.policy_dt <= timestamps[-1] + 1e-12]


def resample_joint_actions(actions, state_timestamps, query_timestamps):
    """Sample the 60 Hz action active at each policy start timestamp."""
    actions = np.asarray(actions)
    state_timestamps = np.asarray(state_timestamps, dtype=np.float64)
    query = np.asarray(query_timestamps, dtype=np.float64)
    indices = causal_indices(state_timestamps[:-1], query)
    if np.any(indices < 0) or np.any(indices >= len(actions)):
        raise ValueError("policy action query crosses the episode boundary")
    return actions[indices].copy(), indices.astype(np.int64)


class ActionChunkExecutor:
    """Consume a predicted chunk once; replacing it implements replanning."""
    def __init__(self):
        self._chunk = None
        self._next = 0

    def replan(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError("action chunk must be nonempty [N,A]")
        self._chunk, self._next = actions.copy(), 0

    def pop(self):
        if self._chunk is None or self._next >= len(self._chunk):
            raise RuntimeError("action chunk exhausted; replan is required")
        action = self._chunk[self._next].copy()
        self._next += 1
        return action

    @property
    def remaining(self):
        return 0 if self._chunk is None else len(self._chunk) - self._next
