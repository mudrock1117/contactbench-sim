"""Pure timing contract for 12 Hz latent planning and 24 Hz AT decoding."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RDPTimeConfig:
    physics_hz: float = 60.0
    slow_policy_hz: float = 12.0
    fast_action_hz: float = 24.0
    observation_horizon: int = 2
    action_horizon: int = 32
    execution_horizon: int = 2

    def __post_init__(self):
        if self.physics_hz <= 0 or self.slow_policy_hz <= 0 or self.fast_action_hz <= 0:
            raise ValueError("frequencies must be positive")
        if not np.isclose(self.fast_action_hz / self.slow_policy_hz, 2.0):
            raise ValueError("official RDP contract requires two 24 Hz ticks per 12 Hz plan")
        if self.observation_horizon < 1 or self.action_horizon < 1:
            raise ValueError("horizons must be positive")
        if self.execution_horizon != 2:
            raise ValueError("12/24 Hz RDP executes exactly two fast actions before replanning")

    @property
    def fast_dt(self) -> float:
        return 1.0 / self.fast_action_hz

    @property
    def slow_dt(self) -> float:
        return 1.0 / self.slow_policy_hz

    @property
    def dataset_obs_steps(self) -> int:
        return self.observation_horizon * 2

    @property
    def action_start_index(self) -> int:
        # Exact upstream LatentDiffusionUnetImagePolicy slicing rule.
        return self.dataset_obs_steps - 1

    @property
    def n_action_steps(self) -> int:
        return self.action_horizon - self.dataset_obs_steps + 1

    def fast_query_times(self, anchor_s: float) -> np.ndarray:
        start = anchor_s - self.action_start_index * self.fast_dt
        return start + np.arange(self.action_horizon, dtype=np.float64) * self.fast_dt

    def slow_observation_times(self, anchor_s: float) -> np.ndarray:
        # Causal 24 Hz history [..] downsampled with upstream [::-2][::-1],
        # yielding the two 12 Hz observations and ending exactly at anchor.
        fast = self.fast_query_times(anchor_s)[:self.dataset_obs_steps]
        return fast[::-2][::-1]

    def physics_steps_for_fast_tick(self, tick: int) -> int:
        # Bresenham schedule for non-integral 60/24 = 2.5: 3,2,3,2...
        a = round((tick + 1) * self.physics_hz / self.fast_action_hz)
        b = round(tick * self.physics_hz / self.fast_action_hz)
        return int(a - b)
