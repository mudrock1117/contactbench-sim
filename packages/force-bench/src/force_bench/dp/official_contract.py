"""Pure validation for official DP timing and recorded command fields."""
from __future__ import annotations
import numpy as np

EEF_FIELDS={
    "action_anchor_tcp_pose_world_xyzw": (7,),
    "action_tcp_target_world_xyzw": (7,),
    "action_anchor_gripper_width_m": (),
    "action_gripper_width_m": (),
}
AUX_FIELDS=("action_target_velocity","action_feedforward")

def official_action_query_times(current_time, *, policy_hz, horizon, n_obs_steps):
    """Upstream trajectory: current command is exactly index n_obs_steps-1."""
    if policy_hz<=0 or horizon<1 or not 1<=n_obs_steps<=horizon:
        raise ValueError("invalid official DP time configuration")
    return float(current_time)+(np.arange(horizon)-(n_obs_steps-1))/float(policy_hz)

def validate_eef_episode_arrays(arrays, transition_count, *, auxiliary_tolerance=1e-8):
    missing=[key for key in (*EEF_FIELDS,*AUX_FIELDS) if key not in arrays]
    if missing:
        raise ValueError(f"missing exact command fields {missing}")
    for key,suffix in EEF_FIELDS.items():
        value=np.asarray(arrays[key])
        expected=(int(transition_count),)+suffix
        if value.shape!=expected or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite shape {expected}, got {value.shape}")
    for key in AUX_FIELDS:
        value=np.asarray(arrays[key])
        if value.shape!=(int(transition_count),9) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite shape {(int(transition_count),9)}")
        peak=float(np.max(np.abs(value),initial=0.0))
        if peak>auxiliary_tolerance:
            raise ValueError(
                f"{key} peak {peak:g} is nonzero; 10D pose targets do not encode this "
                "auxiliary expert control and no conversion will be fabricated")
    return True
