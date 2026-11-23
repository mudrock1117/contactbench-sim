# A3 multimodal policy interfaces

The original 60 Hz episode arrays remain unchanged. `PolicyTimeConfig(60, 12,
2, 16)` enables an additional causal 12 Hz view: observations use the newest
state and sensor frame at or before the policy tick, while each selected 9D
joint target is held on `[action_start, action_end)`. Incomplete tail windows
are dropped; no window is padded across an episode boundary.

Expert RGB collection is opt-in. `TB_COLLECT_RGB=1` keeps the legacy fixed
camera behavior. Set `TB_COLLECT_RGB_CAMERAS` to `none`, `fixed`, `wrist`, or
`both`. Each selected camera writes its own `rgb/<name>` timestamps, PNGs,
calibration, per-frame world pose, and SHA-256 manifest. Both cameras are
sampled from the same post-physics state in one rollout.

The wrist mount is defined by
`T_world_camera = T_world_link7 @ T_link7_camera`; `T_world_link7` is read from
the simulated body's actual pose on every sampled frame. The fixed camera's
world transform is constant. Images are `uint8` RGB and policy-facing arrays
are CHW.

`ClosedLoopPolicyEnv` accepts `visual_mode=none|fixed|wrist|both`, optional
GelSight pressure rasters, an observation history, and a shared
`PolicyTimeConfig`. `replan(chunk)` replaces the unused action tail and
`step_next()` consumes one command exactly once. Reset clears the chunk and
sensor history.

GelSight output is a two-pad `float32` pressure raster in Pa with an explicit
validity vector. It is not a tactile RGB image. The Allen-key state evaluator
only reports the subset supported without expert phases. Full task success is
left unknown until the original torque/probe/contact evidence is available.
