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

For Allen key, `T_link7_camera` is now calibrated through the fixed TCP frame:
`T_link7_camera = T_link7_tcp @ T_tcp_camera`. Newton's tiled camera uses +X
right, +Y up, and -Z optical forward. The default side mount is at
`[55, 0, -35] mm` in TCP coordinates and looks at `[0, 0, 110] mm`, along the
tool approach region, with a 72 degree vertical FOV. The 55 mm lateral offset
puts the origin outside the central wrist/finger silhouette. This transform is
computed once; no object pose is used at runtime. Override the six coordinates
with `AK_WRIST_CAM_P*` and `AK_WRIST_CAM_T*`, and FOV with
`AK_WRIST_CAM_FOV`. `python -m force_bench.data.wrist_mount_preview OUT.png`
plots the built-in `tool`, `wide`, and `close` rigid candidates without loading
the simulator.

`SensorTiledCamera` is ray based and this wrapper does not expose a separate
near-clip setting. Speckles concentrated on the former wrist view are therefore
more likely to be rays starting inside/very near robot geometry, hard-shadow
self-intersection, or undersampling than PNG corruption. The side offset
addresses the first cause; actual scene rendering is still required to
distinguish the remaining renderer artifacts.

## A3.3 offline wrist-mount comparison

`python -m force_bench.data.compare_wrist_mounts EPISODE OUTPUT.png` rebuilds
the recorded task scene and restores exact rows from
`artifacts/**/rollout_states.npz`. It refuses episodes without the complete
`body_q/body_qd` artifact; the 18D policy proprioception and privileged
nonrobot poses in `episode.npz` are not sufficient to reconstruct robot-link
render state and are never interpolated. Scene task arguments and recorded
physics environment variables are reapplied for topology compatibility.

The default columns are the requested rigid TCP mounts A-D. Every column in a
row is rendered from the same restored physical-state row. A valid Allen key
`expert_log.npz` selects approach, contact, rotation, and exit phases; otherwise
four uniformly spaced columns are explicitly labelled `time sample`. The PNG
is accompanied by a JSON report containing timestamps, parameters, privileged
offline projections of the recorded key tip and screw recess, image-bound
checks, and center-ray shape/body IDs. These diagnostics are not added to an
episode, dataset sample, or online policy observation, and they do not replace
manual inspection of the rendered pixels.

Custom candidates are supplied as a JSON object mapping names to
`{"position": [x,y,z], "target": [x,y,z], "fov_deg": value}` in TCP units of
metres/degrees. All transforms are computed once as
`T_link7_tcp @ T_tcp_camera`; no per-frame object look-at is performed.

Ubuntu example (rendering is intentionally an explicit user action):

```bash
python -m force_bench.data.compare_wrist_mounts \
  outputs/datasets/allen_dual/EPISODE_ID \
  outputs/wrist_mount_comparison.png
```

This writes `wrist_mount_comparison.png` and
`wrist_mount_comparison.json`. If an older/moved episode lacks the state
artifact, pass the original artifact explicitly with
`--states /path/to/rollout_states.npz`. If no such artifact exists, exact
offline rerendering is impossible: retain the episode unchanged and perform a
new opt-in collection that publishes full states; do not synthesize link poses
from 9D actions or 18D proprioception.

`ClosedLoopPolicyEnv` accepts `visual_mode=none|fixed|wrist|both`, optional
GelSight pressure rasters, an observation history, and a shared
`PolicyTimeConfig`. `replan(chunk)` replaces the unused action tail and
`step_next()` consumes one command exactly once. Reset clears the chunk and
sensor history.

GelSight output is a two-pad `float32` pressure raster in Pa with an explicit
validity vector. It is not a tactile RGB image. The Allen-key state evaluator
only reports the subset supported without expert phases. Full task success is
left unknown until the original torque/probe/contact evidence is available.
