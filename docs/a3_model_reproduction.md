# A3 closure and B1 interface contract

## Wrist camera decision

Allen key uses candidate A by default: `position_tcp=(0.055,0,-0.035) m`,
`target_tcp=(0,0,0.110) m`, vertical FOV 72 degrees. The camera looks along
camera `-Z`; `+X` is image right and `+Y` is image up. The rigid transform is
computed once as `T_link7_camera = T_link7_tcp @ T_tcp_camera`, and every frame
uses `T_world_camera = T_world_link7 @ T_link7_camera` from measured link state.

The offline comparison tool renders and projects through the same
`TeleopCamera`, after the same `set_transform`, so projection and shape-index
ray diagnostics share pose, FOV and image size. Its `possible_robot_occlusion`
is only a centre-pixel diagnostic: geometry can naturally hide a target, and a
projection inside the image does not prove visibility. B/C/D remain historical
comparison presets, not runtime defaults.

## Policy observation and time contract

Default policy time is 60 Hz physics, 12 Hz policy/RGB, observation horizon 2
and action horizon 16. At policy time `t`, RGB is the newest image whose
timestamp is `<= t`; no future image is used. A predicted action `i` is held on
`[action_start_timestamps[i], action_end_timestamps[i])`. Replanning discards
only the unexecuted tail, and `step_next()` consumes one action exactly once.

The policy observation is `rgb` (`uint8`, CHW), 18-D `proprio` (nine measured
robot positions followed by nine velocities), timestamps/ages, optional
GelSight pressure `[2,H,W]` in Pa, and a task-only language instruction. Object
poses beyond index 18 are privileged and excluded by default. Online and
offline camera calibration comes from the same task definitions.

## Tactile contract and limitation

`TB_GELSIGHT=1` mounts simulated GelSight Mini pads; Allen key is now an opt-in
GelSight-capable task. The two saved arrays are pressure grids, never tactile
RGB. `tactile_valid` distinguishes a real sensor sample from zero fill. Tactile
is stored on the state timebase; the policy loader selects the same causal
observation indices used for proprio. The live environment independently
controls tactile sampling frequency.

RDP and FTP-1 may require optical tactile RGB, markers, shear, or a learned
sensor representation. This repository currently supplies pressure in Pa only;
that model-input difference must be handled and validated during each model's
reproduction rather than relabelled here.

## Success semantics

The Allen evaluator does not use expert phases. `physical_success` combines
the original screw angle/height/turn thresholds with observed gripper release,
a 50 mm TCP retreat and 0.4 s below the original 5 mm/s link-speed threshold.
It is **not** the paper/appendix success rate. `strict_success` stays `null`
until the original torque, contact, tool return, no-cam-out and two-direction
probe checks can be reproduced independently. `unsupported_checks` names every
missing check. Timeout, action failure and simulation exception remain distinct
rollout termination reasons.

The thread/seat/preload solver wrapper is an environment mechanism and remains
active for expert and policy physics. Expert approach, grasp, insertion,
turning, extraction, return, retreat and probe commands are expert behavior and
are never injected into policy execution. The post-retreat probe is therefore
currently unsupported in policy evaluation.

## Four-model input audit

| Item | Current reliable interface | Reproduction consequence |
|---|---|---|
| RGB | fixed+wrist, CHW uint8, causal 12 Hz | ready for a vision encoder |
| proprio | 18-D measured q/qd | train-split normalization still required |
| language | task-only string | ready as metadata; tokenizer/model is model-specific |
| tactile | two pressure grids in Pa | not equivalent to tactile RGB |
| action | 9-D joint target, 7 rad + 2 m, 12 Hz resampled | valid DP engineering baseline |
| windows | history 2, prediction 16 | shared offline/online contract |
| split | task/episode/seed split support | fit normalization on training episodes only |

The paper-style 10-D end-effector action is `(delta p[3], R6D[6], absolute
gripper_width[1])`. `delta p` must state whether it is world, base or tool
frame and its 12 Hz interval; R6D is the first two rotation-matrix columns (not
a 3-D rotation vector). Canonical episodes record commanded 9-D joint targets,
not commanded TCP deltas. Full-state artifacts can derive *observed motion*
between poses, but that is not the expert command. Consequently no automatic
9D-to-10D conversion is claimed. A paper-faithful dataset needs commanded TCP
targets recorded at collection time, or a separately named behavior-delta
target with this caveat.

## B1 Diffusion Policy plan

Use the authors' `real-stanford/diffusion_policy` implementation in a separate
PyTorch environment, leaving Newton/Warp/MuJoCo-Warp in the simulation
environment. Add an adapter around `EpisodeWindowDataset` rather than changing
its time semantics. Encode wrist RGB (and optionally fixed RGB) with the chosen
official image encoder, concatenate normalized 18-D proprio, and predict the
normalized 16x9 joint-target chunk conditioned on two observations. Fit robust
or min/max action and proprio statistics on the training split only and save
them in every checkpoint.

Milestones: (1) schema/dataloader-only inspection, (2) overfit one successful
episode and verify checkpoint reload determinism, (3) train on the episode-level
training split, (4) restore model plus normalization stats, (5) run receding
horizon inference through `replan`/`step_next`, and (6) report strict QA only
when available, otherwise report physical subset separately. The RTX 4090 is
ample for the initial image-UNet baseline; batch size and image augmentation
should be selected empirically in the isolated environment. RDP, pi0.5 and
FTP-1 algorithm details are intentionally deferred until their official code
and exact input/action contracts are reviewed.

## Commands (do not run during implementation)

```bash
# 5 policy periods, dual 320x240 RGB, causal history=2
python -m force_bench.policy.verify_closed_loop \
  tighten_screw_with_allen_key_from_table --steps 5

# One expert episode: dual RGB + proprio + 9-D actions + two GelSight pressure grids
TB_COLLECT_RGB=1 TB_COLLECT_RGB_CAMERAS=both \
TB_COLLECT_RGB_FPS=12 TB_COLLECT_RGB_WIDTH=320 TB_COLLECT_RGB_HEIGHT=240 \
force-bench collect --task tighten_screw_with_allen_key_from_table \
  --episodes 1 --seed 0 --gelsight --output outputs/datasets/allen_multimodal

# SHA/timestamp/schema validation followed by tactile statistics/alignment
python -m force_bench.data.inspect outputs/datasets/allen_multimodal
python -m force_bench.data.inspect_multimodal \
  outputs/datasets/allen_multimodal/<episode-id>

# Evaluation is printed on every step; this includes timeout and the
# strict_success/physical_success/unsupported_checks split.
TB_GELSIGHT=1 python -m force_bench.policy.verify_closed_loop \
  tighten_screw_with_allen_key_from_table --steps 10 --tactile

# Suggested CPU unit tests
pytest -q packages/force-bench/tests/test_policy_interfaces.py \
  packages/force-bench/tests/test_multimodal_policy.py \
  packages/force-bench/tests/test_data_collection.py \
  packages/force-bench/tests/test_dataset.py
```
