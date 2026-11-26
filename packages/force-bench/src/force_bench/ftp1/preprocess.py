"""ContactBench <-> pinned FTP-1 inference wrapper preprocessing."""
from __future__ import annotations

import numpy as np

PROMPT = "pick up the thin object on the table."
TACTILE_KEY = "right_tactile_gripper"
TACTILE_SENSOR = "GelSightMini"
TACTILE_AREAS = [0, 1]
RIGHT_GRIPPER_SLOT = 44  # right hand slot 28 at canonical UAS offset 16

def prepare_observation(wrapper, wrist_rgb, tactile_rgb, gripper_width_m):
    """Build wrapper inputs using official OpenPI resize and FTP-1 normalization.

    The policy-chunk TCP anchor is the current measured TCP. Thus the current
    relative wrist state is identity (9D); the scalar gripper width is absolute.
    FTP1InferenceWrapper performs its configured domain/shared normalization.
    """
    from openpi.shared.image_tools import resize_with_pad
    import cv2
    wrist = np.asarray(wrist_rgb)
    tactile = np.asarray(tactile_rgb)
    if wrist.dtype != np.uint8 or wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError("wrist RGB must be uint8 HxWx3")
    if tactile.dtype != np.uint8 or tactile.shape != (2, 240, 320, 3):
        raise ValueError("GelSightMini RGB must be uint8 [2,240,320,3]")
    wrist_224 = np.asarray(resize_with_pad(wrist, 224, 224), dtype=np.uint8)
    tactile_224 = np.stack([cv2.resize(x, (224, 224)) for x in tactile]).astype(np.uint8)
    state = np.zeros((1, wrapper.get_state_dim()), dtype=np.float32)
    if state.shape[1] != 120:
        raise ValueError(f"locked FTP-1 default UAS requires 120D state, got {state.shape[1]}")
    state[0, :9] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    state[0, RIGHT_GRIPPER_SLOT] = float(gripper_width_m)
    return {
        "images": {"right_wrist_camera_rgb_0": wrist_224},
        "state": state,
        "prompt": PROMPT,
        "tactiles": {TACTILE_KEY: tactile_224[None].astype(np.float32)},
        "tactile_function_areas": {TACTILE_KEY: TACTILE_AREAS},
        "tactile_sensors": {TACTILE_KEY: TACTILE_SENSOR},
    }


def decode_action_chunk(actions):
    """Select FTP-1 right TCP pose9D + absolute gripper slot into A4's 10D contract.

    Pose is already relative to the inference anchor because action_pose_rep=relative;
    do not apply another relative-pose transform here.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 120 or actions.shape[0] != 32:
        raise ValueError("expected official FTP-1 [32,120] denormalized action chunk")
    return np.concatenate((actions[:, :9], actions[:, RIGHT_GRIPPER_SLOT:RIGHT_GRIPPER_SLOT + 1]), axis=1)
