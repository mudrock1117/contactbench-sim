"""ContactBench <-> pinned FTP-1 inference wrapper preprocessing."""
from __future__ import annotations

import numpy as np

PROMPT = "pick up the thin object on the table."
TACTILE_KEY = "right_tactile_gripper"
TACTILE_SENSOR = "GelSightMini"
TACTILE_AREAS = [0, 1]
RIGHT_GRIPPER_HAND_SLOT = 28

def _right_action_slices(state_dim):
    """Resolve right-arm fields from the pinned FTP-1 UAS layout API."""
    from openpi.dataset_zarr import FTP1_GRIPPER_HAND_SLOT_INDEX
    from openpi.ftp1_action_groups import get_ftp1_action_group_slices
    from openpi.models_pytorch.ftp1_model_config import (
        FTP1_RESERVED_ACTION_DIM,
        FTP1_SINGLE_ARM_ACTION_REP_DIM,
        FTP1_SINGLE_ARM_JOINT_DIM,
    )

    groups = get_ftp1_action_group_slices(
        FTP1_SINGLE_ARM_ACTION_REP_DIM,
        FTP1_SINGLE_ARM_JOINT_DIM,
        FTP1_RESERVED_ACTION_DIM,
    )
    if 2 * FTP1_SINGLE_ARM_ACTION_REP_DIM + 9 + FTP1_RESERVED_ACTION_DIM != state_dim:
        raise ValueError(f"unsupported FTP-1 UAS dimension: {state_dim}")
    position = groups["right-wrist-pos"]
    rotation = groups["right-wrist-rot"]
    hand = groups["right-hand-joints"]
    slot = FTP1_GRIPPER_HAND_SLOT_INDEX
    if position[1] != rotation[0] or hand[1] - hand[0] <= slot or slot != RIGHT_GRIPPER_HAND_SLOT:
        raise ValueError("pinned FTP-1 layout does not expose right gripper hand slot 28")
    return slice(*position), slice(*rotation), hand[0] + slot

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
    if tactile.dtype != np.uint8 or tactile.shape != (2, 320, 240, 3):
        raise ValueError("GelSightMini RGB must be uint8 [2,320,240,3]")
    wrist_224 = np.asarray(resize_with_pad(wrist, 224, 224), dtype=np.uint8)
    tactile_224 = np.stack([cv2.resize(x, (224, 224)) for x in tactile]).astype(np.uint8)
    state = np.zeros((1, wrapper.get_state_dim()), dtype=np.float32)
    if state.shape[1] != 120:
        raise ValueError(f"locked FTP-1 default UAS requires 120D state, got {state.shape[1]}")
    pos_slice, rot_slice, gripper_slot = _right_action_slices(state.shape[1])
    state[0, pos_slice] = (0.0, 0.0, 0.0)
    state[0, rot_slice] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    state[0, gripper_slot] = float(gripper_width_m)
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
    pos_slice, rot_slice, gripper_slot = _right_action_slices(actions.shape[1])
    return np.concatenate(
        (actions[:, pos_slice], actions[:, rot_slice], actions[:, gripper_slot:gripper_slot + 1]),
        axis=1,
    )
