"""Causal ContactBench episode conversion helpers for the official FTP-1 Zarr loader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POLICY_HZ = 24.0
PROMPT = "pick up the thin object on the table."
PAD_ORDER = ("gel_left", "gel_right")

def causal_indices(source_timestamps, target_timestamps):
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or not len(source):
        raise ValueError("timestamps must be nonempty vectors")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("timestamps must be finite")
    if len(source) > 1 and np.any(np.diff(source) <= 0):
        raise ValueError("source timestamps must increase strictly")
    indices = np.searchsorted(source, target, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("no causal sample exists at or before a requested time")
    if np.any(source[indices] > target + 1e-12):
        raise AssertionError("future observation leakage")
    return indices.astype(np.int64)


def _read_rgb_stream(root, relative):
    import imageio.v2 as iio
    directory = Path(root) / relative
    times = np.load(directory / "timestamps.npy", allow_pickle=False)
    files = sorted((directory / "images").glob("*.png"))
    if len(files) != len(times) or not len(files):
        raise ValueError(f"incomplete RGB stream: {directory}")
    images = [np.asarray(iio.imread(p), dtype=np.uint8) for p in files]
    return times, images


def _quat_to_rotvec(quat_xyzw):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_rotvec()


def episode_to_ftp1(episode_path):
    from force_bench.data.episode import load_episode
    episode = load_episode(episode_path)
    meta, arrays = episode.metadata, episode.arrays
    if meta.get("task") != "retrieve_card_from_deck":
        raise ValueError("FTP-1 B4.2 converter only accepts retrieve_card_from_deck")
    if meta.get("outcome", {}).get("success") is not True:
        raise ValueError("failed/unvalidated episodes are not trainable FTP-1 input")
    if "eef_action_fields" not in meta or "action_anchor_tcp_pose_world_xyzw" not in arrays:
        raise ValueError("episode lacks the A4 fixed-reference TCP command audit fields")
    if "action_tcp_target_world_xyzw" not in arrays or "action_gripper_width_m" not in arrays:
        raise ValueError("episode lacks exact A4 command targets")
    if "tactile_rgb" not in meta:
        raise ValueError("episode lacks opt-in real tactile RGB")
    timestamps = arrays["timestamps"]
    if timestamps.dtype != np.float64 or len(timestamps) != len(arrays["actions"]) + 1:
        raise ValueError("episode timestamp/action alignment is invalid")
    root = Path(episode_path)
    tactile = []
    tactile_times = None
    for pad in PAD_ORDER:
        ts, images = _read_rgb_stream(root, f"rgb/tactile/{pad}")
        if tactile_times is None:
            tactile_times = ts
        elif not np.array_equal(ts, tactile_times):
            raise ValueError("left/right tactile RGB timestamps must align")
        tactile.append(images)
    wrist_times, wrist_images = _read_rgb_stream(root, "rgb/wrist")
    # Policy grid begins at the first optical observation, then uses exact 24Hz
    # ticks. Each state/image lookup is latest-at-or-before-tick only.
    end = float(timestamps[-2])
    start = float(tactile_times[0])
    n = int(np.floor((end - start) * POLICY_HZ + 1e-9)) + 1
    times = start + np.arange(n, dtype=np.float64) / POLICY_HZ
    obs_idx = causal_indices(timestamps[:-1], times)
    wrist_idx = causal_indices(wrist_times, times)
    tactile_idx = causal_indices(tactile_times, times)
    obs = arrays["observations"][obs_idx]
    anchors = arrays["action_anchor_tcp_pose_world_xyzw"][obs_idx]
    gripper = obs[:, 7:9].sum(axis=1).astype(np.float32)
    if not np.isfinite(gripper).all():
        raise ValueError("gripper width contains non-finite values")
    # FTP-1's public Zarr contract has one world-frame wrist trajectory. The
    # pinned loader derives both proprioception and future pose targets from it.
    # Keep the exact A4 commanded target separately for auditable provenance.
    world_pose = np.concatenate((anchors[:, :3], _quat_to_rotvec(anchors[:, 3:])), axis=1).astype(np.float32)
    command = arrays["action_tcp_target_world_xyzw"][obs_idx]
    command_rotvec = _quat_to_rotvec(command[:, 3:])
    command_pose = np.concatenate((command[:, :3], command_rotvec), axis=1).astype(np.float32)
    command_width = arrays["action_gripper_width_m"][obs_idx].reshape(-1, 1).astype(np.float32)
    return {
        "timestamps": times,
        "right_wrist_camera_rgb": np.stack([wrist_images[i] for i in wrist_idx]),
        "right_wrist_pose": world_pose,
        "right_hand_joints": gripper[:, None],
        "right_hand_joints_idx": np.full((n, 1), 28, dtype=np.int32),
        "right_tactile_data_gripper": np.stack([
            np.stack([tactile[0][i], tactile[1][i]], axis=0) for i in tactile_idx
        ]),
        "right_tactile_area_gripper": np.tile(np.asarray([0, 1], dtype=np.int32), (n, 1)),
        "right_tactile_sensor_gripper": np.full(n, "GelSightMini", dtype="<U16"),
        "right_tactile_type_gripper": np.full(n, "image", dtype="<U8"),
        "sub_task_instruction": np.full(n, PROMPT, dtype="<U64"),
        "contactbench_a4_action_target_world_xyz_rotvec": command_pose,
        "contactbench_a4_action_gripper_width_m": command_width,
        "contactbench_time_alignment": np.stack((obs_idx, wrist_idx, tactile_idx), axis=1).astype(np.int64),
    }
