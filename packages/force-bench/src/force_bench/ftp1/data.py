"""Causal ContactBench episode conversion helpers for the official FTP-1 Zarr loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

POLICY_HZ = 24.0
NATIVE_TACTILE_SHAPE = (320, 240, 3)
TAXIM_RENDERER = "hydrogel.render.TaximRenderer.render"
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


def _read_rgb_stream(root, relative, *, image_shape=None):
    import imageio.v2 as iio
    directory = Path(root) / relative
    times = np.load(directory / "timestamps.npy", allow_pickle=False)
    files = sorted((directory / "images").glob("*.png"))
    expected_names = [f"{i:06d}.png" for i in range(len(times))]
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))
            or [p.name for p in files] != expected_names):
        raise ValueError(f"incomplete or unordered RGB stream: {directory}")
    images = []
    for path in files:
        image = np.asarray(iio.imread(path))
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or (image_shape is not None and image.shape != image_shape)):
            raise ValueError(f"invalid RGB image shape/type in {path}: {image.shape} {image.dtype}")
        images.append(image.copy())
    return times.astype(np.float64, copy=False), images


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
    tactile_meta = meta.get("tactile_rgb", {})
    if (tactile_meta.get("enabled") is not True
            or tactile_meta.get("renderer") != TAXIM_RENDERER
            or tactile_meta.get("sensor_id") != "GelSightMini"
            or tactile_meta.get("pad_order") != list(PAD_ORDER)
            or tuple(tactile_meta.get("native_shape", ())) != NATIVE_TACTILE_SHAPE):
        raise ValueError("episode lacks real calibrated Taxim GelSightMini RGB in left/right 320x240 order")
    timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
    actions = np.asarray(arrays["actions"])
    observations = np.asarray(arrays["observations"])
    if (timestamps.ndim != 1 or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
            or len(timestamps) != len(actions) + 1
            or observations.shape[0] != len(timestamps)):
        raise ValueError("episode timestamp/action/observation alignment is invalid")
    for key in ("action_anchor_tcp_pose_world_xyzw", "action_tcp_target_world_xyzw",
                "action_gripper_width_m"):
        if len(arrays[key]) != len(actions):
            raise ValueError(f"{key} must have one command-start value per action")
    root = Path(episode_path)
    tactile = []
    tactile_times = None
    for pad in PAD_ORDER:
        ts, images = _read_rgb_stream(root, f"rgb/tactile/{pad}", image_shape=NATIVE_TACTILE_SHAPE)
        if tactile_times is None:
            tactile_times = ts
        elif not np.array_equal(ts, tactile_times):
            raise ValueError("left/right tactile RGB timestamps must align")
        tactile.append(images)
    if len(tactile_times) < 2 or not any(
            np.any(frame != images[0]) for images in tactile for frame in images[1:]):
        raise ValueError("optical tactile RGB is static; refusing conversion")
    wrist_times, wrist_images = _read_rgb_stream(root, "rgb/wrist")
    if (tactile_times[0] < timestamps[0] - 1e-9 or tactile_times[-1] > timestamps[-1] + 1e-9
            or wrist_times[0] < timestamps[0] - 1e-9 or wrist_times[-1] > timestamps[-1] + 1e-9):
        raise ValueError("RGB timestamps fall outside episode timestamp bounds")
    # Policy grid begins at the first optical observation, then uses exact 24Hz
    # ticks. Each state/image lookup is latest-at-or-before-tick only.
    end = float(timestamps[-2])
    start = float(tactile_times[0])
    n = int(np.floor((end - start) * POLICY_HZ + 1e-9)) + 1
    times = start + np.arange(n, dtype=np.float64) / POLICY_HZ
    obs_idx = causal_indices(timestamps[:-1], times)
    wrist_idx = causal_indices(wrist_times, times)
    tactile_idx = causal_indices(tactile_times, times)
    obs = observations[obs_idx]
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
        "contactbench_observation_timestamps": timestamps[:-1][obs_idx],
        "contactbench_wrist_rgb_timestamps": wrist_times[wrist_idx],
        "contactbench_tactile_rgb_timestamps": tactile_times[tactile_idx],
    }
