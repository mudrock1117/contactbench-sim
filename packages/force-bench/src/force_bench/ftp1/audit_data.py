"""Structural, temporal, and optical-content audit for FTP-1 Zarr datasets."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .data import TIMESTAMP_ATOL

REQUIRED = (
    "timestamps",
    "right_wrist_camera_rgb",
    "right_wrist_pose",
    "right_hand_joints",
    "right_hand_joints_idx",
    "right_tactile_data_gripper",
    "right_tactile_area_gripper",
    "right_tactile_sensor_gripper",
    "right_tactile_type_gripper",
    "sub_task_instruction",
    "contactbench_time_alignment",
    "contactbench_observation_timestamps",
    "contactbench_wrist_rgb_timestamps",
    "contactbench_tactile_rgb_timestamps",
)
PROMPT = "pick up the thin object on the table."
POLICY_HZ = 24.0
NATIVE_TACTILE_SHAPE = (2, 320, 240, 3)


def _array(data, key):
    return np.asarray(data[key][:])


def validate_source_age(ts, source, name, sensor_hz, physics_hz=60.0):
    """Bound causal age by the maximum quantized sensor period."""
    limit = math.ceil(physics_hz / sensor_hz) / physics_hz
    if source.shape != ts.shape or not np.isfinite(source).all():
        raise ValueError(f"{name}: causal source timestamps must be finite [T]")
    age = ts - source
    bad = np.flatnonzero((age < -TIMESTAMP_ATOL) | (age > limit + TIMESTAMP_ATOL))
    if len(bad):
        i = int(bad[0])
        raise ValueError(
            f"{name}: source timestamp is future-dated or exceeds its sampling-age bound; "
            f"row={i}, policy={ts[i]:.12f}, source={source[i]:.12f}, "
            f"age={age[i]:.12f}s, limit={limit:.12f}s"
        )


def audit(path):
    from openpi.replay_buffer import ReplayBuffer

    replay = ReplayBuffer.create_from_path(str(path), mode="r")
    data = replay.data
    keys = set(data.keys())
    missing = [key for key in REQUIRED if key not in keys]
    if missing:
        raise ValueError(f"missing FTP-1 fields: {missing}")
    if "meta" not in replay.root or "episode_ends" not in replay.root["meta"]:
        raise ValueError("official ReplayBuffer meta/episode_ends is required")
    ends = np.asarray(replay.root["meta"]["episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1 or not len(ends) or np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing positive boundaries")
    final_end = int(ends[-1])
    for key in keys:
        array = data[key]
        if not array.shape or array.shape[0] != final_end:
            raise ValueError(f"array length does not match final episode boundary: {key}")

    ts = _array(data, "timestamps").astype(np.float64)
    if ts.ndim != 1 or len(ts) != final_end or not np.isfinite(ts).all():
        raise ValueError("timestamps must be finite [T] and align with episode_ends")
    tactile = data["right_tactile_data_gripper"]
    if tactile.ndim != 5 or tactile.shape[1:] != NATIVE_TACTILE_SHAPE:
        raise ValueError(f"expected native Taxim uint8 [T,2,320,240,3], got {tactile.shape}")
    if np.dtype(tactile.dtype) != np.dtype(np.uint8):
        raise ValueError(f"tactile data must remain uint8, got {tactile.dtype}")
    if not np.array_equal(_array(data, "right_tactile_area_gripper")[0], [0, 1]):
        raise ValueError("area order must be [gel_left, gel_right] -> [0,1]")
    if set(_array(data, "right_tactile_sensor_gripper").astype(str)) != {"GelSightMini"}:
        raise ValueError("unexpected FTP-1 sensor ID")
    if set(_array(data, "right_tactile_type_gripper").astype(str)) != {"image"}:
        raise ValueError("tactile type must be image")
    if np.any(_array(data, "right_hand_joints_idx") != 28):
        raise ValueError("gripper channel must map to canonical hand slot 28")
    if not np.all(_array(data, "sub_task_instruction").astype(str) == PROMPT):
        raise ValueError("instruction mismatch")

    alignment = _array(data, "contactbench_time_alignment")
    if alignment.shape != (final_end, 3) or not np.issubdtype(alignment.dtype, np.integer) or np.any(alignment < 0):
        raise ValueError("alignment must be nonnegative integer [T, observation/wrist/tactile]")
    source_times = [
        _array(data, "contactbench_observation_timestamps").astype(np.float64),
        _array(data, "contactbench_wrist_rgb_timestamps").astype(np.float64),
        _array(data, "contactbench_tactile_rgb_timestamps").astype(np.float64),
    ]
    for source, name, hz in zip(
        source_times, ("observation", "wrist_rgb", "tactile_rgb"), (60.0, 12.0, POLICY_HZ)
    ):
        validate_source_age(ts, source, name, hz)

    changed_sensors = [False, False]
    start = 0
    for boundary in ends:
        end = int(boundary)
        episode_ts = ts[start:end]
        if not len(episode_ts) or (len(episode_ts) > 1 and np.any(np.diff(episode_ts) <= 0)):
            raise ValueError("timestamps must increase strictly within each episode")
        if len(episode_ts) > 1 and not np.allclose(np.diff(episode_ts), 1.0 / POLICY_HZ, atol=1e-6):
            raise ValueError("timestamps must follow the 24 Hz episode grid")
        for column, source in enumerate(source_times):
            source_episode = source[start:end]
            indices = alignment[start:end, column]
            if len(source_episode) > 1 and np.any(np.diff(source_episode) < 0):
                raise ValueError("causal source timestamps must be nondecreasing per episode")
            if len(indices) > 1 and np.any(np.diff(indices) < 0):
                raise ValueError("causal source indices must be nondecreasing per episode")
        first = np.asarray(tactile[start], dtype=np.uint8)
        episode_changed = [False, False]
        for index in range(start + 1, end):
            frame = np.asarray(tactile[index], dtype=np.uint8)
            for sensor in range(2):
                if not episode_changed[sensor] and np.any(frame[sensor] != first[sensor]):
                    episode_changed[sensor] = True
        if not any(episode_changed):
            raise ValueError(f"episode ending at row {end} has static tactile RGB")
        changed_sensors = [old or new for old, new in zip(changed_sensors, episode_changed)]
        start = end

    if not any(changed_sensors):
        raise ValueError("both tactile RGB streams are static; refusing dataset as non-optical/empty")
    print(
        f"FTP-1 audit passed: {final_end} samples, {len(ends)} episodes; "
        f"native=320x240 uint8; pad image-motion={changed_sensors}"
    )
    return final_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    audit(Path(args.data))


if __name__ == "__main__":
    main()
