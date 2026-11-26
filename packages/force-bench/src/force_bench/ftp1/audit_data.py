"""Structural and causal audit for ContactBench FTP-1 Zarr datasets."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

REQUIRED = ("timestamps", "right_wrist_camera_rgb", "right_wrist_pose",
            "right_hand_joints", "right_hand_joints_idx",
            "right_tactile_data_gripper", "right_tactile_area_gripper",
            "right_tactile_sensor_gripper", "right_tactile_type_gripper",
            "sub_task_instruction")

def audit(path):
    from openpi.replay_buffer import ReplayBuffer
    data = ReplayBuffer.create_from_path(str(path), mode="r")
    missing = [k for k in REQUIRED if k not in data.data]
    if missing:
        raise ValueError(f"missing FTP-1 fields: {missing}")
    total = 0
    for key, value in data.data.items():
        if len(value) < 1:
            raise ValueError(f"empty Zarr array: {key}")
    ts = np.asarray(data.data["timestamps"])
    if ts.ndim != 1 or not np.isfinite(ts).all():
        raise ValueError("timestamps must be finite [T]")
    tactile = data.data["right_tactile_data_gripper"]
    if tactile.ndim != 5 or tactile.shape[1:] != (2, 240, 320, 3):
        raise ValueError(f"expected native tactile RGB [T,2,240,320,3], got {tactile.shape}")
    if not np.array_equal(np.asarray(data.data["right_tactile_area_gripper"][0]), [0, 1]):
        raise ValueError("GelSight Mini area order must be [gel_left, gel_right] -> [0,1]")
    if set(np.asarray(data.data["right_tactile_sensor_gripper"]).astype(str)) != {"GelSightMini"}:
        raise ValueError("unexpected FTP-1 sensor id")
    if set(np.asarray(data.data["right_tactile_type_gripper"]).astype(str)) != {"image"}:
        raise ValueError("tactile type must be image")
    if np.any(np.asarray(data.data["right_hand_joints_idx"]) != 28):
        raise ValueError("right gripper must map to canonical hand slot 28")
    if not np.all(np.asarray(data.data["sub_task_instruction"]).astype(str) == "pick up the thin object on the table."):
        raise ValueError("instruction mismatch")
    total = int(len(ts))
    print(f"FTP-1 audit passed: {total} 24Hz samples; sensor=GelSightMini; areas=[0,1]")
    return total

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); a=p.parse_args()
    audit(Path(a.data))

if __name__ == "__main__":
    main()
