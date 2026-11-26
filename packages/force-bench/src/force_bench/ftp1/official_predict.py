"""Offline inference on real ContactBench Zarr observations."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .official import load_policy, predict

def main():
    import zarr
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = zarr.open_group(a.data, mode="r")
    d = root["data"]
    i = a.index
    ends = np.asarray(root["meta/episode_ends"][:])
    if i < 0 or i + 32 >= d["right_wrist_pose"].shape[0] or not np.any((ends > i) & (ends > i + 32)):
        raise ValueError("invalid window")
    if i + 32 >= ends[np.searchsorted(ends, i, side="right")]:
        raise ValueError("window crosses episode")
    wrapper = load_policy(a.checkpoint)
    raw, actions = predict(wrapper, d["right_wrist_camera_rgb"][i],
        d["right_tactile_data_gripper"][i], float(d["right_hand_joints"][i,0]))
    poses = np.asarray(d["right_wrist_pose"][i:i+33])
    anchor = Rotation.from_rotvec(poses[0,3:])
    target_rot = anchor.inv() * Rotation.from_rotvec(poses[1:,3:])
    target_pos = anchor.inv().apply(poses[1:,:3] - poses[0,:3])
    widths = np.asarray(d["right_hand_joints"][i+1:i+33,0])
    x = actions[:,3:6]
    y = actions[:,6:9]
    x = x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    y = y - (x*y).sum(axis=1,keepdims=True)*x
    y = y / np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-8)
    pred_rot = Rotation.from_matrix(np.stack([x,y,np.cross(x,y)],axis=-1))
    np.savez_compressed(a.output, raw=raw, actions=actions, target_position=target_pos,
                        target_rotation=target_rot.as_matrix(), target_width=widths)
    print(json.dumps({"raw_shape":list(raw.shape),"action_shape":list(actions.shape),
        "translation_mae_m":float(np.abs(actions[:,:3]-target_pos).mean()),
        "rotation_mean_rad":float((pred_rot.inv()*target_rot).magnitude().mean()),
        "width_mae_m":float(np.abs(actions[:,9]-widths).mean())}))

if __name__ == "__main__":
    main()
