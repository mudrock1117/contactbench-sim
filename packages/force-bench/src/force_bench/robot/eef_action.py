"""Pure NumPy A4 10D relative-EEF representation.

This module is intentionally safe in training-only environments: importing it
must not import Warp, Newton, MuJoCo, geometry, or online action adapters.
"""
from __future__ import annotations
import numpy as np

def rotation_matrix_to_6d(rotation):
    """First two rotation-matrix columns, concatenated column-major."""
    rotation=np.asarray(rotation,dtype=np.float64)
    if rotation.shape!=(3,3): raise ValueError("rotation must have shape [3,3]")
    return rotation[:,:2].T.reshape(6).astype(np.float32)

def rotation_6d_to_matrix(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(6,) or not np.isfinite(value).all():
        raise ValueError("rotation 6D must be finite shape [6]")
    a1,a2=value[:3],value[3:]; n1=np.linalg.norm(a1)
    if n1<1e-8: raise ValueError("rotation 6D first axis is degenerate")
    b1=a1/n1; a2=a2-np.dot(b1,a2)*b1; n2=np.linalg.norm(a2)
    if n2<1e-8: raise ValueError("rotation 6D second axis is degenerate")
    b2=a2/n2
    return np.stack((b1,b2,np.cross(b1,b2)),axis=1)

def _matrix_to_rotvec(rotation):
    angle=float(np.arccos(np.clip((np.trace(rotation)-1.0)/2.0,-1.0,1.0)))
    if angle<1e-9: return np.zeros(3)
    axis=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],
                   rotation[1,0]-rotation[0,1]])/(2.0*np.sin(angle))
    return axis*angle

def pose_xyzw_to_matrix(pose):
    pose=np.asarray(pose,dtype=np.float64)
    if pose.shape!=(7,): raise ValueError("pose must be xyz+xyzw shape [7]")
    x,y,z,w=pose[3:]; n=x*x+y*y+z*z+w*w
    if n<1e-12: raise ValueError("pose quaternion is degenerate")
    s=2.0/n; transform=np.eye(4); transform[:3,3]=pose[:3]
    transform[:3,:3]=[[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                      [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                      [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]]
    return transform

def encode_relative_eef(anchor_world,target_world,gripper_width_m):
    anchor_world=np.asarray(anchor_world,dtype=np.float64)
    target_world=np.asarray(target_world,dtype=np.float64)
    if anchor_world.shape!=(4,4) or target_world.shape!=(4,4):
        raise ValueError("poses must be [4,4]")
    relative=np.linalg.inv(anchor_world)@target_world
    return np.r_[relative[:3,3],rotation_matrix_to_6d(relative[:3,:3]),
                 float(gripper_width_m)].astype(np.float32)

def decode_relative_eef(anchor_world,action):
    action=np.asarray(action,dtype=np.float64)
    if action.shape!=(10,) or not np.isfinite(action).all():
        raise ValueError("relative EEF action must be finite shape [10]")
    relative=np.eye(4); relative[:3,3]=action[:3]
    relative[:3,:3]=rotation_6d_to_matrix(action[3:9])
    return np.asarray(anchor_world,dtype=np.float64)@relative,float(action[9])

def relative_eef_history(poses_world,widths_m,anchor_world):
    poses_world=np.asarray(poses_world,dtype=np.float64)
    widths_m=np.asarray(widths_m,dtype=np.float64)
    if poses_world.ndim!=3 or poses_world.shape[1:]!=(4,4) or widths_m.shape!=(len(poses_world),):
        raise ValueError("history poses/widths have incompatible shapes")
    return np.stack([encode_relative_eef(anchor_world,pose,width)
                     for pose,width in zip(poses_world,widths_m)])
