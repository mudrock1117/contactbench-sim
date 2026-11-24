"""ContactBench adapter for pinned official Diffusion Policy."""
from __future__ import annotations
import copy
import numpy as np
import torch
from force_bench.data.dataset import EpisodeWindowDataset, WindowSpec
from force_bench.data.episode import discover_episodes, load_episode
from force_bench.data.timebase import PolicyTimeConfig, causal_indices
from force_bench.robot.eef_action import encode_relative_eef, relative_eef_history
from force_bench.dp.official import activate_upstream
activate_upstream()
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

def _pose_matrix(xyzw):
    xyzw=np.asarray(xyzw,dtype=np.float64); p=xyzw[:3]; x,y,z,w=xyzw[3:]
    n=x*x+y*y+z*z+w*w
    if n<1e-12: raise ValueError("zero quaternion in recorded TCP pose")
    s=2.0/n
    R=np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=p; return T

class ContactBenchOfficialDPDataset(BaseImageDataset):
    """Index 1 is current action because upstream slices at n_obs_steps-1."""
    REQUIRED=("action_anchor_tcp_pose_world_xyzw","action_tcp_target_world_xyzw",
              "action_gripper_width_m","action_anchor_gripper_width_m")
    def __init__(self,data_path,policy_hz=12,observation_horizon=2,horizon=16,
                 camera="wrist",success_only=True,seed=0,val_ratio=0.0):
        super().__init__(); self.seed=int(seed); self.val_ratio=float(val_ratio)
        timing=PolicyTimeConfig(60,float(policy_hz),int(observation_horizon),int(horizon))
        spec=WindowSpec(observation_horizon=int(observation_horizon),action_horizon=int(horizon),
                        camera_name=str(camera),include_privileged=False,policy_time_config=timing)
        paths=discover_episodes(data_path,success_only=bool(success_only))
        if not paths: raise ValueError(f"no episodes under {data_path}")
        self.base=EpisodeWindowDataset(paths,spec); self.episodes={}
        for path in paths:
            ep=load_episode(path); missing=[k for k in self.REQUIRED if k not in ep.arrays]
            if missing:
                raise ValueError(f"{path}: missing exact 10D fields {missing}; re-collect with A4 fields. "
                                 "Measured joint deltas are not expert EEF commands.")
            self.episodes[str(ep.metadata["episode_id"])]=ep
        self.timing=timing; self.horizon=int(horizon)
    def __len__(self): return len(self.base)
    def __getitem__(self,index):
        sample=self.base[index]; ep=self.episodes[str(sample["episode_id"])]
        current=float(sample["timestamp"][-1])
        query=current+(np.arange(self.horizon)-(self.timing.observation_horizon-1))*self.timing.policy_dt
        idx=causal_indices(ep["timestamps"][:-1],query)
        if np.any(idx<0) or np.any(idx>=len(ep["actions"])):
            raise IndexError("official action horizon crosses episode boundary")
        current_idx=int(causal_indices(ep["timestamps"][:-1],[current])[0])
        anchor=_pose_matrix(ep["action_anchor_tcp_pose_world_xyzw"][current_idx])
        actions=np.stack([encode_relative_eef(anchor,_pose_matrix(ep["action_tcp_target_world_xyzw"][i]),
                                              ep["action_gripper_width_m"][i]) for i in idx])
        hidx=causal_indices(ep["timestamps"][:-1],np.asarray(sample["timestamp"]))
        hposes=np.stack([_pose_matrix(ep["action_anchor_tcp_pose_world_xyzw"][i]) for i in hidx])
        widths=np.asarray(ep["action_anchor_gripper_width_m"])[hidx]
        agent=relative_eef_history(hposes,widths,anchor)
        image=np.asarray(sample["rgb"],dtype=np.float32)/255.0
        return {"obs":{"wrist_rgb":torch.from_numpy(image),
                       "agent_pos":torch.from_numpy(agent.astype(np.float32))},
                "action":torch.from_numpy(actions.astype(np.float32))}
    def get_validation_dataset(self):
        return copy.copy(self)
    def get_normalizer(self,mode="limits",**kwargs):
        actions=[]; states=[]
        for i in range(len(self)):
            item=self[i]; actions.append(item["action"].numpy()); states.append(item["obs"]["agent_pos"].numpy())
        normalizer=LinearNormalizer()
        normalizer.fit({"action":np.concatenate(actions),"agent_pos":np.concatenate(states)},
                       last_n_dims=1,mode=mode,**kwargs)
        normalizer["wrist_rgb"].set_identity(); return normalizer
    def get_all_actions(self):
        return torch.cat([self[i]["action"] for i in range(len(self))],dim=0)
