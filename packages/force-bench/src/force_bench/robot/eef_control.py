"""Online execution adapter for the pure A4 10D EEF representation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .actions import EndEffectorActionAdapter, ActionResult
from .eef_action import _matrix_to_rotvec, decode_relative_eef

@dataclass
class FixedEEFChunkAnchor:
    world_from_tcp: np.ndarray
    timestamp: float

class RelativeEEF10DActionAdapter:
    def __init__(self,rig,config):
        self.base=EndEffectorActionAdapter(rig,config)
        self.anchor=None

    def set_chunk_anchor(self,timestamp=0.0):
        self.anchor=FixedEEFChunkAnchor(self.base.actual_tcp_transform().copy(),float(timestamp))
        return self.anchor

    def apply(self,action,*,control_dt):
        if self.anchor is None:
            return ActionResult(False,None,reason="10D EEF chunk has no fixed anchor")
        try:
            target,width=decode_relative_eef(self.anchor.world_from_tcp,action)
        except ValueError as error:
            return ActionResult(False,None,reason=str(error))
        current=self.base.actual_tcp_transform()
        world_delta=target[:3,3]-current[:3,3]
        tool_delta=_matrix_to_rotvec(current[:3,:3].T@target[:3,:3])
        return self.base.apply(np.r_[world_delta,tool_delta,width],control_dt=control_dt)

    def actual_tcp_transform(self):
        return self.base.actual_tcp_transform()

    def actual_tcp_pose(self):
        return self.base.actual_tcp_pose()
