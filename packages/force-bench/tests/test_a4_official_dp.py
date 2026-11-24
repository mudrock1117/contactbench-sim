import numpy as np
import pytest
from force_bench.data.timebase import PolicyFrameScheduler, PolicyTimeConfig, causal_indices
from force_bench.robot.eef_action import decode_relative_eef, encode_relative_eef, relative_eef_history, rotation_6d_to_matrix

def _pose(x=0.0,y=0.0,z=0.0,angle=0.0):
    c,s=np.cos(angle),np.sin(angle); T=np.eye(4)
    T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]; T[:3,3]=[x,y,z]; return T

@pytest.mark.parametrize("hz,expected",[(12,[5,5,5,5]),(24,[3,2,3,2])])
def test_policy_scheduler_supports_12_and_24_hz(hz,expected):
    scheduler=PolicyFrameScheduler(PolicyTimeConfig(60,hz,2,16))
    assert [scheduler.pop() for _ in range(4)]==expected

def test_causal_sampling_never_uses_future():
    source=np.arange(0,0.5,1/60); query=np.arange(0,0.5,1/24)
    idx=causal_indices(source,query); assert np.all(source[idx]<=query+1e-12)

def test_10d_eef_roundtrip_and_fixed_anchor():
    anchor=_pose(0.2,-0.1,0.4,0.3); targets=[_pose(0.21,-0.09,0.39,a) for a in (0.35,0.4)]
    for action,target in zip([encode_relative_eef(anchor,t,0.03) for t in targets],targets):
        decoded,width=decode_relative_eef(anchor,action)
        np.testing.assert_allclose(decoded,target,atol=1e-6); assert width==pytest.approx(0.03)
        R=rotation_6d_to_matrix(action[3:9]); np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-6)

def test_relative_history_is_10d_and_anchored_at_current():
    poses=np.stack([_pose(z=0.1),_pose(z=0.2)])
    history=relative_eef_history(poses,np.array([0.04,0.03]),poses[-1])
    assert history.shape==(2,10); np.testing.assert_allclose(history[-1,:3],0,atol=1e-7)
    assert history[-1,-1]==pytest.approx(0.03)

def test_24hz_noninteger_ratio_rejects_legacy_integer_property():
    with pytest.raises(ValueError,match="physics_frames_for_step"):
        _=PolicyTimeConfig(60,24,2,16).physics_frames_per_action
