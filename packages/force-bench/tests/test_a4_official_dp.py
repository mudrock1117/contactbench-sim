import numpy as np
import pytest
from force_bench.data.timebase import PolicyFrameScheduler, PolicyTimeConfig, causal_indices
from force_bench.robot.eef_action import decode_relative_eef, encode_relative_eef, relative_eef_history, rotation_6d_to_matrix

def _pose(x=0.0,y=0.0,z=0.0,angle=0.0):
    c,s=np.cos(angle),np.sin(angle); T=np.eye(4)
    T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]; T[:3,3]=[x,y,z]; return T

@pytest.mark.parametrize("hz,expected",[(12,[5,5,5,5]),(24,[2,3,2,3])])
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


def test_official_action_offset_places_current_at_index_one():
    from force_bench.dp.official_contract import official_action_query_times
    query=official_action_query_times(2.0,policy_hz=12,horizon=16,n_obs_steps=2)
    assert query.shape==(16,)
    assert query[1]==pytest.approx(2.0)
    assert query[0]<2.0 and np.all(np.diff(query)>0)

def test_eef_field_validation_rejects_missing_and_auxiliary_controls():
    from force_bench.dp.official_contract import validate_eef_episode_arrays
    count=3
    arrays={
        "action_anchor_tcp_pose_world_xyzw":np.ones((count,7)),
        "action_tcp_target_world_xyzw":np.ones((count,7)),
        "action_anchor_gripper_width_m":np.ones(count),
        "action_gripper_width_m":np.ones(count),
        "action_target_velocity":np.zeros((count,9)),
        "action_feedforward":np.zeros((count,9)),
    }
    assert validate_eef_episode_arrays(arrays,count)
    arrays["action_feedforward"][1,0]=0.01
    with pytest.raises(ValueError,match="nonzero"):
        validate_eef_episode_arrays(arrays,count)
    del arrays["action_gripper_width_m"]
    with pytest.raises(ValueError,match="missing exact command fields"):
        validate_eef_episode_arrays(arrays,count)

def test_official_workspace_configs_cover_upstream_contract():
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[3]
    for name in ("official_smoke.yaml","official_paper.yaml"):
        path=root/"packages/force-bench/config/dp"/name
        text=path.read_text()
        assert r"\${" not in text
        cfg=yaml.safe_load(text)
        assert cfg["task"]["env_runner"]["_target_"].endswith("ContactBenchOfflineImageRunner")
        assert cfg["training"]["tqdm_interval_sec"]>0
        assert cfg["checkpoint"]["save_last_ckpt"] is True
        assert "save_last_snapshot" in cfg["checkpoint"]
        assert cfg["n_obs_steps"]==2 and cfg["horizon"]==16 and cfg["n_action_steps"]==8
        assert cfg["task"]["dataset"]["policy_hz"]==12
    smoke=yaml.safe_load((root/"packages/force-bench/config/dp/official_smoke.yaml").read_text())
    assert smoke["logging"]["mode"] in ("disabled","offline")

def test_official_runner_is_upstream_base_runner_when_available(tmp_path):
    try:
        from force_bench.dp.official_runner import ContactBenchOfflineImageRunner
        from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
    except (ImportError,FileNotFoundError):
        pytest.skip("pinned upstream training environment is not installed")
    assert issubclass(ContactBenchOfflineImageRunner,BaseImageRunner)
    assert ContactBenchOfflineImageRunner(tmp_path).run(None)=={"offline_runner":1.0}
