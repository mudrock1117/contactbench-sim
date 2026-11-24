"""CPU-only contracts for A3 multimodal timing and calibration."""
from types import SimpleNamespace

import numpy as np
import pytest

from force_bench.data.rgb import (LiveRGBCamera, camera_lookat_matrix,
                                  wrist_mount_from_tcp)
from force_bench.data.task_language import task_instruction
from force_bench.data.timebase import (ActionChunkExecutor, PolicyTimeConfig,
                                       causal_indices, policy_action_ticks,
                                       resample_joint_actions)
from force_bench.data.compare_wrist_mounts import (DEFAULT_CANDIDATES, load_candidates,
                                                    load_full_states)
from force_bench.policy.evaluation import AllenKeyPolicyEvaluator


class Array:
    def __init__(self, value): self.value = np.asarray(value, dtype=float)
    def numpy(self): return self.value.copy()


class FakeCamera:
    def __init__(self, model, position, target, width, height, fov_deg):
        self.width, self.height = width, height
        self.T = np.eye(4); self.T[:3, 3] = position
    def set_transform(self, transform): self.T = np.asarray(transform).copy()
    def transform(self): return self.T.copy()
    def render(self, state):
        return np.full((self.height, self.width, 3), int(round(self.T[0, 3] * 10)) % 255,
                       dtype=np.uint8)


def _camera_rig(x=0.0):
    pose = np.array([[x, 0, 0, 0, 0, 0, 1]], dtype=float)
    return SimpleNamespace(fps=60, model=SimpleNamespace(body_label=["fr3/fr3_link7"]),
                           s0=SimpleNamespace(body_q=Array(pose)))


def test_wrist_camera_uses_actual_link_pose_and_fixed_camera_does_not_move():
    rig = _camera_rig()
    fixed = LiveRGBCamera(rig, "fixed", {"position": [1, 2, 3], "target": [0, 0, 0]},
                          width=4, height=3, camera_factory=FakeCamera)
    wrist = LiveRGBCamera(rig, "wrist", {"link_label": "fr3/fr3_link7",
                          "T_link_camera": np.eye(4)}, width=4, height=3,
                          camera_factory=FakeCamera)
    fixed.sample(rig.s0, 0, force=True); wrist.sample(rig.s0, 0, force=True)
    rig.s0.body_q.value[0, 0] = 0.5
    fixed.sample(rig.s0, 1 / 12); wrist.sample(rig.s0, 1 / 12)
    assert fixed.observe(1 / 12)["T_world_camera"][0, 3] == 1
    assert wrist.observe(1 / 12)["T_world_camera"][0, 3] == pytest.approx(0.5)
    assert wrist.observe(1 / 12)["rgb"].shape == (3, 3, 4)
    assert wrist.observe(1 / 12)["rgb"].dtype == np.uint8
    assert fixed.observe(1 / 12)["rgb_timestamp"] == wrist.observe(1 / 12)["rgb_timestamp"]


def test_wrist_mount_optical_minus_z_points_along_tcp_tool_axis():
    position = np.array([0.055, 0.0, -0.035])
    target = np.array([0.0, 0.0, 0.110])
    T = wrist_mount_from_tcp(np.eye(4), position, target, up_tcp=[0, 1, 0])
    expected = (target - position) / np.linalg.norm(target - position)
    np.testing.assert_allclose(-T[:3, 2], expected, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], position)
    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_camera_lookat_rejects_degenerate_up_axis():
    with pytest.raises(ValueError, match="parallel"):
        camera_lookat_matrix([0, 0, 0], [0, 0, 1], up=[0, 0, 1])


def test_wrist_candidate_defaults_match_a3_3_specification():
    candidates = load_candidates()
    assert tuple(candidates) == ("A", "B", "C", "D")
    np.testing.assert_allclose(candidates["A"]["position"], [0.055, 0, -0.035])
    np.testing.assert_allclose(candidates["B"]["position"], [0.080, 0, -0.035])
    np.testing.assert_allclose(candidates["C"]["target"], [0, 0, 0.140])
    assert candidates["A"]["fov_deg"] == candidates["C"]["fov_deg"] == 72
    assert candidates["D"]["fov_deg"] == 80
    assert DEFAULT_CANDIDATES["D"]["target"] == [0.0, 0.0, 0.140]


def test_full_state_loader_requires_exact_renderer_state_and_ignores_object_labels(tmp_path):
    artifacts = tmp_path / "artifacts/task"
    artifacts.mkdir(parents=True)
    path = artifacts / "rollout_states.npz"
    np.savez(path, t=np.array([0.1, 0.2]), body_q=np.zeros((2, 3, 7)),
             body_qd=np.zeros((2, 3, 6)), body_labels=np.array(["a", "b", "c"], dtype=object))
    loaded, times, body_q, body_qd = load_full_states(tmp_path)
    assert loaded == path
    np.testing.assert_array_equal(times, [0.1, 0.2])
    assert body_q.shape == (2, 3, 7) and body_qd.shape == (2, 3, 6)


def test_full_state_loader_refuses_episode_only_data_instead_of_interpolating(tmp_path):
    (tmp_path / "artifacts").mkdir()
    with pytest.raises(ValueError, match="full-state artifact"):
        load_full_states(tmp_path)


def test_policy_timebase_is_causal_and_resamples_60hz_actions_at_12hz():
    config = PolicyTimeConfig(observation_horizon=2, action_horizon=2)
    state_t = np.arange(0, 31) / 60
    ticks = policy_action_ticks(state_t, config)
    actions = np.arange(30 * 9).reshape(30, 9)
    sampled, indices = resample_joint_actions(actions, state_t, ticks[:3])
    np.testing.assert_array_equal(indices, [0, 5, 10])
    np.testing.assert_array_equal(sampled, actions[indices])
    rgb_t = np.array([0, 1 / 12, 2 / 12])
    query = np.array([1 / 60, 6 / 60])
    selected = causal_indices(rgb_t, query)
    assert np.all(rgb_t[selected] <= query)


def test_action_chunks_are_consumed_once_and_replanning_replaces_tail():
    executor = ActionChunkExecutor()
    executor.replan(np.array([[1], [2], [3]]))
    np.testing.assert_array_equal(executor.pop(), [1])
    executor.replan(np.array([[8], [9]]))
    np.testing.assert_array_equal(executor.pop(), [8])
    np.testing.assert_array_equal(executor.pop(), [9])
    with pytest.raises(RuntimeError, match="exhausted"):
        executor.pop()


def test_language_has_no_outcome_and_allen_qa_fails_closed_without_expert_evidence():
    instruction = task_instruction("tighten_screw_with_allen_key_from_table")
    assert "tighten" in instruction.lower() and "success" not in instruction.lower()
    spec = {"kind": "allen_key", "mode": "tighten", "screw_q_index": 9,
            "screw_z_index": 10, "start_deg": -615, "turn_min_deg": 600,
            "loosen_min_deg": 180, "lead_mm_per_rad": 2.34,
            "open_finger_dof_m": 0.012, "closed_finger_dof_m": 0.003,
            "link7_body_index": 0, "T_link7_tcp": np.eye(4).tolist(),
            "retreat_min_m": 0.05, "robot_stop_speed_m_s": 0.005,
            "robot_stop_duration_s": 0.4}
    state_q = np.zeros(11); state_q[7:9] = 0.012
    rig = SimpleNamespace(
        policy_qa_spec=spec,
        model=SimpleNamespace(body_label=["fr3/fr3_link7"]),
        s0=SimpleNamespace(joint_q=Array(state_q),
                           body_q=Array([[0, 0, 0, 0, 0, 0, 1]])))
    report = AllenKeyPolicyEvaluator()(None, rig, 2.0)
    assert report["strict_success"] is None and report["success"] is None
    assert report["physical_success"] is False
    assert report["unsupported_checks"]


def test_allen_physical_success_requires_release_retreat_and_stopped_window():
    spec = {"kind": "allen_key", "mode": "tighten", "screw_q_index": 9,
            "screw_z_index": 10, "start_deg": -615, "turn_min_deg": 600,
            "loosen_min_deg": 180, "lead_mm_per_rad": 2.34,
            "open_finger_dof_m": 0.012, "closed_finger_dof_m": 0.003,
            "link7_body_index": 0, "T_link7_tcp": np.eye(4).tolist(),
            "retreat_min_m": 0.05, "robot_stop_speed_m_s": 0.005,
            "robot_stop_duration_s": 0.4}
    q = np.zeros(11); q[7:9] = 0.003
    state = SimpleNamespace(joint_q=Array(q),
                            body_q=Array([[0, 0, 0, 0, 0, 0, 1]]))
    rig = SimpleNamespace(policy_qa_spec=spec, s0=state,
                          model=SimpleNamespace(body_label=["fr3/fr3_link7"]))
    evaluator = AllenKeyPolicyEvaluator()
    assert not evaluator(None, rig, 0.0)["physical_success"]
    q[7:9] = 0.012; state.joint_q.assign(q)
    evaluator(None, rig, 0.1)
    state.body_q.assign([[0.06, 0, 0, 0, 0, 0, 1]])
    evaluator(None, rig, 0.2)
    report = None
    for timestamp in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        report = evaluator(None, rig, timestamp)
    assert report["physical_success"] is True
    assert report["strict_success"] is None

