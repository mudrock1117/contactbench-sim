"""CPU-only contracts for learning windows, action adapters, and policy env semantics."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest

from force_bench.data.dataset import (EpisodeWindowDataset, WindowSpec,
                                      policy_observation_indices, split_episode_paths)
from force_bench.data.episode import EpisodeWriter, atomic_json, load_episode
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.robot.actions import (EndEffectorActionAdapter, EndEffectorActionConfig,
                                       JointPositionActionAdapter, compose_relative_pose)


class Array:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.asarray(value, dtype=float).copy()


def _metadata(task="allen", seed=0):
    fields = [
        {"name": "robot_joint_position", "start": 0, "stop": 9,
         "units": ["rad"] * 7 + ["m"] * 2, "privileged": False},
        {"name": "robot_joint_velocity", "start": 9, "stop": 18,
         "units": ["rad/s"] * 7 + ["m/s"] * 2, "privileged": False},
        {"name": "body_pose/object", "start": 18, "stop": 25,
         "units": ["m"] * 3 + ["unitless"] * 4, "privileged": True},
    ]
    return {"task": task, "seed": seed, "action_dim": 9,
            "action_type": "joint_position_target", "observation_fields": fields}


def _rgb_stage(root, timestamps):
    stage = root / ".rgb-stage"
    images = stage / "images"
    images.mkdir(parents=True)
    for index in range(len(timestamps)):
        iio.imwrite(images / f"{index:06d}.png", np.full((3, 4, 3), index, np.uint8))
    np.save(stage / "timestamps.npy", np.asarray(timestamps, np.float64))
    camera = {"name": "fixed", "mount": "fixed_world", "frame": "world",
              "position": [1, 0, 1], "target": [0, 0, 0], "fov_deg": 40,
              "width": 4, "height": 3, "image_format": "png",
              "color_space": "RGB", "dtype": "uint8"}
    atomic_json(stage / "camera.json", camera)
    files = [stage / "timestamps.npy", stage / "camera.json", *sorted(images.glob("*.png"))]
    atomic_json(stage / "manifest.json", {"files_sha256": {
        str(path.relative_to(stage)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files}})
    return stage


def _episode(root, name="ep", task="allen", seed=0, transitions=10):
    metadata = _metadata(task, seed)
    metadata["rgb"] = {"camera_names": ["fixed"], "camera_type": "fixed_world"}
    writer = EpisodeWriter(root, metadata, np.arange(25), episode_id=name)
    for index in range(transitions):
        writer.append(np.full(9, index), np.arange(25) + index + 1,
                      timestamp=(index + 1) / 60)
    stage = _rgb_stage(root, [0, 5 / 60, 10 / 60])
    return writer.finish(attachments={"rgb/fixed": stage})


def test_dataset_aligns_12hz_rgb_to_60hz_state_without_privileged_pose(tmp_path):
    path = _episode(tmp_path)
    dataset = EpisodeWindowDataset([path], WindowSpec(observation_horizon=3, action_horizon=2))
    assert len(dataset) == 7
    sample = dataset[0]
    assert sample["proprio"].shape == (3, 18)
    assert sample["actions"].shape == (2, 9)
    assert sample["rgb"].shape == (3, 3, 3, 4)
    np.testing.assert_array_equal(sample["rgb_indices"], [0, 0, 0])
    np.testing.assert_allclose(sample["rgb_timestamps"], 0)
    np.testing.assert_allclose(sample["action_start_timestamps"], [2 / 60, 3 / 60])
    np.testing.assert_allclose(sample["action_end_timestamps"], [3 / 60, 4 / 60])
    assert dataset.schema["actions"]["shape"] == [2, 9]


def test_privileged_fields_are_fail_closed_and_only_explicitly_included():
    indices, _ = policy_observation_indices(_metadata())
    np.testing.assert_array_equal(indices, np.arange(18))
    all_indices, _ = policy_observation_indices(_metadata(), include_privileged=True)
    np.testing.assert_array_equal(all_indices, np.arange(25))
    with pytest.raises(ValueError, match="cannot be inferred"):
        policy_observation_indices({})


def test_dataset_rejects_rgb_checksum_failure(tmp_path):
    path = _episode(tmp_path)
    image = path / "rgb/fixed/images/000000.png"
    image.write_bytes(image.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        EpisodeWindowDataset([path])


def test_group_splits_never_leak_task_seed_or_episode(tmp_path):
    paths = [_episode(tmp_path, f"ep{i}", task="a" if i < 2 else "b", seed=i % 2)
             for i in range(4)]
    for level in ("task", "seed", "episode"):
        split = split_episode_paths(paths, level=level, fractions=(0.5, 0.25, 0.25), seed=4)
        owners = {}
        for part, members in split.items():
            for path in members:
                metadata = load_episode(path).metadata
                key = (metadata["task"] if level == "task" else
                       metadata["seed"] if level == "seed" else metadata["episode_id"])
                assert owners.setdefault(key, part) == part


def test_relative_pose_frames_and_rotation_units():
    current = np.eye(4)
    current[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    world = compose_relative_pose(current, [1, 0, 0], [np.pi / 2, 0, 0],
                                  translation_frame="world", rotation_frame="world")
    tool = compose_relative_pose(current, [1, 0, 0], [np.pi / 2, 0, 0],
                                 translation_frame="tool", rotation_frame="tool")
    np.testing.assert_allclose(world[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(tool[:3, 3], [0, 1, 0], atol=1e-12)
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(tool[:3, :3], current[:3, :3] @ Rx, atol=1e-12)
    np.testing.assert_allclose(world[:3, :3], Rx @ current[:3, :3], atol=1e-12)
    Rwb = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    base = compose_relative_pose(np.eye(4), [1, 0, 0], [0, 0, 0],
                                 translation_frame="base", base_rotation=Rwb)
    np.testing.assert_allclose(base[:3, 3], [0, 1, 0])


class FakeIK:
    def __init__(self, fail=False):
        self.fail = fail
        self.joint_q = Array(np.zeros((1, 9)))
        self.model = SimpleNamespace(joint_limit_lower=Array(np.full(9, -2.0)),
                                     joint_limit_upper=Array(np.full(9, 2.0)))
        self._tcp = np.array([0, 0, 0, 0, 0, 0, 1.0])

    def body_pose(self, label):
        return np.array([0, 0, 0, 0, 0, 0, 1.0])

    def solve(self, position, quaternion, iterations=16):
        if self.fail:
            return np.full(9, np.nan)
        self._tcp = np.r_[position, quaternion]
        answer = np.r_[np.full(7, 1.0), 0.0, 0.0]
        self.joint_q.assign(answer[None])
        return answer

    def tcp_pose(self):
        return self._tcp.copy()


def _action_rig(fail=False):
    body = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=float)
    return SimpleNamespace(
        model=SimpleNamespace(body_label=["fr3/fr3_link7"],
                              joint_limit_lower=Array(np.full(9, -2.0)),
                              joint_limit_upper=Array(np.full(9, 2.0))),
        ik=FakeIK(fail), s0=SimpleNamespace(body_q=Array(body)),
        control=SimpleNamespace(joint_target_q=Array(np.zeros(9))))


def test_eef_adapter_clips_actions_and_converts_absolute_gripper_width():
    rig = _action_rig()
    cfg = EndEffectorActionConfig(max_translation_m=0.01, max_rotation_rad=0.1,
                                  joint_velocity_limit_rad_s=np.ones(7))
    result = EndEffectorActionAdapter(rig, cfg).apply(
        [0.5, 0, 0, 0, 0, 1.0, 0.06], control_dt=0.02)
    assert result.success and result.clipped
    np.testing.assert_allclose(result.target_tcp[:3, 3], [0.01, 0, 0])
    np.testing.assert_allclose(result.joint_target[:7], 0.02)
    np.testing.assert_allclose(result.joint_target[7:9], 0.03)


def test_eef_ik_failure_does_not_mutate_robot_command():
    rig = _action_rig(fail=True)
    before = rig.control.joint_target_q.numpy()
    result = EndEffectorActionAdapter(rig).apply(np.zeros(7), control_dt=1 / 12)
    assert not result.success and "IK failed" in result.reason
    np.testing.assert_array_equal(rig.control.joint_target_q.numpy(), before)


def test_joint_action_interface_remains_9d_and_clips_limits():
    rig = _action_rig()
    result = JointPositionActionAdapter(rig).apply(np.full(9, 3.0))
    assert result.success and result.clipped
    np.testing.assert_array_equal(result.joint_target, np.full(9, 2.0))
    assert not JointPositionActionAdapter(rig).apply(np.zeros(7)).success


class FakeState:
    def __init__(self):
        self.joint_q, self.joint_qd = Array(np.zeros(9)), Array(np.zeros(9))
        self.body_q = Array(np.array([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]]))

    def clear_forces(self):
        pass


def _policy_scene():
    state0, state1 = FakeState(), FakeState()
    control = SimpleNamespace(joint_target_q=Array(np.zeros(9)),
                              joint_target_qd=Array(np.zeros(9)), joint_f=Array(np.zeros(9)))

    class Solver:
        def step(self, source, destination, ctl, contacts, dt):
            destination.joint_q.assign(ctl.joint_target_q.numpy())
            destination.joint_qd.assign(np.zeros(9))

        def update_contacts(self, contacts, state):
            pass

    rig = SimpleNamespace(
        name="fake", fps=60, substeps=1, s0=state0, s1=state1, control=control,
        model=SimpleNamespace(joint_label=[f"fr3/j{i}" for i in range(9)],
                              body_label=["fr3/fr3_link7", "object"],
                              joint_limit_lower=Array(np.full(9, -2)),
                              joint_limit_upper=Array(np.full(9, 2))),
        solver=Solver(), contacts=object(), pipeline=SimpleNamespace(collide=lambda *args: None),
        effective_collide_every_substep=lambda: True, substep_hook=None,
        solver_options={})
    return SimpleNamespace(name="fake", rig=rig)


def test_policy_env_timestamps_action_effect_and_truncation():
    env = ClosedLoopPolicyEnv(scene_factory=_policy_scene,
                              config=PolicyEnvConfig(control_hz=30, max_steps=1))
    initial = env.reset()
    assert initial["timestamp"] == 0 and initial["proprio"].shape == (18,)
    obs, reward, terminated, truncated, info = env.step(np.ones(9))
    assert reward == 0 and not terminated and truncated
    assert obs["timestamp"] == pytest.approx(2 / 60)
    assert info["command_timestamp"] == 0
    assert info["effective_until"] == pytest.approx(2 / 60)
    np.testing.assert_array_equal(obs["robot_joint_position"], np.ones(9))
    assert env.evaluate()["success"] is None


def test_policy_env_invalid_action_terminates_without_physics_step():
    env = ClosedLoopPolicyEnv(scene_factory=_policy_scene,
                              config=PolicyEnvConfig(control_hz=30))
    env.reset()
    obs, _, terminated, truncated, info = env.step(np.zeros(7))
    assert terminated and not truncated and obs["timestamp"] == 0
    assert not info["action_success"]


def test_policy_env_repeated_reset_clears_time_history_and_action_chunk():
    env = ClosedLoopPolicyEnv(scene_factory=_policy_scene,
                              config=PolicyEnvConfig(control_hz=30, observation_horizon=2))
    first = env.reset()
    env.replan(np.ones((2, 9)))
    env.step_next()
    second = env.reset()
    assert first["timestamp"] == second["timestamp"] == 0
    assert len(second["history"]) == 2
    assert env.chunk_executor.remaining == 0
