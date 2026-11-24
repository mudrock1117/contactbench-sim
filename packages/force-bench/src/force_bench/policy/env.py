"""A small closed-loop environment over the existing Rig physics pipeline.

This module does not replay scripted experts.  Task solver wrappers remain
active, but task control callbacks that operate auxiliary mechanisms are not
recorded as actions and are deliberately not invoked by this policy API.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from force_bench.data.collection import RigCollector, RigObservation
from force_bench.data.episode import atomic_json
from force_bench.data.collection import read_tactile, tactile_metadata
from force_bench.data.rgb import LiveRGBCamera
from force_bench.data.task_language import task_instruction
from force_bench.data.timebase import ActionChunkExecutor, PolicyTimeConfig, PolicyFrameScheduler
from force_bench.robot.actions import (EndEffectorActionAdapter, EndEffectorActionConfig,
                                       JointPositionActionAdapter)
from force_bench.robot.eef_action import (RelativeEEF10DActionAdapter, pose_xyzw_to_matrix,
                                          relative_eef_history)


@dataclass(frozen=True)
class PolicyEnvConfig:
    control_hz: float = 12.0
    action_type: str = "joint_position"
    max_steps: int = 600
    terminate_on_action_failure: bool = True
    include_privileged_observation: bool = False
    visual_mode: str = "none"
    rgb_fps: float = 12.0
    rgb_width: int = 320
    rgb_height: int = 240
    observation_horizon: int = 1
    include_tactile: bool = False
    tactile_hz: float = 12.0
    tactile_size: int = 16
    time_config: PolicyTimeConfig | None = None

    def __post_init__(self):
        if self.control_hz <= 0 or self.max_steps < 1:
            raise ValueError("control_hz and max_steps must be positive")
        if self.action_type not in ("joint_position", "eef_delta", "eef_relative_10d"):
            raise ValueError("action_type must be joint_position, eef_delta, or eef_relative_10d")
        if self.visual_mode not in ("none", "fixed", "wrist", "both"):
            raise ValueError("visual_mode must be none, fixed, wrist, or both")
        if self.rgb_fps <= 0 or self.observation_horizon < 1:
            raise ValueError("RGB frequency and observation_horizon must be positive")
        if self.include_tactile and (self.tactile_hz <= 0 or self.tactile_size < 1):
            raise ValueError("tactile frequency and size must be positive")
        if self.time_config is not None and (
                not np.isclose(self.control_hz, self.time_config.policy_hz) or
                self.observation_horizon != self.time_config.observation_horizon):
            raise ValueError("online control_hz/history must match PolicyTimeConfig")


class ClosedLoopPolicyEnv:
    """Reset/observe/step/evaluate interface that drives real Rig controls.

    ``scene_factory`` returns a fresh ``teleop.scenes.Scene``-compatible
    object. By default the existing task scene capture is reused, preserving
    the task's solver, contacts, collision cadence, and mechanism wrappers.
    """

    def __init__(self, scene_name=None, *, scene_factory=None, config=PolicyEnvConfig(),
                 evaluator=None, sensors=None, collect_dir=None, seed=0, scene_args=(),
                 eef_action_config=None):
        if scene_factory is None:
            if scene_name is None:
                raise ValueError("scene_name or scene_factory is required")
            from force_bench.teleop.scenes import load_scene
            scene_factory = lambda: load_scene(scene_name, args=tuple(scene_args))
        self.scene_name, self.scene_factory, self.config = scene_name, scene_factory, config
        self.evaluator = evaluator
        if self.evaluator is None:
            from .evaluation import AllenKeyPolicyEvaluator
            self.evaluator = AllenKeyPolicyEvaluator()
        self.sensors = dict(sensors or {})
        self.collect_dir = None if collect_dir is None else Path(collect_dir)
        self.seed = int(seed)
        self.eef_action_config = eef_action_config
        self.scene = self.rig = self.action_adapter = self.observer = None
        self.collector = None
        self.t = 0.0
        self.steps = 0
        self.done = False
        self.last_error = None
        self.rgb_cameras, self._history = {}, []
        self.chunk_executor = ActionChunkExecutor()
        self._tactile_next = 0
        self._tactile_latest = None

    def reset(self):
        if self.collector is not None:
            self._finish_collection({"status": "unvalidated", "success": None,
                                     "source": "policy_reset", "reason": "environment reset"})
        self.scene = self.scene_factory()
        self.rig = self.scene.rig
        if self.config.time_config is not None and not np.isclose(
                self.rig.fps, self.config.time_config.physics_hz):
            raise ValueError("Rig state rate must match PolicyTimeConfig.physics_hz")
        timing = self.config.time_config or PolicyTimeConfig(
            physics_hz=float(self.rig.fps), policy_hz=self.config.control_hz,
            observation_horizon=self.config.observation_horizon, action_horizon=16)
        self.frame_scheduler = PolicyFrameScheduler(timing)
        self.control_dt = timing.policy_dt
        self.observer = RigObservation(self.rig)
        self.tcp_adapter = None
        if hasattr(self.rig, "ik"):
            from dataclasses import replace
            eef_config = self.eef_action_config or replace(
                EndEffectorActionConfig(), workspace_low=np.asarray(self.scene.env_lo, float),
                workspace_high=np.asarray(self.scene.env_hi, float))
            self.tcp_adapter = EndEffectorActionAdapter(self.rig, eef_config)
        if self.config.action_type in ("eef_delta", "eef_relative_10d") and self.tcp_adapter is None:
            raise ValueError("EEF actions require a scene with FrankaIK")
        if self.config.action_type == "joint_position":
            self.action_adapter = JointPositionActionAdapter(self.rig)
        elif self.config.action_type == "eef_delta":
            self.action_adapter = self.tcp_adapter
        else:
            self.action_adapter = RelativeEEF10DActionAdapter(self.rig, eef_config)
        self.t, self.steps, self.done, self.last_error = 0.0, 0, False, None
        reset_evaluator = getattr(self.evaluator, "reset", None)
        if reset_evaluator is not None:
            reset_evaluator(self.scene, self.rig, self.t)
        self.rgb_cameras, self._history = {}, []
        self.chunk_executor = ActionChunkExecutor()  # never cross an episode boundary
        names = (["fixed", "wrist"] if self.config.visual_mode == "both"
                 else ([] if self.config.visual_mode == "none" else [self.config.visual_mode]))
        if names:
            from force_bench.data.rgb import ensure_collection_cameras
            ensure_collection_cameras(self.rig, names)
        for name in names:
            definition = getattr(self.rig, f"collection_{name}_camera", None)
            if definition is None and name == "fixed":
                definition = getattr(self.rig, "collection_camera", None)
            if definition is None:
                raise ValueError(f"scene does not define the {name} policy camera")
            camera = LiveRGBCamera(self.rig, name, definition, fps=self.config.rgb_fps,
                                   width=self.config.rgb_width, height=self.config.rgb_height)
            camera.sample(self.rig.s0, 0.0, force=True)
            self.rgb_cameras[name] = camera
        self._tactile_next, self._tactile_latest = 0, None
        if self.config.include_tactile:
            self._sample_tactile(force=True)
        if self.collect_dir is not None:
            rgb_config = ({"enabled": True, "cameras": names, "fps": self.config.rgb_fps,
                           "width": self.config.rgb_width, "height": self.config.rgb_height}
                          if names else None)
            self.collector = RigCollector(self.rig, self.collect_dir,
                                          task=self.scene_name or getattr(self.scene, "name", "policy"),
                                          seed=self.seed, source="closed_loop_policy",
                                          tactile_size=self.config.tactile_size,
                                          rgb_config=rgb_config)
            self.collector.writer.metadata["policy_environment"] = {
                "control_hz": self.config.control_hz,
                "action_type": self.config.action_type,
                "expert_replay": False,
                "expert_replay_limitation": "scripted task callbacks and privileged auxiliary controls are not recorded actions",
            }
        observation = self.observe()
        self._history = [observation]
        return self._with_history(observation)

    def observe(self):
        if self.rig is None:
            raise RuntimeError("reset must be called first")
        full = self.observer.sample()
        policy = full if self.config.include_privileged_observation else full[:18]
        result = {"timestamp": np.float64(self.t), "proprio": policy.copy(),
                  "robot_joint_position": full[:9].copy(),
                  "robot_joint_velocity": full[9:18].copy()}
        if self.tcp_adapter is not None:
            result["tcp_pose_world_xyzw"] = self.tcp_adapter.actual_tcp_pose()
        if self.rgb_cameras:
            streams = {name: camera.observe(self.t) for name, camera in self.rgb_cameras.items()}
            if len(streams) == 1:
                result.update(next(iter(streams.values())))
            else:
                result["cameras"] = streams
                result.update({"rgb": {name: value["rgb"] for name, value in streams.items()},
                               "rgb_timestamp": {name: value["rgb_timestamp"] for name, value in streams.items()},
                               "rgb_valid": {name: value["rgb_valid"] for name, value in streams.items()},
                               "rgb_age_s": {name: value["rgb_age_s"] for name, value in streams.items()},
                               "camera_name": tuple(streams),
                               "calibration": {name: value["calibration"] for name, value in streams.items()}})
        if self.config.include_tactile and self._tactile_latest is not None:
            tactile = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                       for key, value in self._tactile_latest.items()}
            tactile["tactile_age_s"] = np.float64(self.t - tactile["tactile_timestamp"])
            result.update(tactile)
        result["language_instruction"] = task_instruction(
            self.scene_name or getattr(self.scene, "name", "policy"))
        for name, sensor in self.sensors.items():
            result[name] = sensor(self.rig, self.t)
        return result

    def _with_history(self, current):
        if self.config.observation_horizon == 1:
            return current
        history = self._history[-self.config.observation_horizon:]
        # At reset, left-pad with the initial causal observation only.
        history = [history[0]] * (self.config.observation_horizon - len(history)) + history
        result = dict(current)
        result["history"] = history
        if self.config.action_type == "eef_relative_10d":
            anchor = pose_xyzw_to_matrix(current["tcp_pose_world_xyzw"])
            poses = np.stack([pose_xyzw_to_matrix(item["tcp_pose_world_xyzw"]) for item in history])
            widths = np.asarray([item["robot_joint_position"][7:9].sum() for item in history])
            result["eef_relative_history"] = relative_eef_history(poses, widths, anchor)
        return result

    def _sample_tactile(self, *, force=False):
        if not force and self.t + 1e-12 < self._tactile_next / self.config.tactile_hz:
            return
        tactile, valid = read_tactile(self.rig, self.config.tactile_size)
        self._tactile_latest = {"tactile": tactile, "tactile_valid": valid,
                                "tactile_timestamp": np.float64(self.t),
                                "tactile_age_s": np.float64(0.0),
                                "tactile_calibration": tactile_metadata(self.config.tactile_size)}
        self._tactile_next += 1
        while self._tactile_next / self.config.tactile_hz <= self.t + 1e-12:
            self._tactile_next += 1

    def _physics_frame(self):
        frame_dt = 1.0 / float(self.rig.fps)
        sim_dt = frame_dt / self.rig.substeps
        every = self.rig.effective_collide_every_substep()
        if not every:
            self.rig.s0.clear_forces()
            self.rig.pipeline.collide(self.rig.s0, self.rig.contacts)
        for _ in range(self.rig.substeps):
            if every:
                self.rig.s0.clear_forces()
                self.rig.pipeline.collide(self.rig.s0, self.rig.contacts)
            self.rig.solver.step(self.rig.s0, self.rig.s1, self.rig.control,
                                 self.rig.contacts, sim_dt)
            self.rig.s0, self.rig.s1 = self.rig.s1, self.rig.s0
            hook = getattr(self.rig, "substep_hook", None)
            if hook is not None:
                hook()
        self.t += frame_dt
        self.rig.solver.update_contacts(self.rig.contacts, self.rig.s0)
        # GelSightRig.run normally performs this frame-end sampling.  The
        # policy environment owns its physics loop, so invoke the identical
        # sensor path explicitly after contact forces have been updated.
        sample_gels = getattr(self.rig, "_sample_gels", None)
        if sample_gels is not None and getattr(self.rig, "gel_sensors", {}):
            solver_forces = self.rig._gel_solver_forces(self.rig.s0)
            gel_forces = {}
            sample_gels(self.rig.s0, frame_dt, gel_forces, solver_forces)
            self.rig.last_forces.update(gel_forces)

    def step(self, action):
        if self.rig is None or self.done:
            raise RuntimeError("reset is required or the episode has ended")
        command_time = self.t
        try:
            result = (self.action_adapter.apply(action)
                      if self.config.action_type == "joint_position"
                      else self.action_adapter.apply(action, control_dt=self.control_dt))
            if not result.success:
                self.last_error = result.reason
                if self.config.terminate_on_action_failure:
                    self.done = True
                observation = self.observe()
                info = {"action_success": False, "reason": result.reason,
                        "command_timestamp": command_time, "effective_until": self.t,
                        "truncated": False, "expert_replay_supported": False}
                if self.done:
                    failure = {"status": "failed", "success": False,
                               "strict_success": False, "physical_success": False,
                               "termination_reason": "action_failure",
                               "source": "action_failure", "reason": result.reason}
                    info["evaluation"] = failure
                    self._finish_collection(failure)
                return observation, 0.0, self.done, False, info
            frames_this_action = self.frame_scheduler.pop()
            for _ in range(frames_this_action):
                if self.collector is not None:
                    self.collector.after_control()
                self._physics_frame()
                for camera in self.rgb_cameras.values():
                    camera.sample(self.rig.s0, self.t)
                if self.config.include_tactile:
                    self._sample_tactile()
                if self.collector is not None:
                    self.collector.after_step(self.t)
            self.steps += 1
            truncated = self.steps >= self.config.max_steps
            evaluation = self._run_evaluator()
            # A physical subset is useful diagnostics, but is never promoted
            # to the complete appendix/paper QA result.
            terminated = evaluation.get("strict_success") is True
            self.done = truncated or terminated
            if truncated and not terminated:
                evaluation = dict(evaluation)
                evaluation.update(status="timeout", source="policy_horizon",
                                  termination_reason="timeout",
                                  reason="policy step limit reached")
            observation = self.observe()
            self._history.append(observation)
            self._history = self._history[-self.config.observation_horizon:]
            observation = self._with_history(observation)
            info = {"action_success": True, "action_clipped": result.clipped,
                    "joint_target": result.joint_target,
                    "command_timestamp": command_time, "effective_until": self.t,
                    "truncated": truncated, "expert_replay_supported": False,
                    "evaluation": evaluation}
            if self.done:
                outcome = dict(evaluation)
                self._finish_collection(outcome)
                if getattr(self, "episode_path", None) is not None:
                    atomic_json(Path(self.episode_path) / "policy_evaluation.json", outcome)
            return observation, float(terminated), terminated, truncated and not terminated, info
        except Exception as error:
            self.last_error, self.done = f"{type(error).__name__}: {error}", True
            if self.collector is not None:
                self._finish_collection({"status": "failed", "success": False,
                                         "strict_success": False,
                                         "physical_success": False,
                                         "termination_reason": "simulation_exception",
                                         "source": "policy_exception", "reason": self.last_error})
            return self.observe(), 0.0, True, False, {
                "action_success": False, "reason": self.last_error,
                "command_timestamp": command_time, "effective_until": self.t,
                "truncated": False, "termination_reason": "simulation_exception",
                "expert_replay_supported": False}

    def replan(self, actions):
        """Replace any unexecuted policy chunk; actions are consumed once."""
        if self.config.time_config is not None and len(actions) > self.config.time_config.action_horizon:
            raise ValueError("predicted chunk exceeds PolicyTimeConfig.action_horizon")
        self.chunk_executor.replan(actions)
        if self.config.action_type == "eef_relative_10d":
            self.action_adapter.set_chunk_anchor(self.t)

    def step_next(self):
        """Execute exactly the next action from the current prediction chunk."""
        return self.step(self.chunk_executor.pop())

    def evaluate(self):
        """Run an explicitly supplied task evaluator; never invent success."""
        report = self._run_evaluator()
        path = getattr(self, "episode_path", None)
        if path is not None:
            atomic_json(Path(path) / "policy_evaluation.json", report)
        return report

    def _run_evaluator(self):
        report = dict(self.evaluator(self.scene, self.rig, self.t))
        for key in ("success", "strict_success", "physical_success"):
            if key in report and not isinstance(report[key], (bool, type(None))):
                raise ValueError(f"evaluator {key} must be bool or None")
        return report

    def _finish_collection(self, outcome):
        collector, self.collector = self.collector, None
        if collector is None:
            return None
        if not collector.writer.actions:
            collector.abort()
            return None
        self.episode_path = collector.finish(outcome=outcome, discard_pending=True)
        return self.episode_path

    def close(self):
        if self.collector is not None:
            self._finish_collection({"status": "unvalidated", "success": None,
                                     "source": "policy_close", "reason": "environment closed"})
