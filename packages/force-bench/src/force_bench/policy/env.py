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
from force_bench.robot.actions import (EndEffectorActionAdapter, EndEffectorActionConfig,
                                       JointPositionActionAdapter)


@dataclass(frozen=True)
class PolicyEnvConfig:
    control_hz: float = 12.0
    action_type: str = "joint_position"
    max_steps: int = 600
    terminate_on_action_failure: bool = True
    include_privileged_observation: bool = False

    def __post_init__(self):
        if self.control_hz <= 0 or self.max_steps < 1:
            raise ValueError("control_hz and max_steps must be positive")
        if self.action_type not in ("joint_position", "eef_delta"):
            raise ValueError("action_type must be joint_position or eef_delta")


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

    def reset(self):
        if self.collector is not None:
            self._finish_collection({"status": "unvalidated", "success": None,
                                     "source": "policy_reset", "reason": "environment reset"})
        self.scene = self.scene_factory()
        self.rig = self.scene.rig
        ratio = float(self.rig.fps) / self.config.control_hz
        self.frames_per_action = int(round(ratio))
        if self.frames_per_action < 1 or not np.isclose(ratio, self.frames_per_action):
            raise ValueError("control_hz must divide the Rig state rate exactly")
        self.control_dt = self.frames_per_action / float(self.rig.fps)
        self.observer = RigObservation(self.rig)
        self.tcp_adapter = None
        if hasattr(self.rig, "ik"):
            from dataclasses import replace
            eef_config = self.eef_action_config or replace(
                EndEffectorActionConfig(), workspace_low=np.asarray(self.scene.env_lo, float),
                workspace_high=np.asarray(self.scene.env_hi, float))
            self.tcp_adapter = EndEffectorActionAdapter(self.rig, eef_config)
        if self.config.action_type == "eef_delta" and self.tcp_adapter is None:
            raise ValueError("eef_delta requires a scene with FrankaIK")
        self.action_adapter = (JointPositionActionAdapter(self.rig)
                               if self.config.action_type == "joint_position" else self.tcp_adapter)
        self.t, self.steps, self.done, self.last_error = 0.0, 0, False, None
        if self.collect_dir is not None:
            self.collector = RigCollector(self.rig, self.collect_dir,
                                          task=self.scene_name or getattr(self.scene, "name", "policy"),
                                          seed=self.seed, source="closed_loop_policy")
            self.collector.writer.metadata["policy_environment"] = {
                "control_hz": self.config.control_hz,
                "action_type": self.config.action_type,
                "expert_replay": False,
                "expert_replay_limitation": "scripted task callbacks and privileged auxiliary controls are not recorded actions",
            }
        return self.observe()

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
        for name, sensor in self.sensors.items():
            result[name] = sensor(self.rig, self.t)
        return result

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
                    self._finish_collection({"status": "failed", "success": False,
                                             "source": "action_failure", "reason": result.reason})
                return observation, 0.0, self.done, False, info
            for _ in range(self.frames_per_action):
                if self.collector is not None:
                    self.collector.after_control()
                self._physics_frame()
                if self.collector is not None:
                    self.collector.after_step(self.t)
            self.steps += 1
            truncated = self.steps >= self.config.max_steps
            self.done = truncated
            observation = self.observe()
            info = {"action_success": True, "action_clipped": result.clipped,
                    "joint_target": result.joint_target,
                    "command_timestamp": command_time, "effective_until": self.t,
                    "truncated": truncated, "expert_replay_supported": False}
            if self.done:
                self._finish_collection({"status": "unvalidated", "success": None,
                                         "source": "policy_horizon",
                                         "reason": "task QA has not been evaluated"})
            return observation, 0.0, False, truncated, info
        except Exception as error:
            self.last_error, self.done = f"{type(error).__name__}: {error}", True
            if self.collector is not None:
                self._finish_collection({"status": "failed", "success": False,
                                         "source": "policy_exception", "reason": self.last_error})
            return self.observe(), 0.0, True, False, {
                "action_success": False, "reason": self.last_error,
                "command_timestamp": command_time, "effective_until": self.t,
                "truncated": False, "expert_replay_supported": False}

    def evaluate(self):
        """Run an explicitly supplied task evaluator; never invent success."""
        if self.evaluator is None:
            report = {"status": "unvalidated", "success": None,
                      "source": "unavailable",
                      "reason": "no task-specific QA adapter supplied; expert QA callbacks are not generic"}
        else:
            report = dict(self.evaluator(self.scene, self.rig, self.t))
            if not isinstance(report.get("success"), (bool, type(None))):
                raise ValueError("evaluator success must be bool or None")
        path = getattr(self, "episode_path", None)
        if path is not None:
            atomic_json(Path(path) / "policy_evaluation.json", report)
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

