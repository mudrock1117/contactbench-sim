"""FR3 policy action adapters built on the repository's existing FrankaIK."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from force_bench.geometry import world_T


def _rotvec_matrix(vector):
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _quat_of_matrix(rotation):
    # Stable branch implementation, output xyzw as used by Newton.
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([(matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 0.0)) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (matrix[j, i] + matrix[i, j]) / s
        q[k] = (matrix[k, i] + matrix[i, k]) / s
        q[3] = (matrix[k, j] - matrix[j, k]) / s
    return q / np.linalg.norm(q)


def compose_relative_pose(current, translation, rotation_vector, *, translation_frame="world",
                          rotation_frame="tool", base_rotation=None):
    """Compose a 6D delta with a world-from-TCP transform.

    Translation is metres. Rotation is an axis-angle vector in radians.
    Frames are ``world``, ``base`` or ``tool``. A base rotation (world from
    base) is required when either delta is expressed in the base frame.
    """
    current = np.asarray(current, dtype=float)
    if current.shape != (4, 4):
        raise ValueError("current pose must be a 4x4 transform")
    translation, rotation_vector = np.asarray(translation, float), np.asarray(rotation_vector, float)
    if translation.shape != (3,) or rotation_vector.shape != (3,):
        raise ValueError("translation and rotation_vector must have shape [3]")
    R, target = current[:3, :3], current.copy()
    Rwb = None if base_rotation is None else np.asarray(base_rotation, float)
    frames = {"world": np.eye(3), "tool": R, "base": Rwb}
    if translation_frame not in frames or rotation_frame not in frames:
        raise ValueError("frames must be world, base, or tool")
    if frames[translation_frame] is None or frames[rotation_frame] is None:
        raise ValueError("base_rotation is required for base-frame actions")
    target[:3, 3] += frames[translation_frame] @ translation
    delta = _rotvec_matrix(rotation_vector)
    if rotation_frame == "tool":
        target[:3, :3] = R @ delta
    else:
        F = frames[rotation_frame]
        target[:3, :3] = F @ delta @ F.T @ R
    return target


@dataclass(frozen=True)
class EndEffectorActionConfig:
    translation_frame: str = "world"
    rotation_frame: str = "tool"
    max_translation_m: float = 0.02
    max_rotation_rad: float = 0.20
    workspace_low: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, 0.0]))
    workspace_high: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.5]))
    gripper_width_range_m: tuple[float, float] = (0.0, 0.08)
    ik_position_tolerance_m: float = 0.003
    ik_rotation_tolerance_rad: float = 0.05
    joint_margin_rad: float = 1e-4
    joint_velocity_limit_rad_s: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]))


@dataclass
class ActionResult:
    success: bool
    joint_target: np.ndarray | None
    target_tcp: np.ndarray | None = None
    clipped: bool = False
    reason: str | None = None


class JointPositionActionAdapter:
    """Preserve the recorded 9D joint-position-target control interface."""

    def __init__(self, rig):
        self.rig = rig

    def apply(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (9,) or not np.isfinite(action).all():
            return ActionResult(False, None, reason="joint action must be finite shape [9]")
        lower = np.asarray(self.rig.model.joint_limit_lower.numpy()[:9], float)
        upper = np.asarray(self.rig.model.joint_limit_upper.numpy()[:9], float)
        target = np.clip(action, lower, upper)
        self.rig.control.joint_target_q.assign(
            np.concatenate((target, self.rig.control.joint_target_q.numpy()[9:])))
        return ActionResult(True, target.copy(), clipped=not np.array_equal(target, action))


class EndEffectorActionAdapter:
    """Map a 7D relative TCP command to the existing 9D joint target.

    Input is ``[dx,dy,dz, rx,ry,rz, gripper_width]``. The first three are
    metres, the rotation vector is radians, and gripper width is the absolute
    distance between fingers in metres. It is deliberately not called a 10D
    paper action and performs no model-specific action conversion.
    """

    def __init__(self, rig, config=EndEffectorActionConfig(), *, base_transform=None):
        self.rig, self.config = rig, config
        labels = list(rig.model.body_label)
        self.link7 = labels.index("fr3/fr3_link7")
        T7 = world_T(rig.ik.body_pose("fr3/fr3_link7"))
        Ttcp = world_T(rig.ik.tcp_pose())
        self.link7_to_tcp = np.linalg.inv(T7) @ Ttcp
        if base_transform is None:
            base_transform = getattr(rig, "robot_base", None)
        if base_transform is None:
            self.base_transform = np.eye(4)
        else:
            values = np.asarray(base_transform, dtype=float)
            self.base_transform = world_T(values) if values.shape == (7,) else values
        if self.base_transform.shape != (4, 4):
            raise ValueError("base_transform must be a world-from-base 4x4 or xyzw pose")

    def actual_tcp_transform(self):
        return world_T(self.rig.s0.body_q.numpy()[self.link7]) @ self.link7_to_tcp

    def actual_tcp_pose(self):
        transform = self.actual_tcp_transform()
        return np.r_[transform[:3, 3], _quat_of_matrix(transform[:3, :3])]

    def apply(self, action, *, control_dt):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            return ActionResult(False, None, reason="end-effector action must be finite shape [7]")
        cfg = self.config
        translation = np.clip(action[:3], -cfg.max_translation_m, cfg.max_translation_m)
        rotation = action[3:6].copy()
        angle = np.linalg.norm(rotation)
        if angle > cfg.max_rotation_rad:
            rotation *= cfg.max_rotation_rad / angle
        width = float(np.clip(action[6], *cfg.gripper_width_range_m))
        clipped = not (np.array_equal(translation, action[:3]) and np.array_equal(rotation, action[3:6])
                       and width == action[6])
        current = self.actual_tcp_transform()
        target = compose_relative_pose(
            current, translation, rotation, translation_frame=cfg.translation_frame,
            rotation_frame=cfg.rotation_frame, base_rotation=self.base_transform[:3, :3])
        bounded = np.clip(target[:3, 3], cfg.workspace_low, cfg.workspace_high)
        clipped |= not np.array_equal(bounded, target[:3, 3])
        target[:3, 3] = bounded
        seed = self.rig.ik.joint_q.numpy().copy()
        q_solution = self.rig.ik.solve(target[:3, 3], _quat_of_matrix(target[:3, :3]), iterations=16)
        fk = world_T(self.rig.ik.tcp_pose())
        position_error = float(np.linalg.norm(fk[:3, 3] - target[:3, 3]))
        rotation_error = float(np.arccos(np.clip((np.trace(fk[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
        lower = np.asarray(self.rig.ik.model.joint_limit_lower.numpy()[:7], float) + cfg.joint_margin_rad
        upper = np.asarray(self.rig.ik.model.joint_limit_upper.numpy()[:7], float) - cfg.joint_margin_rad
        valid = (np.isfinite(q_solution[:7]).all() and position_error <= cfg.ik_position_tolerance_m
                 and rotation_error <= cfg.ik_rotation_tolerance_rad
                 and np.all(q_solution[:7] >= lower) and np.all(q_solution[:7] <= upper))
        if not valid:
            self.rig.ik.joint_q.assign(seed)
            return ActionResult(False, None, target_tcp=target, clipped=clipped,
                                reason=f"IK failed: position_error={position_error:.6g} m, "
                                       f"rotation_error={rotation_error:.6g} rad")
        previous = self.rig.control.joint_target_q.numpy().copy()
        max_step = np.asarray(cfg.joint_velocity_limit_rad_s) * float(control_dt)
        arm = np.clip(q_solution[:7], previous[:7] - max_step, previous[:7] + max_step)
        arm = np.clip(arm, lower, upper)
        target_q = previous.copy()
        target_q[:7] = arm
        target_q[7:9] = width / 2.0
        self.rig.control.joint_target_q.assign(target_q)
        clipped |= not np.allclose(arm, q_solution[:7], rtol=0.0, atol=0.0)
        return ActionResult(True, target_q[:9].copy(), target_tcp=target, clipped=clipped)

