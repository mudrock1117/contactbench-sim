"""Policy-rollout evaluation that is independent of expert phase state."""
from __future__ import annotations

import numpy as np

from force_bench.geometry import world_T


class AllenKeyPolicyEvaluator:
    """Conservative physical evaluator; never promotes its subset to paper QA."""

    def __init__(self):
        self.reset()

    def reset(self, scene=None, rig=None, timestamp=0.0):
        self.previous_t = None
        self.previous_arm_positions = None
        self.still_since = None
        self.had_closed_grasp = False
        self.release_tcp = None

    @staticmethod
    def _tcp_world(rig, spec):
        pose = rig.s0.body_q.numpy()[int(spec["link7_body_index"])]
        return (world_T(pose) @ np.asarray(spec["T_link7_tcp"], float))[:3, 3]

    def __call__(self, scene, rig, timestamp):
        spec = getattr(rig, "policy_qa_spec", None)
        if not spec or spec.get("kind") != "allen_key":
            return {"status": "unvalidated", "success": None,
                    "strict_success": None, "physical_success": None,
                    "source": "unavailable", "unsupported_checks": [],
                    "reason": "scene has no Allen-key state QA definition"}

        state_q = rig.s0.joint_q.numpy()
        angle_deg = float(np.degrees(state_q[int(spec["screw_q_index"])]))
        rise_mm = float(state_q[int(spec["screw_z_index"])] / 0.001)
        mode, start_deg = spec["mode"], float(spec["start_deg"])
        checks = {}
        if mode == "tighten":
            turned = angle_deg - start_deg
            checks.update(
                angle_at_seat={"pass": angle_deg >= -1.0, "value_deg": angle_deg,
                               "threshold_deg": -1.0},
                head_seated={"pass": abs(rise_mm) < 0.2, "value_mm": rise_mm,
                             "max_abs_mm": 0.2},
                minimum_turn={"pass": turned >= spec["turn_min_deg"],
                              "value_deg": turned,
                              "threshold_deg": spec["turn_min_deg"]})
            unsupported = ["seating torque >= 0.8 * TAU_TIGHT",
                           "stationary under two post-retreat probe torques"]
        else:
            backed = start_deg - angle_deg
            expected = 0.8 * spec["lead_mm_per_rad"] * np.radians(backed)
            checks.update(
                minimum_backoff={"pass": backed >= spec["loosen_min_deg"],
                                 "value_deg": backed,
                                 "threshold_deg": spec["loosen_min_deg"]},
                thread_rise={"pass": rise_mm >= expected, "value_mm": rise_mm,
                             "minimum_expected_mm": expected})
            unsupported = ["preload release and peak drive torque > initial preload",
                           "free rotation under two post-retreat probe torques"]

        fingers = state_q[7:9]
        closed = bool(np.max(fingers) <= float(spec["closed_finger_dof_m"]) + 0.001)
        self.had_closed_grasp |= closed
        released = bool(self.had_closed_grasp and
                        np.min(fingers) >= float(spec["open_finger_dof_m"]) - 0.001)
        tcp = self._tcp_world(rig, spec)
        if released and self.release_tcp is None:
            self.release_tcp = tcp.copy()
        retreat = (0.0 if self.release_tcp is None
                   else float(np.linalg.norm(tcp - self.release_tcp)))
        checks["tool_released"] = {"pass": released, "finger_dof_m": fingers.tolist(),
                                    "open_target_m": spec["open_finger_dof_m"]}
        checks["robot_retreat"] = {"pass": retreat >= spec["retreat_min_m"],
                                    "value_m": retreat,
                                    "threshold_m": spec["retreat_min_m"]}

        labels = [str(label) for label in rig.model.body_label]
        arm_indices = [i for i, label in enumerate(labels) if label.startswith("fr3/")]
        arm_positions = rig.s0.body_q.numpy()[arm_indices, :3]
        speed = np.inf
        if self.previous_t is not None and timestamp > self.previous_t:
            speed = float(np.linalg.norm(arm_positions - self.previous_arm_positions,
                                         axis=1).max() / (timestamp - self.previous_t))
        if speed < spec["robot_stop_speed_m_s"]:
            self.still_since = timestamp if self.still_since is None else self.still_since
        else:
            self.still_since = None
        still_s = 0.0 if self.still_since is None else float(timestamp - self.still_since)
        checks["robot_stopped"] = {
            "pass": still_s >= spec["robot_stop_duration_s"],
            "max_link_speed_m_s": None if not np.isfinite(speed) else speed,
            "threshold_m_s": spec["robot_stop_speed_m_s"],
            "stationary_duration_s": still_s,
            "required_duration_s": spec["robot_stop_duration_s"]}
        self.previous_t = float(timestamp)
        self.previous_arm_positions = arm_positions.copy()

        unsupported += ["key/screw and robot contact histories after extraction",
                        "key returned to table with support, pose and rest checks",
                        "tool-in-hand, no-cam-out and direct-contact rollout histories"]
        physical = bool(all(value["pass"] for value in checks.values()))
        return {"status": "physical_pass" if physical else "physical_fail",
                "success": None, "strict_success": None,
                "physical_success": physical,
                "source": "independent_physical_state",
                "task_mode": mode, "timestamp": float(timestamp), "checks": checks,
                "unsupported_checks": unsupported,
                "reason": ("physical state subset passed; this is not appendix success"
                           if physical else "one or more independent physical checks failed")}


_compat_evaluator = AllenKeyPolicyEvaluator()


def allen_key_state_evaluation(scene, rig, timestamp):
    """Backward-compatible function wrapper; new environments own an instance."""
    return _compat_evaluator(scene, rig, timestamp)
