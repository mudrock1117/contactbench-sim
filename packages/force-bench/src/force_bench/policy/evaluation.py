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


class CardRetrievalPolicyEvaluator:
    """Online-measurable subset of the original retrieve-card appendix QA.

    The original verdict also audits rollout contact histories.  Those checks
    are not inferred here, so ``strict_success`` and ``success`` deliberately
    remain ``None`` even when the physical state subset passes.
    """
    def reset(self, scene=None, rig=None, timestamp=0.0):
        self.initial = None
        self.release_link_position = None
        self.had_closed_grasp = False
        if rig is not None:
            labels = [str(value) for value in rig.model.body_label]
            required = ("card_picked", "card_a", "card_b", "card_c")
            if all(label in labels for label in required):
                body_q = rig.s0.body_q.numpy()
                self.initial = {label: body_q[labels.index(label)].copy()
                                for label in required}

    @staticmethod
    def _rotation(quaternion):
        x, y, z, w = quaternion
        return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                         [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                         [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

    def __call__(self, scene, rig, timestamp):
        labels = [str(value) for value in rig.model.body_label]
        required = ("card_picked", "card_a", "card_b", "card_c")
        if self.initial is None or not all(label in labels for label in required):
            return {"status": "unvalidated", "success": None, "strict_success": None,
                    "physical_success": None, "source": "unavailable",
                    "unsupported_checks": ["card bodies unavailable"],
                    "reason": "scene does not expose card retrieval state"}
        from force_bench.tasks import card_pick
        body_q = rig.s0.body_q.numpy()
        picked = body_q[labels.index("card_picked")]
        normal = self._rotation(picked[3:7])[:, 0]
        card_ok = bool(abs(float(normal[2])) > 0.95 and picked[2] < 0.004 and picked[0] > 0.0)
        deck = {}
        deck_ok = True
        for label in ("card_a", "card_b", "card_c"):
            pose, initial = body_q[labels.index(label)], self.initial[label]
            displacement = float(np.linalg.norm(pose[:3] - initial[:3]))
            n0 = self._rotation(initial[3:7])[:, 0]
            n1 = self._rotation(pose[3:7])[:, 0]
            tilt = float(np.degrees(np.arccos(np.clip(abs(n0 @ n1), -1, 1))))
            in_slot = bool(card_pick.SLOT_X[0] - 0.01 < pose[0] < card_pick.SLOT_X[1] + 0.01)
            passed = displacement < 0.005 and tilt < 3.0 and in_slot
            deck[label] = {"pass": bool(passed), "displacement_m": displacement,
                           "tilt_deg": tilt, "in_slot": in_slot}
            deck_ok &= passed
        fingers = rig.s0.joint_q.numpy()[7:9]
        self.had_closed_grasp |= bool(np.max(fingers) <= 0.025)
        released = bool(self.had_closed_grasp and np.min(fingers) >= 0.039)
        # Original OPEN_W=0.04 m, with the same 1 mm release tolerance used
        # by the independent Allen-key evaluator.
        link = body_q[labels.index("fr3/fr3_link7"), :3]
        if released and self.release_link_position is None:
            self.release_link_position = link.copy()
        retreat = (0.0 if self.release_link_position is None else
                   float(np.linalg.norm(link - self.release_link_position)))
        checks = {
            "single_card_on_table": {"pass": card_ok,
                                     "card_position_m": picked[:3].tolist()},
            "deck_remains_neat": {"pass": bool(deck_ok), "cards": deck,
                                  "max_displacement_m": 0.005, "max_tilt_deg": 3.0},
            "released": {"pass": released, "finger_dof_m": fingers.tolist()},
            "robot_retreat": {"pass": retreat >= 0.05, "value_m": retreat,
                              "threshold_m": 0.05}}
        physical = bool(all(value["pass"] for value in checks.values()))
        return {"status": "physical_pass" if physical else "physical_fail",
                "success": None, "strict_success": None, "physical_success": physical,
                "source": "independent_physical_state", "timestamp": float(timestamp),
                "checks": checks,
                "unsupported_checks": [
                    "post-release robot/object contact history",
                    "complete rollout no-cam-out and direct-contact integrity histories"],
                "reason": ("physical state subset passed; not original appendix QA"
                           if physical else "one or more physical checks failed")}


_compat_evaluator = AllenKeyPolicyEvaluator()


def allen_key_state_evaluation(scene, rig, timestamp):
    """Backward-compatible function wrapper; new environments own an instance."""
    return _compat_evaluator(scene, rig, timestamp)
