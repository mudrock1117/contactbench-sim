"""Policy-rollout evaluation adapters that never depend on expert phases."""
from __future__ import annotations

import numpy as np


def allen_key_state_evaluation(scene, rig, timestamp):
    """Evaluate the state-observable subset of the published Allen-key QA.

    The original criterion also requires expert-only phase/torque/probe
    histories.  Consequently this adapter deliberately returns ``success=None``
    until an independent closed-loop probe protocol supplies those facts.
    """
    spec = getattr(rig, "policy_qa_spec", None)
    if not spec or spec.get("kind") != "allen_key":
        return {"status": "unvalidated", "success": None, "source": "unavailable",
                "reason": "scene has no Allen-key state QA definition"}
    q = rig.s0.joint_q.numpy()
    angle_deg = float(np.degrees(q[int(spec["screw_q_index"])]))
    rise_mm = float(q[int(spec["screw_z_index"])] / 0.001)
    mode, start_deg = spec["mode"], float(spec["start_deg"])
    checks = {}
    if mode == "tighten":
        checks["angle_at_seat"] = {"pass": angle_deg >= -1.0,
                                     "value_deg": angle_deg, "threshold_deg": -1.0}
        checks["head_seated"] = {"pass": abs(rise_mm) < 0.2,
                                  "value_mm": rise_mm, "max_abs_mm": 0.2}
        checks["minimum_turn"] = {"pass": angle_deg - start_deg >= spec["turn_min_deg"],
                                   "value_deg": angle_deg - start_deg,
                                   "threshold_deg": spec["turn_min_deg"]}
        unsupported = ["seating torque", "stationary under two post-retreat probe torques",
                       "key pickup/return and robot-contact checks"]
    else:
        backed = start_deg - angle_deg
        checks["minimum_backoff"] = {"pass": backed >= spec["loosen_min_deg"],
                                      "value_deg": backed,
                                      "threshold_deg": spec["loosen_min_deg"]}
        checks["thread_rise"] = {"pass": rise_mm >= 0.8 * spec["lead_mm_per_rad"] * np.radians(backed),
                                  "value_mm": rise_mm}
        unsupported = ["preload release and peak drive torque",
                       "free rotation under two post-retreat probe torques",
                       "key pickup/return and robot-contact checks"]
    return {"status": "partial", "success": None, "source": "physical_state_subset",
            "task_mode": mode, "timestamp": float(timestamp), "checks": checks,
            "unsupported_checks": unsupported,
            "reason": "full original QA requires evidence not recorded by the policy interface"}

