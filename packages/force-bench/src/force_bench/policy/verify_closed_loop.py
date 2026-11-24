"""Minimal live policy-loop smoke check (intended to be run explicitly)."""
from __future__ import annotations

import argparse
import json
import numpy as np

from .env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.data.timebase import PolicyTimeConfig


def _summary(observation):
    cameras = observation.get("cameras", {})
    return {
        "timestamp": float(observation["timestamp"]),
        "proprio_shape": list(np.shape(observation["proprio"])),
        "history_length": len(observation.get("history", [observation])),
        "cameras": {name: {"rgb_shape": list(value["rgb"].shape),
                            "rgb_dtype": str(value["rgb"].dtype),
                            "rgb_timestamp": float(value["rgb_timestamp"]),
                            "rgb_age_s": float(value["rgb_age_s"]),
                            "valid": bool(value["rgb_valid"])}
                    for name, value in cameras.items()},
        "tactile": (_tactile_summary(observation)
                    if "tactile" in observation else None),
    }


def _tactile_summary(observation):
    pressure = np.asarray(observation["tactile"])
    return {"shape": list(pressure.shape), "dtype": str(pressure.dtype),
            "valid": np.asarray(observation["tactile_valid"], bool).tolist(),
            "timestamp": float(observation["tactile_timestamp"]),
            "age_s": float(observation["tactile_age_s"]),
            "pressure_min_pa": float(pressure.min()),
            "pressure_max_pa": float(pressure.max()),
            "nonzero_elements": int(np.count_nonzero(pressure)),
            "calibration": observation["tactile_calibration"]}


def _emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--steps", type=int, default=5, choices=range(5, 11))
    parser.add_argument("--tactile", action="store_true")
    args = parser.parse_args(argv)
    timing = PolicyTimeConfig(physics_hz=60, policy_hz=12,
                              observation_horizon=2, action_horizon=16)
    _emit("scene_build_start", scene=args.scene)
    if args.tactile:
        _emit("gelsight_initialization_start", requested=True)
    env = ClosedLoopPolicyEnv(
        args.scene, config=PolicyEnvConfig(
            control_hz=12, visual_mode="both", rgb_fps=12,
            rgb_width=320, rgb_height=240, observation_horizon=2,
            include_tactile=args.tactile, time_config=timing,
            max_steps=args.steps))
    executed, final_reason = 0, "not_started"
    try:
        _emit("reset_start")
        observation = env.reset()
        _emit("scene_build_end", rig_type=type(env.rig).__name__,
              build_s=float(getattr(env.scene, "build_s", 0.0)))
        if args.tactile:
            _emit("gelsight_initialization_end",
                  sensors=sorted(getattr(env.rig, "gel_sensors", {})))
        _emit("reset_end", **_summary(observation))
        for index in range(args.steps):
            # A hold command is still a real controller command and advances
            # the unchanged physics pipeline for one 12 Hz policy interval.
            action = observation["robot_joint_position"].copy()
            env.replan(np.repeat(action[None], timing.action_horizon, axis=0))
            observation, reward, terminated, truncated, info = env.step_next()
            executed = index + 1
            _emit("step", index=index, **_summary(observation),
                  command_timestamp=info["command_timestamp"],
                  effective_until=info["effective_until"],
                  action_success=info["action_success"],
                  terminated=terminated, truncated=truncated,
                  evaluation=info.get("evaluation"))
            if terminated or truncated:
                final_reason = (info.get("evaluation") or {}).get(
                    "termination_reason", "terminated" if terminated else "truncated")
                break
        else:
            final_reason = "requested_steps_completed"
    except Exception as error:
        final_reason = f"exception:{type(error).__name__}"
        raise
    finally:
        env.close()
        _emit("finished", executed_steps=executed, termination_reason=final_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
