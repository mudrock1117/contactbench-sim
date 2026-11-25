"""24 Hz ContactBench closed loop for the official pi0.5 socket policy."""
from __future__ import annotations

import argparse
import json
import socket
import numpy as np

from force_bench.data.timebase import PolicyTimeConfig
from force_bench.dp.transport import receive_arrays, send_arrays
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.policy.evaluation import CardRetrievalPolicyEvaluator
from force_bench.robot.eef_action import pose_xyzw_to_matrix, relative_eef_history


def _request(conn, observation):
    rgb = np.asarray(observation["rgb"], np.uint8)
    if rgb.shape != (3, 240, 320) or not bool(observation["rgb_valid"]):
        raise RuntimeError("pi0.5 requires one valid causal wrist RGB frame")
    if float(observation["rgb_timestamp"]) > float(observation["timestamp"]) + 1e-12:
        raise RuntimeError("future wrist RGB is not allowed")
    pose = pose_xyzw_to_matrix(observation["tcp_pose_world_xyzw"])
    width = float(observation["robot_joint_position"][7:9].sum())
    state = relative_eef_history(
        pose[None], np.asarray([width]), pose)[0].astype(np.float32)
    send_arrays(conn, {"rgb_wrist": rgb, "state": state})
    response = receive_arrays(conn)
    if "error" in response:
        raise RuntimeError(bytes(np.asarray(response["error"], np.uint8)).decode())
    actions = np.asarray(response["actions"], np.float32)
    if actions.shape != (32, 10) or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid pi0.5 actions {actions.shape}")
    return actions


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="retrieve_card_from_deck")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--execution-horizon", type=int, default=8)
    p.add_argument("--max-fast-steps", type=int, default=24)
    a = p.parse_args(argv)
    if not 1 <= a.execution_horizon <= 32:
        raise ValueError("execution_horizon must be in [1,32]")
    timing = PolicyTimeConfig(
        physics_hz=60, policy_hz=24,
        observation_horizon=1, action_horizon=32)
    env = ClosedLoopPolicyEnv(
        a.scene, evaluator=CardRetrievalPolicyEvaluator(),
        config=PolicyEnvConfig(
            control_hz=24, visual_mode="wrist", rgb_fps=12,
            rgb_width=320, rgb_height=240, observation_horizon=1,
            action_type="eef_relative_10d", include_tactile=False,
            time_config=timing, max_steps=a.max_fast_steps))
    executed = replans = 0
    conn = socket.create_connection((a.host, a.port))
    try:
        observation = env.reset()
        while not env.done and executed < a.max_fast_steps:
            actions = _request(conn, observation)
            env.replan(actions)
            replans += 1
            for _ in range(min(a.execution_horizon, a.max_fast_steps - executed)):
                observation, _, terminated, truncated, info = env.step_next()
                executed += 1
                print(json.dumps({
                    "event": "fast_step", "step": executed,
                    "replan": replans,
                    "simulation_time_s": float(observation["timestamp"]),
                    "rgb_timestamp": float(observation["rgb_timestamp"]),
                    "rgb_age_s": float(observation["rgb_age_s"]),
                    "action_success": bool(info["action_success"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }), flush=True)
                if terminated or truncated:
                    break
        print(json.dumps({
            "event": "finished", "fast_steps": executed,
            "replans": replans, "evaluation": env.evaluate(),
        }), flush=True)
    finally:
        conn.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
