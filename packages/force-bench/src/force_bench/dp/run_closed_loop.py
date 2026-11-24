from __future__ import annotations

import argparse
import json
import numpy as np

from force_bench.data.timebase import PolicyTimeConfig
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.policy.evaluation import CardRetrievalPolicyEvaluator

from .transport import InferenceClient


def observation_arrays(observation, camera_names):
    history = observation.get("history", [observation])
    proprio = np.stack([np.asarray(item["proprio"], np.float32) for item in history])
    images = {}
    for name in camera_names:
        values = []
        for item in history:
            stream = item["cameras"][name] if "cameras" in item else item
            if not bool(stream["rgb_valid"]):
                raise RuntimeError(f"{name} RGB is invalid at t={item['timestamp']}")
            values.append(np.asarray(stream["rgb"], np.uint8))
        images[name] = np.stack(values)
    return proprio, images


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DP server predictions in ContactBench")
    parser.add_argument("--scene", default="retrieve_card_from_deck")
    parser.add_argument("--camera", choices=("fixed", "wrist", "both"), default="wrist")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--collect-dir")
    args = parser.parse_args(argv)
    if not 1 <= args.execution_horizon <= 16:
        parser.error("execution horizon must be in [1,16]")
    camera_names = ("fixed", "wrist") if args.camera == "both" else (args.camera,)
    timing = PolicyTimeConfig(physics_hz=60, policy_hz=12,
                              observation_horizon=2, action_horizon=16)
    env = ClosedLoopPolicyEnv(
        args.scene, evaluator=CardRetrievalPolicyEvaluator(), collect_dir=args.collect_dir,
        config=PolicyEnvConfig(control_hz=12, visual_mode=args.camera, rgb_fps=12,
                               rgb_width=320, rgb_height=240, observation_horizon=2,
                               action_type="joint_position", time_config=timing,
                               max_steps=args.max_steps))
    client = InferenceClient(args.host, args.port)
    executed = replans = 0
    try:
        observation = env.reset()
        while not env.done:
            proprio, images = observation_arrays(observation, camera_names)
            prediction = client.predict(proprio, images)
            if prediction.shape != (16, 9) or not np.isfinite(prediction).all():
                raise ValueError(f"invalid predicted action chunk {prediction.shape}")
            env.replan(prediction); replans += 1
            for _ in range(args.execution_horizon):
                observation, _, terminated, truncated, info = env.step_next()
                executed += 1
                print(json.dumps({"event": "step", "step": executed, "replan": replans,
                                  "timestamp": float(observation["timestamp"]),
                                  "action_success": info["action_success"],
                                  "terminated": terminated, "truncated": truncated}), flush=True)
                if terminated or truncated:
                    break
        print(json.dumps({"event": "finished", "steps": executed, "replans": replans,
                          "evaluation": env.evaluate()}), flush=True)
    finally:
        client.close(); env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
