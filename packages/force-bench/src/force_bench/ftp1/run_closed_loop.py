"""24 Hz FTP-1 simulation client with real optical tactile frames."""
import argparse
import json
import socket
import numpy as np
from force_bench.data.timebase import PolicyTimeConfig
from force_bench.dp.transport import receive_arrays, send_arrays
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.policy.evaluation import CardRetrievalPolicyEvaluator

class OpticalSensor:
    def __init__(self):
        from hydrogel.render import TaximRenderer, raster_to_image
        self.renderer = TaximRenderer()
        self.raster = raster_to_image
        self.latest = None
        self.timestamp = None

    def __call__(self, rig, t):
        frames = getattr(rig, "tactile_frames", {})
        if not all(len(frames.get(p, [])) for p in ("gel_left","gel_right")):
            return None
        if self.timestamp != t:
            self.latest = np.stack([self.renderer.render(self.raster(
                np.asarray(frames[p][-1].normal_displacement_m))).copy()
                for p in ("gel_left","gel_right")])
            if self.latest.shape != (2,320,240,3) or self.latest.dtype != np.uint8:
                raise ValueError("invalid Taxim RGB")
            self.timestamp = t
        return self.latest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="retrieve_card_from_deck")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--execution-horizon", type=int, default=8)
    p.add_argument("--max-fast-steps", type=int, default=24)
    a = p.parse_args()
    if not 1 <= a.execution_horizon <= 32:
        raise ValueError("invalid execution horizon")
    optical = OpticalSensor()
    env = ClosedLoopPolicyEnv(a.scene, evaluator=CardRetrievalPolicyEvaluator(),
        sensors={"optical":optical}, config=PolicyEnvConfig(
        control_hz=24, visual_mode="wrist", rgb_fps=12,
        observation_horizon=1, action_type="eef_relative_10d",
        max_steps=a.max_fast_steps, time_config=PolicyTimeConfig(
        physics_hz=60,policy_hz=24,observation_horizon=1,action_horizon=32)))
    executed = replans = 0
    try:
        with socket.create_connection((a.host,a.port)) as conn:
            obs = env.reset()
            if obs["optical"] is None:
                obs, _ = env.warmup_sensors(lambda rig,o: o["optical"] is not None)
            while not env.done and executed < a.max_fast_steps:
                if not obs["rgb_valid"] or obs["rgb_timestamp"] > obs["timestamp"] + 1e-9:
                    raise RuntimeError("invalid causal wrist RGB")
                if obs["optical"] is None or optical.timestamp != obs["timestamp"]:
                    raise RuntimeError("missing current optical tactile frame")
                send_arrays(conn, {"rgb_wrist":obs["rgb"].transpose(1,2,0),
                    "tactile_rgb":obs["optical"],
                    "width":np.asarray(obs["robot_joint_position"][7:9].sum())})
                response = receive_arrays(conn)
                if "error" in response:
                    raise RuntimeError(response["error"].tobytes().decode())
                actions = response["actions"]
                if actions.shape != (32,10) or not np.isfinite(actions).all():
                    raise ValueError("invalid FTP-1 actions")
                env.replan(actions)
                replans += 1
                for _ in range(min(a.execution_horizon,a.max_fast_steps-executed)):
                    obs,_,terminated,truncated,info = env.step_next()
                    executed += 1
                    print(json.dumps({"step":executed,"replan":replans,
                        "simulation_time_s":float(obs["timestamp"]),
                        "action_success":bool(info["action_success"]),
                        "terminated":bool(terminated),"truncated":bool(truncated),
                        "reason":info.get("reason")}),flush=True)
                    if terminated or truncated:
                        break
            print(json.dumps({"event":"finished","fast_steps":executed,
                "replans":replans,"evaluation":env.evaluate()}),flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()
