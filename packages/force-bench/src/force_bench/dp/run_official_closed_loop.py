from __future__ import annotations
import argparse, json
import numpy as np
from force_bench.data.timebase import PolicyTimeConfig
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.policy.evaluation import CardRetrievalPolicyEvaluator
from .transport import InferenceClient

def main(argv=None):
    p=argparse.ArgumentParser(description="Official 10D DP closed loop")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765)
    p.add_argument("--execution-horizon",type=int,default=8); p.add_argument("--max-steps",type=int,default=600)
    p.add_argument("--policy-hz",type=float,choices=(12.0,),default=12.0,
                   help="must match official B1 training config"); a=p.parse_args(argv)
    if a.execution_horizon != 8:
        p.error("official B1 n_action_steps is 8; execution-horizon must be 8")
    timing=PolicyTimeConfig(60,a.policy_hz,2,16)
    env=ClosedLoopPolicyEnv("retrieve_card_from_deck",evaluator=CardRetrievalPolicyEvaluator(),
        config=PolicyEnvConfig(control_hz=a.policy_hz,visual_mode="wrist",rgb_fps=a.policy_hz,
        rgb_width=320,rgb_height=240,observation_horizon=2,action_type="eef_relative_10d",
        time_config=timing,max_steps=a.max_steps))
    client=InferenceClient(a.host,a.port); steps=replans=0
    try:
        obs=env.reset()
        while not env.done:
            history=obs["history"]; images=np.stack([x["rgb"] for x in history])
            agent=np.asarray(obs["eef_relative_history"],dtype=np.float32)
            chunk=client.predict(agent,{"wrist":images})
            if chunk.shape!=(8,10) or not np.isfinite(chunk).all():
                raise ValueError(f"official predict_action must return [8,10], got {chunk.shape}")
            env.replan(chunk); replans+=1
            for _ in range(a.execution_horizon):
                obs,_,terminated,truncated,info=env.step_next(); steps+=1
                print(json.dumps({"step":steps,"replan":replans,"timestamp":float(obs["timestamp"]),
                                  "action_success":info["action_success"]}),flush=True)
                if terminated or truncated: break
        print(json.dumps({"steps":steps,"replans":replans,"evaluation":env.evaluate()}),flush=True)
    finally:
        client.close(); env.close()
if __name__=="__main__": raise SystemExit(main())
