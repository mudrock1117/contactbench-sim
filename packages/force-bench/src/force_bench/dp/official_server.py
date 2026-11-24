from __future__ import annotations
import argparse, json, socket
import numpy as np
import torch
from .official import validate_checkpoint_contract, workspace_from_yaml
from .transport import receive_arrays, send_arrays

def main(argv=None):
    p=argparse.ArgumentParser(description="Pinned official DP inference server")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765)
    p.add_argument("--device",default="cuda:0"); a=p.parse_args(argv)
    ws=workspace_from_yaml(a.config)
    payload=ws.load_checkpoint(path=a.checkpoint,map_location=a.device)
    validate_checkpoint_contract(ws.cfg,payload)
    policy=(getattr(ws,"ema_model",None) or ws.model).to(a.device).eval()
    obs_steps=int(ws.cfg.n_obs_steps); action_steps=int(ws.cfg.n_action_steps)
    image_shape=tuple(ws.cfg.shape_meta.obs.wrist_rgb.shape)
    agent_shape=tuple(ws.cfg.shape_meta.obs.agent_pos.shape)
    action_dim=int(ws.cfg.shape_meta.action.shape[0])
    print(json.dumps({"event":"ready","implementation":"real-stanford/diffusion_policy@5ba07ac",
                      "host":a.host,"port":a.port,"n_action_steps":action_steps}),flush=True)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((a.host,a.port)); listener.listen(1)
        while True:
            conn,_=listener.accept()
            with conn:
                while True:
                    try: request=receive_arrays(conn)
                    except ConnectionError: break
                    try:
                        raw_image=np.asarray(request["rgb_wrist"])
                        raw_agent=np.asarray(request["proprio"])
                        if raw_image.shape!=(obs_steps,*image_shape):
                            raise ValueError(f"wrist RGB {raw_image.shape} != {(obs_steps,*image_shape)}")
                        if raw_agent.shape!=(obs_steps,*agent_shape):
                            raise ValueError(f"agent_pos {raw_agent.shape} != {(obs_steps,*agent_shape)}")
                        image=torch.as_tensor(raw_image,device=a.device).float().unsqueeze(0)/255.0
                        agent=torch.as_tensor(raw_agent,device=a.device).float().unsqueeze(0)
                        with torch.inference_mode():
                            result=policy.predict_action({"wrist_rgb":image,"agent_pos":agent})
                        action=result["action"][0].cpu().numpy()
                        if action.shape!=(action_steps,action_dim) or not np.isfinite(action).all():
                            raise RuntimeError(f"invalid official action output {action.shape}")
                        send_arrays(conn,{"actions":action})
                    except Exception as error:
                        send_arrays(conn,{"error":np.frombuffer(
                            f"{type(error).__name__}: {error}".encode(),dtype=np.uint8)})
if __name__=="__main__": raise SystemExit(main())
