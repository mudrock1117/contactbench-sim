from __future__ import annotations
import argparse, json, socket
import numpy as np
import torch
from .official import workspace_from_yaml
from .transport import receive_arrays, send_arrays

def main(argv=None):
    p=argparse.ArgumentParser(description="Pinned official DP inference server")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765)
    p.add_argument("--device",default="cuda:0"); a=p.parse_args(argv)
    ws=workspace_from_yaml(a.config); ws.load_checkpoint(path=a.checkpoint)
    policy=getattr(ws,"ema_model",None) or ws.model
    policy.to(a.device).eval()
    print(json.dumps({"event":"ready","implementation":"real-stanford/diffusion_policy@5ba07ac",
                      "host":a.host,"port":a.port}),flush=True)
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
                        image=torch.as_tensor(request["rgb_wrist"],device=a.device).float()/255.0
                        agent=torch.as_tensor(request["proprio"],device=a.device).float()
                        if image.ndim==4: image=image.unsqueeze(0)
                        if agent.ndim==2: agent=agent.unsqueeze(0)
                        with torch.inference_mode():
                            result=policy.predict_action({"wrist_rgb":image,"agent_pos":agent})
                        # Official slicing: [To-1 : To-1+n_action_steps].
                        send_arrays(conn,{"actions":result["action"][0].cpu().numpy()})
                    except Exception as error:
                        send_arrays(conn,{"error":np.frombuffer(
                            f"{type(error).__name__}: {error}".encode(),dtype=np.uint8)})
if __name__=="__main__": raise SystemExit(main())
