from __future__ import annotations
import argparse, json, numpy as np, torch
from hydra.utils import instantiate
from .official import validate_checkpoint_contract, workspace_from_yaml

def main(argv=None):
    p=argparse.ArgumentParser(description="Offline prediction with the configured official dataset")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",default="cuda:0"); a=p.parse_args(argv)
    ws=workspace_from_yaml(a.config)
    payload=ws.load_checkpoint(path=a.checkpoint,map_location=a.device)
    validate_checkpoint_contract(ws.cfg,payload)
    policy=(getattr(ws,"ema_model",None) or ws.model).to(a.device).eval()
    dataset=instantiate(ws.cfg.task.dataset)
    item=dataset[0]
    obs={key:value.unsqueeze(0).to(a.device) for key,value in item["obs"].items()}
    with torch.inference_mode():
        result=policy.predict_action(obs)
    action=result["action"][0].cpu().numpy()
    expected=(int(ws.cfg.n_action_steps),int(ws.cfg.shape_meta.action.shape[0]))
    if action.shape!=expected:
        raise RuntimeError(f"official predict_action returned {action.shape}, expected {expected}")
    offset=int(ws.cfg.n_obs_steps)-1
    expert=item["action"][offset:offset+len(action)].numpy()
    print(json.dumps({"shape":list(action.shape),"action_pred_shape":list(result["action_pred"].shape[1:]),
      "finite":bool(np.isfinite(action).all()),"min":float(action.min()),"max":float(action.max()),
      "mae":float(np.abs(action-expert).mean()),"observation_offset":offset}))
    return 0
if __name__=="__main__": raise SystemExit(main())
