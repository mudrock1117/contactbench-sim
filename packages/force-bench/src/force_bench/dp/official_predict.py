from __future__ import annotations
import argparse, json, numpy as np, torch
from .official import workspace_from_yaml
from .official_dataset import ContactBenchOfficialDPDataset
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",default="outputs/datasets/card_multimodal"); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(argv); ws=workspace_from_yaml(a.config); ws.load_checkpoint(path=a.checkpoint)
    policy=(getattr(ws,"ema_model",None) or ws.model).to(a.device).eval()
    item=ContactBenchOfficialDPDataset(a.data)[0]
    obs={k:v.unsqueeze(0).to(a.device) for k,v in item["obs"].items()}
    with torch.inference_mode(): result=policy.predict_action(obs)["action"][0].cpu().numpy()
    expert=item["action"][1:1+len(result)].numpy()
    print(json.dumps({"shape":list(result.shape),"finite":bool(np.isfinite(result).all()),
      "min":float(result.min()),"max":float(result.max()),"mae":float(np.abs(result-expert).mean())}))
if __name__=="__main__": raise SystemExit(main())
