from __future__ import annotations
import argparse
from .official import run_workspace
def main(argv=None):
    p=argparse.ArgumentParser(description="Run pinned official Diffusion Policy workspace")
    p.add_argument("--config",required=True); args=p.parse_args(argv)
    run_workspace(args.config); return 0
if __name__=="__main__": raise SystemExit(main())
