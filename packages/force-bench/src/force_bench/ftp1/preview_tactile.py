"""Write a real-image GelSight Mini left/right contact sheet for visual inspection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--episode",required=True); p.add_argument("--index",type=int,default=0); p.add_argument("--output",required=True); a=p.parse_args()
    import imageio.v2 as iio
    from .data import _read_rgb_stream, PAD_ORDER
    pairs=[]
    for pad in PAD_ORDER:
        _, images=_read_rgb_stream(a.episode,f"rgb/tactile/{pad}")
        if a.index<0 or a.index>=len(images): raise IndexError(a.index)
        pairs.append(images[a.index])
    iio.imwrite(a.output,np.concatenate(pairs,axis=1))
    print(f"wrote real Taxim RGB preview: {a.output}")
if __name__ == "__main__": main()
