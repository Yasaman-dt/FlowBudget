"""Export already-running sampling stages after completion, without resampling."""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
    for folder in a.folders:
        while not (folder/'sampling.json').exists():time.sleep(15)
        with np.load(folder/'images.npz') as archive:
            pixels=archive['arr_0']
            out=folder/'images';out.mkdir(exist_ok=True)
            for i,pixel in enumerate(pixels):
                path=out/f'{i:06d}.png'
                if not path.exists():Image.fromarray(pixel).save(path)
        print(f'Exported {len(pixels)} images to {out}',flush=True)
