"""Export saved FlowDCN Heun FID samples as individual PNGs."""

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / 'results' / 'fid' / 'fid_ode_heun'
BUDGETS = [2, 4, 8, 16, 21, 32, 64, 125]
IMAGES = 10000
SWEEP_PID = 2303005


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def export_budget(folder, expected_images=IMAGES):
    sampling = folder / 'sampling.json'
    archive = folder / 'images.npz'
    if not sampling.is_file() or not archive.is_file():
        raise FileNotFoundError(f'Completed sampling and NPZ required in {folder}')
    info = json.loads(sampling.read_text())
    if info['images'] != expected_images or info['solver'] != 'heun':
        raise RuntimeError(f'Unexpected sample settings in {sampling}')
    image_dir = folder / 'images'
    image_dir.mkdir(exist_ok=True)
    complete = folder / 'images_export_complete.json'
    if complete.is_file():
        saved = json.loads(complete.read_text())
        if saved.get('images') != expected_images:
            raise RuntimeError(f'Incompatible export marker in {folder}')
        if all((image_dir / f'{index:06d}.png').is_file()
               for index in range(expected_images)):
            print(f'PNG export already complete for {folder.name}', flush=True)
            return
    with np.load(archive) as data:
        pixels = data['arr_0']
    if pixels.shape != (expected_images, 256, 256, 3) or pixels.dtype != np.uint8:
        raise RuntimeError(f'Unexpected sample array in {archive}: {pixels.shape}, {pixels.dtype}')
    for index in range(expected_images):
        target = image_dir / f'{index:06d}.png'
        if not target.is_file():
            Image.fromarray(pixels[index]).save(target)
        if (index + 1) % 1000 == 0:
            print(f'{folder.name}: {index + 1}/{expected_images} PNGs', flush=True)
    save(complete, dict(images=expected_images, source=str(archive),
                        directory=str(image_dir)))
    print(f'Completed PNG export for {folder.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--wait-for-sweep', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    lock = (output / 'images_export.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_for_sweep:
        ticks = process_start_ticks(SWEEP_PID)
        while ticks is not None and process_start_ticks(SWEEP_PID) == ticks:
            print(f'Waiting for Heun sweep PID {SWEEP_PID}', flush=True)
            time.sleep(60)
        complete = output / 'complete.json'
        if not complete.is_file() or json.loads(complete.read_text()).get('status') != 'complete':
            raise RuntimeError('Heun sweep did not complete successfully; PNG export not started')
    for k in BUDGETS:
        export_budget(output / f'k_{k:04d}')
    save(output / 'images_export_complete.json', dict(status='complete', budgets=BUDGETS,
                                                       images_per_budget=IMAGES))


if __name__ == '__main__':
    main()
