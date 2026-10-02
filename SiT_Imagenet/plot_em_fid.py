#!/usr/bin/env python3
"""Plot completed EM FID results from the local sweep CSV."""
import csv
import math
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent / 'results/fid/fid_sweep_em_10k'


def main():
    with (ROOT / 'fid_vs_k.csv').open(newline='') as handle:
        points = sorted((int(r['k_em_updates']), float(r['fid'])) for r in csv.DictReader(handle))
    if not points or any(k <= 0 or not math.isfinite(f) or f < 0 for k, f in points):
        raise ValueError('Expected positive step counts and finite nonnegative FID results')
    width, height = 960, 560
    left, right, top, bottom = 90, 855, 48, 474
    lo, hi = math.log10(points[0][0]), math.log10(points[-1][0])
    pad = max(.02, (hi-lo)*.02)
    lo, hi = lo-pad, hi+pad
    ymax = max(1., max(f for _, f in points)*1.07)
    sx = lambda k: left+(math.log10(k)-lo)/(hi-lo)*(right-left)
    sy = lambda f: bottom-f/ymax*(bottom-top)
    blue = '#1565c0'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Unguided SiT ImageNet: Euler–Maruyama FID</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{blue}" stroke-width="1.5"/>',
             f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black" stroke-width="1.5"/>']
    for i in range(6):
        f = ymax*i/5
        y = sy(f)
        parts.extend([f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                      f'<text x="80" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="{blue}">{f:.1f}</text>'])
    for k, _ in points:
        x = sx(k)
        parts.extend([f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom+5}" stroke="black"/>',
                      f'<text x="{x:.2f}" y="496" text-anchor="middle" font-family="sans-serif" font-size="11">{k}</text>'])
    coords = ' '.join(f'{sx(k):.2f},{sy(f):.2f}' for k, f in points)
    parts.append(f'<polyline points="{coords}" fill="none" stroke="{blue}" stroke-width="3"/>')
    for k, f in points:
        parts.append(f'<circle cx="{sx(k):.2f}" cy="{sy(f):.2f}" r="4" fill="{blue}"/>')
    parts.extend(['<text x="480" y="536" text-anchor="middle" font-family="sans-serif" font-size="14">Euler–Maruyama step budget K</text>',
                  f'<text x="23" y="280" text-anchor="middle" transform="rotate(-90 23 280)" font-family="sans-serif" font-size="14" fill="{blue}">FID-10K</text>',
                  '</svg>'])
    svg = ROOT / 'fid_vs_k.svg'
    png = ROOT / 'fid_vs_k.png'
    svg.write_text('\n'.join(parts)+'\n')
    renderer = shutil.which('rsvg-convert')
    if not renderer:
        raise RuntimeError('rsvg-convert is required for the PNG')
    subprocess.run([renderer,'--width=2880','--height=1680',str(svg),'--output',str(png)],check=True)
    print(f'Plotted {len(points)} completed results: {svg} and {png}')


if __name__ == '__main__':
    main()
