"""Shared CSV and dependency-free SVG reporting for the SiT Euler sweep."""
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _svg_text(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _plot_title(rows):
    """Return a correct experiment title from the recorded sampler metadata."""
    solver = str(rows[0].get("solver", "")).lower()
    if "rectified_flow" in solver:
        return "Rectified Flow CIFAR-10: FID-10K vs K (log scale)"
    return "Unguided SiT Euler: FID-10K vs K (log scale)"


def write_fid_plot(rows, path):
    width, height = 760, 480
    left, right, top, bottom = 85, 30, 35, 70
    ks = [int(row["k_euler_updates"]) for row in rows]
    fids = [float(row["fid"]) for row in rows]
    log_x_min, log_x_max = math.log10(min(ks)), math.log10(max(ks))
    y_min, y_max = min(fids), max(fids)
    y_pad = max(1.0, (y_max - y_min) * 0.08)
    x_pad = max(0.02, (log_x_max - log_x_min) * 0.05)
    log_x_min, log_x_max = log_x_min - x_pad, log_x_max + x_pad
    y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda x: left + (math.log10(x) - log_x_min) / (log_x_max - log_x_min) * plot_w
    sy = lambda y: top + (y_max - y) / (y_max - y_min) * plot_h
    point_string = " ".join(f"{sx(k):.2f},{sy(fid):.2f}" for k, fid in zip(ks, fids))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="18">{_svg_text(_plot_title(rows))}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
    ]
    for fraction in range(6):
        y = y_min + (y_max - y_min) * fraction / 5
        py = sy(y)
        parts += [
            f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#dddddd"/>',
            f'<text x="{left-8}" y="{py+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.2f}</text>',
        ]
    for k in ks:
        px = sx(k)
        parts.append(f'<text x="{px:.2f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{k}</text>')
    parts.append(f'<polyline points="{point_string}" fill="none" stroke="#1565c0" stroke-width="2.5"/>')
    for k, fid in zip(ks, fids):
        parts.append(f'<circle cx="{sx(k):.2f}" cy="{sy(fid):.2f}" r="4" fill="#1565c0"/>')
    parts += [
        f'<text x="{width / 2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">K (Euler updates, log scale)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="14">FID-10K (lower is better)</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts) + "\n")


def write_fid_plot_png(rows, path, svg_path=None):
    """Write a high-resolution PNG matching the SVG exactly when possible."""
    if svg_path is not None and shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "--width=2280", "--height=1440", str(svg_path), "--output", str(path)],
            check=True,
        )
        return

    # Portable fallback when an SVG renderer is unavailable.
    width, height = 1800, 1200
    left, right, top, bottom = 210, 70, 110, 165
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        label_font = ImageFont.truetype(font_path, 28)
        title_font = ImageFont.truetype(font_path, 38)
    except OSError:
        label_font = title_font = ImageFont.load_default()
    ks = [int(row["k_euler_updates"]) for row in rows]
    fids = [float(row["fid"]) for row in rows]
    log_x_min, log_x_max = math.log10(min(ks)), math.log10(max(ks))
    y_min, y_max = min(fids), max(fids)
    x_pad = max(0.02, (log_x_max - log_x_min) * 0.05)
    y_pad = max(1.0, (y_max - y_min) * 0.08)
    log_x_min, log_x_max = log_x_min - x_pad, log_x_max + x_pad
    y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda x: left + (math.log10(x) - log_x_min) / (log_x_max - log_x_min) * plot_w
    sy = lambda y: top + (y_max - y) / (y_max - y_min) * plot_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for fraction in range(6):
        y = y_min + (y_max - y_min) * fraction / 5
        py = sy(y)
        draw.line((left, py, width - right, py), fill="#dddddd")
        draw.text((left - 20, py), f"{y:.2f}", anchor="rm", font=label_font, fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=1)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=1)
    points = [(sx(k), sy(fid)) for k, fid in zip(ks, fids)]
    if len(points) > 1:
        draw.line(points, fill="#1565c0", width=3)
    for k, (px, py) in zip(ks, points):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill="#1565c0")
        draw.text((px, height - bottom + 28), str(k), anchor="ma", font=label_font, fill="black")
    draw.text((width / 2, 35), _plot_title(rows),
              anchor="ma", font=title_font, fill="black")
    draw.text((width / 2, height - 55), "K (Euler updates, log scale)",
              anchor="ma", font=label_font, fill="black")
    draw.text((28, 55), "FID-10K", font=label_font, fill="black")
    image.save(path)


def rebuild_summary(output_dir):
    output_dir = Path(output_dir)
    rows = []
    for metrics_path in output_dir.glob("k_*/metrics.json"):
        rows.append(json.loads(metrics_path.read_text()))
    if not rows:
        raise ValueError(f"No metrics.json files found in {output_dir}")
    rows.sort(key=lambda row: int(row["k_euler_updates"]))
    with (output_dir / "fid_vs_k.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    svg_path = output_dir / "fid_vs_k.svg"
    write_fid_plot(rows, svg_path)
    write_fid_plot_png(rows, output_dir / "fid_vs_k.png", svg_path)
    return rows
