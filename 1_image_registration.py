#!/usr/bin/env python3
"""
Step 1: local interactive landmark selection for CODEX-Visium registration.

Writes points/<sample>_points.json files consumed by
3_online_parallel_overlay_transform.py.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.widgets import Button
from PIL import Image
from skimage import exposure

Image.MAX_IMAGE_PIXELS = None


def read_sample_table(path: Path):
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        has_header = header and header[0].lower() in {"sample", "sample_name"}
        if not has_header:
            f.seek(0)
            header = ["sample_name", "visium_path", "codex_reference_tif", "codex_channel"]
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            rows.append(row)
    return rows


def resolve_segmented_outputs(visium_path: str) -> Path:
    path = Path(visium_path).expanduser()
    if path.name == "segmented_outputs":
        return path
    if (path / "segmented_outputs").is_dir():
        return path / "segmented_outputs"
    return path


def get_visium_source(segmented_dir: Path) -> Path:
    for name in ("tissue_hires_image.png", "tissue_image.png", "tissue_lowres_image.png"):
        candidate = segmented_dir / "spatial" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No tissue image found under {segmented_dir}/spatial")


def get_visium_spacing(segmented_dir: Path, image_path: Path) -> float:
    scalefactors_path = segmented_dir / "spatial" / "scalefactors_json.json"
    if not scalefactors_path.exists():
        return 0.5
    with scalefactors_path.open() as f:
        sf = json.load(f)
    fullres = float(sf.get("microns_per_pixel", 0.5))
    key = "tissue_hires_scalef" if "hires" in image_path.name else "tissue_lowres_scalef"
    scale = float(sf.get(key, 1.0))
    return fullres / scale if scale else fullres


def get_codex_spacing(tif_path: Path) -> float:
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        desc = None
        if "ImageDescription" in page.tags:
            desc = page.tags["ImageDescription"].value
            if isinstance(desc, bytes):
                desc = desc.decode("utf-8", errors="ignore")
        if desc:
            match = re.search(r'PhysicalSizeX="([0-9.]+)"', desc)
            if match:
                return float(match.group(1))
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                pixels = root.find(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
                if pixels is not None and pixels.get("PhysicalSizeX"):
                    return float(pixels.get("PhysicalSizeX"))
            except ET.ParseError:
                pass
    return 0.50814825


def load_codex_channel(tif_path: Path, channel: int, max_size: int = 2000) -> np.ndarray:
    with tifffile.TiffFile(tif_path) as tif:
        shape = tif.series[0].shape
        if len(shape) == 2:
            arr = tif.asarray()
        elif len(shape) == 3 and shape[0] < 100:
            arr = tif.asarray(key=channel)
        else:
            full = tif.asarray()
            arr = full[:, :, channel] if full.shape[-1] > channel else full[channel]
    arr = arr.astype(np.float32)
    nonzero = arr[arr > 0]
    if nonzero.size:
        lo, hi = np.percentile(nonzero, [0.5, 99.9])
        arr = np.clip(arr, lo, hi)
    arr = exposure.rescale_intensity(arr, out_range=(0, 255)).astype(np.uint8)
    return downsample_for_display(arr, max_size=max_size)[0]


def downsample_for_display(arr: np.ndarray, max_size: int):
    scale = 1.0
    if max(arr.shape[:2]) > max_size:
        scale = max_size / max(arr.shape[:2])
        new_size = (int(arr.shape[1] * scale), int(arr.shape[0] * scale))
        arr = np.array(Image.fromarray(arr).resize(new_size, Image.LANCZOS))
    return arr, scale


class PointSelector:
    def __init__(self, source_path: Path, reference_path: Path, channel: int, existing=None):
        source_full = np.array(Image.open(source_path))
        source_display, self.source_scale = downsample_for_display(source_full, 2000)
        reference_display = load_codex_channel(reference_path, channel, max_size=2000)
        with tifffile.TiffFile(reference_path) as tif:
            ref_shape = tif.series[0].shape
        if len(ref_shape) == 3 and ref_shape[0] < 100:
            ref_hw = ref_shape[1:]
        elif len(ref_shape) == 3:
            ref_hw = ref_shape[:2]
        else:
            ref_hw = ref_shape
        self.reference_scale = reference_display.shape[0] / ref_hw[0]

        self.source_points = []
        self.reference_points = []
        if existing:
            self.source_points = (np.array(existing["source_points"]) * self.source_scale).tolist()
            self.reference_points = (np.array(existing["reference_points"]) * self.reference_scale).tolist()

        self.next_side = "source"
        self.done = False
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.axes[0].imshow(source_display)
        self.axes[1].imshow(reference_display, cmap="gray")
        self.axes[0].set_title("Visium H&E: click point")
        self.axes[1].set_title(f"CODEX channel {channel}")
        for ax in self.axes:
            ax.axis("off")

        done_ax = plt.axes([0.72, 0.02, 0.08, 0.04])
        undo_ax = plt.axes([0.81, 0.02, 0.08, 0.04])
        clear_ax = plt.axes([0.90, 0.02, 0.08, 0.04])
        self.done_button = Button(done_ax, "Done")
        self.undo_button = Button(undo_ax, "Undo")
        self.clear_button = Button(clear_ax, "Clear")
        self.done_button.on_clicked(self.on_done)
        self.undo_button.on_clicked(self.on_undo)
        self.clear_button.on_clicked(self.on_clear)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes == self.axes[0] and self.next_side == "source":
            self.source_points.append([event.xdata, event.ydata])
            self.next_side = "reference"
        elif event.inaxes == self.axes[1] and self.next_side == "reference":
            self.reference_points.append([event.xdata, event.ydata])
            self.next_side = "source"
        self.redraw()

    def on_undo(self, _event):
        if self.next_side == "reference" and self.source_points:
            self.source_points.pop()
            self.next_side = "source"
        elif self.reference_points:
            self.reference_points.pop()
            self.next_side = "reference"
        self.redraw()

    def on_clear(self, _event):
        self.source_points = []
        self.reference_points = []
        self.next_side = "source"
        self.redraw()

    def on_done(self, _event):
        if len(self.source_points) >= 3 and len(self.source_points) == len(self.reference_points):
            self.done = True
            plt.close(self.fig)
        else:
            print("Need at least 3 complete point pairs.")

    def redraw(self):
        for ax in self.axes:
            for artist in list(ax.texts):
                artist.remove()
            for coll in list(ax.collections):
                coll.remove()
        for ax, pts, color in (
            (self.axes[0], self.source_points, "red"),
            (self.axes[1], self.reference_points, "cyan"),
        ):
            if pts:
                arr = np.array(pts)
                ax.scatter(arr[:, 0], arr[:, 1], c=color, s=60, marker="x")
                for i, (x, y) in enumerate(arr, start=1):
                    ax.text(x, y - 18, str(i), color=color, fontsize=10, ha="center")
        self.axes[0].set_title("Visium H&E: click point" if self.next_side == "source" else "Visium H&E")
        self.axes[1].set_title("CODEX: click matching point" if self.next_side == "reference" else "CODEX")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
        if not self.done:
            return None, None
        source = np.array(self.source_points) / self.source_scale
        reference = np.array(self.reference_points) / self.reference_scale
        return source, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-table", required=True, type=Path)
    parser.add_argument("--points-dir", default=Path("points"), type=Path)
    parser.add_argument("--sample", action="append", help="Limit to one or more sample names")
    parser.add_argument("--channel", default=34, type=int)
    parser.add_argument("--reuse-points", action="store_true")
    args = parser.parse_args()

    args.points_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.sample or [])

    for row in read_sample_table(args.sample_table):
        sample = row["sample_name"]
        if selected and sample not in selected:
            continue
        channel = int(row.get("codex_channel") or args.channel)
        segmented_dir = resolve_segmented_outputs(row["visium_path"])
        source_image = get_visium_source(segmented_dir)
        reference_image = Path(row["codex_reference_tif"]).expanduser()
        points_path = args.points_dir / f"{sample}_points.json"

        existing = None
        if args.reuse_points and points_path.exists():
            with points_path.open() as f:
                existing = json.load(f)

        print(f"\nSample: {sample}")
        print(f"  Visium: {source_image}")
        print(f"  CODEX:  {reference_image}")
        print(f"  Channel: {channel}")

        selector = PointSelector(source_image, reference_image, channel, existing=existing)
        source_points, reference_points = selector.show()
        if source_points is None:
            print(f"Skipped {sample}")
            continue

        out = {
            "sample_name": sample,
            "source_points": source_points.tolist(),
            "reference_points": reference_points.tolist(),
            "source_spacing_um": get_visium_spacing(segmented_dir, source_image),
            "reference_spacing_um": get_codex_spacing(reference_image),
            "codex_channel": channel,
            "source_image": str(source_image),
            "reference_image": str(reference_image),
        }
        with points_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {points_path}")


if __name__ == "__main__":
    main()

