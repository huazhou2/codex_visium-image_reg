#!/usr/bin/env python3
"""
Step 3: online parallel overlay generation and coordinate transformation.

Transforms Visium HD H&E and cell segmentation polygons into CODEX-scaled
image space using landmark JSONs from step 1.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import tifffile
from PIL import Image
from shapely.affinity import affine_transform as shapely_affine
from shapely.geometry import mapping, shape
from skimage import exposure
from skimage.transform import AffineTransform, estimate_transform, warp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            rows.append(dict(zip(header, parts)))
    return rows


def resolve_segmented_outputs(visium_path: str) -> Path:
    path = Path(visium_path).expanduser()
    if path.name == "segmented_outputs":
        return path
    if (path / "segmented_outputs").is_dir():
        return path / "segmented_outputs"
    return path


def find_source_image(segmented_dir: Path) -> Path:
    for name in ("tissue_hires_image.png", "tissue_image.png", "tissue_lowres_image.png"):
        p = segmented_dir / "spatial" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No tissue image found in {segmented_dir}/spatial")


def load_scalefactors(segmented_dir: Path) -> dict:
    path = segmented_dir / "spatial" / "scalefactors_json.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def codex_shape(tif_path: Path):
    with tifffile.TiffFile(tif_path) as tif:
        shape = tif.series[0].shape
    if len(shape) == 3 and shape[0] < 100:
        return shape[1:]
    if len(shape) == 3:
        return shape[:2]
    return shape


def load_codex_channel(tif_path: Path, channel: int, max_overlay_dim: int = 4000):
    with tifffile.TiffFile(tif_path) as tif:
        shape = tif.series[0].shape
        if len(shape) == 2:
            arr = tif.asarray()
        elif len(shape) == 3 and shape[0] < 100:
            step = max(1, max(shape[1:]) // max_overlay_dim)
            arr = tif.asarray(key=channel)[::step, ::step]
        else:
            full = tif.asarray()
            if full.shape[-1] > channel:
                step = max(1, max(full.shape[:2]) // max_overlay_dim)
                arr = full[::step, ::step, channel]
            else:
                step = max(1, max(full.shape[1:]) // max_overlay_dim)
                arr = full[channel, ::step, ::step]
    arr = arr.astype(np.float32)
    nonzero = arr[arr > 0]
    if nonzero.size:
        lo, hi = np.percentile(nonzero, [0.5, 99.9])
        arr = np.clip(arr, lo, hi)
    return exposure.rescale_intensity(arr, out_range=(0, 255)).astype(np.uint8)


def transform_one(row: dict, points_dir: Path, out_dir: Path, max_dim: int):
    sample = row["sample_name"]
    segmented_dir = resolve_segmented_outputs(row["visium_path"])
    source_image = find_source_image(segmented_dir)
    reference_image = Path(row["codex_reference_tif"]).expanduser()
    points_path = points_dir / f"{sample}_points.json"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points JSON: {points_path}")

    with points_path.open() as f:
        points = json.load(f)

    src_pts = np.array(points["source_points"], dtype=float)
    dst_pts = np.array(points["reference_points"], dtype=float)
    channel = int(row.get("codex_channel") or points.get("codex_channel", 34))
    visium_spacing = float(points.get("source_spacing_um", 0.5))
    codex_spacing = float(points.get("reference_spacing_um", 0.50814825))

    sf = load_scalefactors(segmented_dir)
    tissue_hires_scalef = float(sf.get("tissue_hires_scalef", points.get("tissue_hires_scalef", 1.0)))

    tform = estimate_transform("similarity", src_pts, dst_pts)
    full_shape = codex_shape(reference_image)
    scale_factor = 1.0
    if max(full_shape) > max_dim:
        scale_factor = max_dim / max(full_shape)
    output_shape = (int(full_shape[0] * scale_factor), int(full_shape[1] * scale_factor))
    scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    tform_scaled = AffineTransform(matrix=scale_matrix @ tform.params)

    sample_out = out_dir / f"{sample}_registered"
    spatial_out = sample_out / "spatial"
    spatial_out.mkdir(parents=True, exist_ok=True)

    visium_hires = np.array(Image.open(source_image))
    warped_hires = warp(
        visium_hires,
        tform_scaled.inverse,
        output_shape=output_shape,
        preserve_range=True,
    ).astype(np.uint8)
    Image.fromarray(warped_hires).save(spatial_out / "tissue_hires_image.png")

    transformed_count = None
    original_count = None
    geojson_path = segmented_dir / "cell_segmentations.geojson"
    if geojson_path.exists():
        with geojson_path.open() as f:
            geo = json.load(f)
        original_count = len(geo.get("features", []))
        combined = tform_scaled.params @ np.array(
            [[tissue_hires_scalef, 0, 0], [0, tissue_hires_scalef, 0], [0, 0, 1]]
        )
        a, b, xoff = combined[0, :]
        d, e, yoff = combined[1, :]
        features = []
        for feature in geo.get("features", []):
            geom = shape(feature["geometry"])
            transformed_geom = shapely_affine(geom, [a, b, d, e, xoff, yoff])
            minx, miny, maxx, maxy = transformed_geom.bounds
            if minx >= -5 and miny >= -5 and maxx <= output_shape[1] + 5 and maxy <= output_shape[0] + 5:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(transformed_geom),
                        "properties": feature.get("properties", {}),
                    }
                )
        transformed_count = len(features)
        with (sample_out / "cell_segmentations.geojson").open("w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    h5_files = list(segmented_dir.glob("filtered_feature_*_matrix.h5")) + list(segmented_dir.glob("*matrix.h5"))
    if h5_files:
        h5_out = sample_out / "filtered_feature_cell_matrix.h5"
        shutil.copy2(h5_files[0], h5_out)
        with h5py.File(h5_out, "a") as h5:
            h5.attrs["transformed"] = True
            h5.attrs["transform_type"] = "similarity"
            h5.attrs["target_space"] = "CODEX"
            h5.attrs["scale_factor"] = scale_factor
            h5.attrs["tissue_hires_scalef"] = tissue_hires_scalef

    codex_norm = load_codex_channel(reference_image, channel)
    if warped_hires.shape[:2] != codex_norm.shape:
        warped_resized = np.array(
            Image.fromarray(warped_hires).resize((codex_norm.shape[1], codex_norm.shape[0]), Image.LANCZOS)
        )
    else:
        warped_resized = warped_hires
    warped_gray = warped_resized[:, :, 0] if warped_resized.ndim == 3 else warped_resized
    overlay = np.zeros((*codex_norm.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = codex_norm
    overlay[:, :, 1] = warped_gray
    overlay[:, :, 2] = codex_norm
    Image.fromarray(overlay).save(sample_out / "overlay.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    orig = np.array(Image.open(source_image))
    if max(orig.shape[:2]) > 4000:
        s = 4000 / max(orig.shape[:2])
        orig = np.array(Image.fromarray(orig).resize((int(orig.shape[1] * s), int(orig.shape[0] * s)), Image.LANCZOS))
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original Visium HD")
    axes[0, 1].imshow(codex_norm, cmap="gray")
    axes[0, 1].set_title(f"CODEX channel {channel}")
    axes[1, 0].imshow(warped_resized)
    axes[1, 0].set_title("Transformed Visium")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay: magenta=CODEX, green=Visium")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(sample_out / "overlay_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    warped_pts = tform(src_pts)
    errors = np.linalg.norm(warped_pts - dst_pts, axis=1)
    info = {
        "sample_name": sample,
        "transform_type": "similarity",
        "transform_matrix": tform.params.tolist(),
        "transform_matrix_scaled": tform_scaled.params.tolist(),
        "scale_factor": float(scale_factor),
        "tissue_hires_scalef": float(tissue_hires_scalef),
        "landmark_points": {
            "n_pairs": int(len(src_pts)),
            "source_points": src_pts.tolist(),
            "destination_points": (dst_pts * scale_factor).tolist(),
        },
        "physical_spacing": {
            "visium_um_per_pixel": visium_spacing,
            "codex_um_per_pixel": codex_spacing,
        },
        "image_dimensions": {
            "codex_full": [int(x) for x in full_shape],
            "output_shape": [int(x) for x in output_shape],
            "downsampled": output_shape != tuple(full_shape),
        },
        "registration_error": {
            "mean_pixels": float(errors.mean()),
            "std_pixels": float(errors.std()),
            "max_pixels": float(errors.max()),
            "mean_microns": float(errors.mean() * codex_spacing),
        },
        "codex_info": {"channel": channel},
        "coordinate_system": "CODEX_scaled" if scale_factor != 1.0 else "CODEX",
        "cell_retention": {
            "original": original_count,
            "transformed_in_bounds": transformed_count,
        },
        "source_files": {
            "visium_segmented_outputs": str(segmented_dir),
            "source_image": str(source_image),
            "reference_image": str(reference_image),
            "points_json": str(points_path),
        },
        "note": f"Cells scaled from fullres to hires (x{tissue_hires_scalef}), then transformed to CODEX",
    }
    with (spatial_out / "transformation_info.json").open("w") as f:
        json.dump(info, f, indent=2)

    readme = (
        f"# Registered Visium HD Data - {sample}\n\n"
        f"- Error: {errors.mean():.2f} +/- {errors.std():.2f} px\n"
        f"- Output shape: {output_shape}\n"
        f"- CODEX channel: {channel}\n"
        f"- Hires scale factor: {tissue_hires_scalef}\n"
    )
    (sample_out / "README.md").write_text(readme)
    return sample, str(sample_out), float(errors.mean()), transformed_count, original_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-table", required=True, type=Path)
    parser.add_argument("--points-dir", default=Path("points"), type=Path)
    parser.add_argument("--out-dir", default=Path("registered"), type=Path)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--max-dim", default=8000, type=int)
    parser.add_argument("--sample", action="append")
    args = parser.parse_args()

    rows = read_sample_table(args.sample_table)
    if args.sample:
        wanted = set(args.sample)
        rows = [r for r in rows if r["sample_name"] in wanted]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(rows)} samples with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transform_one, row, args.points_dir, args.out_dir, args.max_dim) for row in rows]
        for future in as_completed(futures):
            sample, path, mean_error, kept, original = future.result()
            print(f"Done {sample}: error={mean_error:.2f}px cells={kept}/{original} out={path}")


if __name__ == "__main__":
    main()

