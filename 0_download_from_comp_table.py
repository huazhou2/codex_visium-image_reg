#!/usr/bin/env python3
"""
Step 0: table-driven download of all needed files from the cluster into ./data/.

Reads comp_table (5 tab-separated columns):
  1 sample_name
  2 visium spaceranger outs dir   (contains segmented_outputs/)   -> VISIUM
  3 codex reference analysis.tif                                   -> CODEX
  4 codex_channel
  5 codex object_Data.csv         (CODEX cells)                    -> CODEX

Produces the layout the pipeline expects:
  data/<sample>/segmented_outputs/{cell_segmentations.geojson,filtered_feature_cell_matrix.h5,spatial/*}
  data/<sample>_reference.tif
  data/<sample>_codex_cells.csv

Examples:
  python 0_download_from_comp_table.py
  python 0_download_from_comp_table.py --samples NYU658 NYU851
  python 0_download_from_comp_table.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

VISIUM_FILES = [
    ("segmented_outputs/cell_segmentations.geojson", "segmented_outputs", True),
    ("segmented_outputs/filtered_feature_cell_matrix.h5", "segmented_outputs", True),
    ("segmented_outputs/spatial/tissue_hires_image.png", "segmented_outputs/spatial", False),
    ("segmented_outputs/spatial/scalefactors_json.json", "segmented_outputs/spatial", False),
]


def read_comp_table(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        parts += [""] * (5 - len(parts))
        sample, visium_outs, codex_tif, channel, codex_csv = parts[:5]
        rows.append({
            "sample": sample,
            "visium_outs": visium_outs.rstrip("/"),
            "codex_tif": codex_tif,
            "channel": channel,
            "codex_csv": codex_csv,
        })
    return rows


def scp(host: str, remote: str, local: Path, optional: bool, dry_run: bool):
    cmd = ["scp", f"{host}:{remote}", str(local)]
    if dry_run:
        print("  +", " ".join(cmd))
        return
    res = subprocess.run(cmd)
    if res.returncode != 0:
        msg = f"  (skip missing: {remote})" if optional else f"  ERROR fetching {remote}"
        print(msg)
        if not optional:
            raise SystemExit(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-table", type=Path, default=Path("comp_table"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--host", default="zhouh05@bigpurple.nyumc.org")
    ap.add_argument("--samples", nargs="*", help="limit to these sample names")
    ap.add_argument("--sample-table-out", type=Path, default=Path("sample_table.data.tsv"),
                    help="local sample-table written from comp_table after download")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = set(args.samples or [])
    for row in read_comp_table(args.comp_table):
        s = row["sample"]
        if wanted and s not in wanted:
            continue
        print(f"==> {s}")
        sample_root = args.data_dir / s
        (sample_root / "segmented_outputs" / "spatial").mkdir(parents=True, exist_ok=True)

        # VISIUM (col 2)
        for rel, dest_sub, optional in VISIUM_FILES:
            remote = f"{row['visium_outs']}/{rel}"
            local = sample_root / dest_sub
            scp(args.host, remote, local, optional, args.dry_run)

        # CODEX (cols 3 & 5)
        scp(args.host, row["codex_tif"], args.data_dir / f"{s}_reference.tif", False, args.dry_run)
        if row["codex_csv"]:
            scp(args.host, row["codex_csv"], args.data_dir / f"{s}_codex_cells.csv", True, args.dry_run)

    # emit the local sample-table (./data paths) for steps 1 & 3
    rows = read_comp_table(args.comp_table)
    lines = ["sample_name\tvisium_path\tcodex_reference_tif\tcodex_channel"]
    for r in rows:
        if wanted and r["sample"] not in wanted:
            continue
        lines.append("\t".join([
            r["sample"],
            str(args.data_dir / r["sample"] / "segmented_outputs"),
            str(args.data_dir / f"{r['sample']}_reference.tif"),
            r["channel"] or "34",
        ]))
    out = args.sample_table_out
    if args.dry_run:
        print(f"  + write {out}:\n    " + "\n    ".join(lines))
    else:
        out.write_text("\n".join(lines) + "\n")
        print(f"Wrote {out}")

    print("Done. data/ populated. Run locally with: --sample-table", out)


if __name__ == "__main__":
    main()
