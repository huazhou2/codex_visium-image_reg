#!/usr/bin/env bash
set -euo pipefail

# Step 0: table-driven download of all needed files from the cluster into ./data/.
#
# Reads comp_table (5 tab-separated columns):
#   1 sample_name
#   2 visium spaceranger outs dir   (contains segmented_outputs/)   <- VISIUM
#   3 codex reference analysis.tif                                    <- CODEX
#   4 codex_channel
#   5 codex object_Data.csv         (CODEX cells)                     <- CODEX
#
# Produces the layout the pipeline expects:
#   data/<sample>/segmented_outputs/{cell_segmentations.geojson,filtered_feature_cell_matrix.h5,spatial/*}
#   data/<sample>_reference.tif
#   data/<sample>_codex_cells.csv
#
# Usage:
#   bash 0_download_from_comp_table.sh                # all samples
#   SAMPLES="NYU658 NYU851" bash 0_download_from_comp_table.sh
#   DRY_RUN=1 bash 0_download_from_comp_table.sh      # print scp commands only

REMOTE_HOST="${REMOTE_HOST:-zhouh05@bigpurple.nyumc.org}"
COMP_TABLE="${COMP_TABLE:-comp_table}"
DATA_DIR="${DATA_DIR:-data}"
SAMPLES="${SAMPLES:-}"
DRY_RUN="${DRY_RUN:-0}"

run() { if [ "$DRY_RUN" = "1" ]; then echo "  + $*"; else "$@"; fi; }

get() {  # get <remote_abs_path> <local_dest> [optional]
  local src="$1" dst="$2" opt="${3:-}"
  if [ "$opt" = optional ]; then
    run scp "$REMOTE_HOST:$src" "$dst" || echo "  (skip missing: $src)"
  else
    run scp "$REMOTE_HOST:$src" "$dst"
  fi
}

while IFS=$'\t' read -r sample visium_outs codex_tif channel codex_csv; do
  [ -z "${sample:-}" ] && continue
  case "$sample" in \#*) continue;; esac
  if [ -n "$SAMPLES" ] && ! grep -qw "$sample" <<<"$SAMPLES"; then continue; fi

  visium_outs="${visium_outs%/}"
  seg="$visium_outs/segmented_outputs"
  dst="$DATA_DIR/$sample/segmented_outputs"
  echo "==> $sample"
  run mkdir -p "$dst/spatial"

  # VISIUM (col 2)
  get "$seg/cell_segmentations.geojson"      "$dst/"           optional
  get "$seg/filtered_feature_cell_matrix.h5" "$dst/"           optional
  get "$seg/spatial/tissue_hires_image.png"  "$dst/spatial/"
  get "$seg/spatial/scalefactors_json.json"  "$dst/spatial/"

  # CODEX (cols 3 & 5)
  get "$codex_tif"                           "$DATA_DIR/${sample}_reference.tif"
  [ -n "${codex_csv:-}" ] && get "$codex_csv" "$DATA_DIR/${sample}_codex_cells.csv" optional
done < "$COMP_TABLE"

# emit the local sample-table (./data paths) for steps 1 & 3
SAMPLE_TABLE_OUT="${SAMPLE_TABLE_OUT:-sample_table.data.tsv}"
if [ "$DRY_RUN" = "1" ]; then
  echo "  + write $SAMPLE_TABLE_OUT (from comp_table, ./data paths)"
else
  {
    printf 'sample_name\tvisium_path\tcodex_reference_tif\tcodex_channel\n'
    while IFS=$'\t' read -r sample visium_outs codex_tif channel codex_csv; do
      [ -z "${sample:-}" ] && continue
      case "$sample" in \#*) continue;; esac
      if [ -n "$SAMPLES" ] && ! grep -qw "$sample" <<<"$SAMPLES"; then continue; fi
      printf '%s\t%s\t%s\t%s\n' "$sample" "$DATA_DIR/$sample/segmented_outputs" \
        "$DATA_DIR/${sample}_reference.tif" "${channel:-34}"
    done < "$COMP_TABLE"
  } > "$SAMPLE_TABLE_OUT"
  echo "Wrote $SAMPLE_TABLE_OUT"
fi

echo "Done. data/ populated. Run locally with: --sample-table $SAMPLE_TABLE_OUT"
