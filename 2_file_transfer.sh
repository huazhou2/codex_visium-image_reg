#!/usr/bin/env bash
set -euo pipefail

# Step 2: sync this compact pipeline and landmark JSONs to BigPurple.

LOCAL_PROJECT="${LOCAL_PROJECT:-/Volumes/hua_mac/research/aris/harvey/spatial_202506/img_reg_202606}"
REMOTE_HOST="${REMOTE_HOST:-zhouh05@bigpurple.nyumc.org}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/gpfs/data/tsirigoslab/home/zhouh05/harvey/202507/img_reg_202606}"

echo "Local:  ${LOCAL_PROJECT}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_PROJECT}"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_PROJECT}'"

rsync -av \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'data/' \
  --exclude 'registered/' \
  "${LOCAL_PROJECT}/" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/"

cat <<EOF

Done.

Next on BigPurple:
  cd ${REMOTE_PROJECT}
  # comp_table already has the cluster /gpfs paths; use it directly (step 3 ignores its 5th column)
  python 3_online_parallel_overlay_transform.py --sample-table comp_table --points-dir points --out-dir registered --workers 6
EOF

