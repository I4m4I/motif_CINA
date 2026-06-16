#!/usr/bin/env bash
# Regenerate every figure (rl + classification + motif_fre + topology) from data/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="${PYTHON:-python3}"
echo "### rl"; (cd "$SCRIPT_DIR/rl" && PYTHON="$PYTHON" bash plot_all.sh)
echo "### classification"; (cd "$SCRIPT_DIR/classification" && PYTHON="$PYTHON" bash plot_all.sh)
echo "### motif_fre"; "$PYTHON" "$SCRIPT_DIR/motif_fre/draw_motif_comparison.py"; "$PYTHON" "$SCRIPT_DIR/motif_fre/draw_nz_heatmaps.py"
echo "### topology"; "$PYTHON" "$SCRIPT_DIR/topology/draw_topology.py" >/dev/null && echo "saved topology modularity.svg + smallworld.svg"
echo "All figures written to ../figures/"
