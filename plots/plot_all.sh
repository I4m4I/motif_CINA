#!/usr/bin/env bash
# Regenerate every figure (rl + classification + motif_fre + topology) from data/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="${PYTHON:-python3}"
echo "### rl"; (cd "$SCRIPT_DIR/rl" && PYTHON="$PYTHON" bash plot_all.sh)
echo "### classification"; (cd "$SCRIPT_DIR/classification" && PYTHON="$PYTHON" bash plot_all.sh)
echo "### motif_fre"; "$PYTHON" "$SCRIPT_DIR/motif_fre/draw_motif_comparison.py"; "$PYTHON" "$SCRIPT_DIR/motif_fre/draw_nz_heatmaps.py"
echo "### topology"
"$PYTHON" "$SCRIPT_DIR/topology/draw_topology.py" >/dev/null && echo "saved topology modularity.svg + smallworld.svg"
"$PYTHON" "$SCRIPT_DIR/topology/fig6f/plot_cumulative_distribution_curves_fig6f.py" >/dev/null && echo "saved topology fig6f cumulative_distribution_curves"
"$PYTHON" "$SCRIPT_DIR/topology/fig6g/plot_KS_test_heatmap_fig6g.py" >/dev/null && echo "saved topology fig6g KS_test_heatmap"
"$PYTHON" "$SCRIPT_DIR/topology/figS10/plot_weight_matrices_figS10.py" >/dev/null && echo "saved topology figS10 weight_matrices"
"$PYTHON" "$SCRIPT_DIR/topology/fig6a/community_partitioning.py" >/dev/null && echo "saved topology fig6a community-partition panels"
echo "All figures written to ../figures/"
