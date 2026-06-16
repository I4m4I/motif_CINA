# motif_fre data

`fig2_region_motif_counts_log10_1p8NZ_5um_thr0.1.csv` — per-region three-node
motif-frequency table (50 rows = region × cluster, 34 columns):

- `region`, `cluster`, `param_um` (=5), `strength_threshold` (=0.1), `node`,
  `sparsity`, `invalid_log10_1p8NZ_motifs`
- `M01..M13_motif_count` — raw counts of each of the 13 directed three-node motifs
- `M01..M13_log10_1p8NZ` — the display value `log10(1 + 8*NZ)`, where `NZ` is the
  normalized z-score of each motif frequency against a null model.

`../../plots/motif_fre/draw_motif_comparison.py` reads the `FRP` and `MOp` rows,
inverts `NZ = (10^x - 1)/8`, builds the "enhanced" vector (natural-log biasing +
L2 renormalization), re-applies `log10(1 + 8*NZ)`, and draws
`comparison_FRP_chart.svg` and `comparison_MOp_chart.svg` (enhanced vs original).

`../../plots/motif_fre/original_notebook.ipynb` is the original notebook this was
derived from (it used hard-coded NZ vectors instead of the CSV).

## NZ matrix (Fig.2 heatmaps)

`fig2_region_NZ_matrix.csv` — the (50 region × 13 motif) raw NZ-score matrix,
regions ordered by the manuscript's 4-cluster custom order. Extracted from the
large source pickle `wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl`
(`results[5]`, 5 µm), which is too large to ship; this small matrix is all the
two Fig.2 heatmaps need.

`../../plots/motif_fre/draw_nz_heatmaps.py` reads it and draws:

- `nz_score_heatmap.svg` — `log10(1 + 8*NZ)` heatmap (region × motif), the
  original `3.nz_score_heatmap.svg`.
- `cosine_similarity_matrix.svg` — region×region cosine similarity over the
  first 12 motif NZ values, the original `4.cosine_similarity_matrix_RdBu_r.svg`.

Both reproduce the originals exactly (cosine matrix pixel-identical; heatmap data
identical, ±1 px margin from a matplotlib-version difference).
`fig2_source_notebook.ipynb` is the full official Fig.2 notebook for reference
(needs the large pickle + `umap`/`adjustText`).
