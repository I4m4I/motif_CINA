# topology data

Network-property analysis for Fig.6 — modularity and small-worldness of the
trained recurrent weight graphs (thresholded into binary connectivity).

- `Q_Qer_best.jsonl` (3780 records) — per `(net_type, instance n, threshold
  p_thre)`: best modularity `Q_best` of the observed graph vs an ER null model
  (`Q_ER_mean`, `Q_ER_std`, `n_ER`), at `gamma=1.0`.
- `swER_all.jsonl` (660 records) — per `(net_type, n, p_thre)`: small-world
  metrics `sw_er{obs: C/L/E, rand(ER): C/L/E means±sd, ratio: sigma_CL, ...}`.

`net_type ∈ {FRP, MOp, Average, Vanilla, …}`. `../../plots/topology/draw_topology.py`
picks one threshold per network, aggregates mean ± SEM across instances, draws
`figures/topology/modularity.svg` and `smallworld.svg`, and prints paired / Welch /
one-sample t-tests.

> **Vanilla** bars in both figures are placeholder values (`Q=0`,
> `C/L/sigma=1`) — the source data has no real Vanilla entry for these panels.
> This matches the original notebook (`../../plots/topology/original_notebook.ipynb`).

Plotting needs `matplotlib`, `numpy`, `scipy`.

## Fig.6 connectivity panels

Additional Fig.6 / Fig.S10 panels are organized one sub-folder per panel under
`data/topology/<panel>/`, `../../plots/topology/<panel>/`, and
`figures/topology/<panel>/` (same data → plots → figures split as above). Each
script reads its `data/topology/<panel>/` inputs and writes its
`figures/topology/<panel>/` outputs.

- **`fig6a/`** — community partitioning of the recurrent weight graphs.
  `fig6a/models/{FRP,MOp,Average,Vanilla}.pt` are prepared checkpoints;
  `plot...fig6a` is `community_partitioning.py`, which reads the `wrec` tensor,
  binarizes it, runs Infomap community detection, and draws
  `figures/topology/fig6a/network_{label}.png`. Needs `torch` + `infomap`. The
  Vanilla panel uses a fixed random 25-node sample (seed 42) for reproducibility.
- **`fig6f/`** — cumulative weight-distribution curves.
  `cumulative_distribution_curves_fig6f.npz` holds per-model CDF (x, y);
  `KS_test_values_for_fig6f_annotation.csv` holds the K–S annotation. →
  `figures/topology/fig6f/cumulative_distribution_curves_fig6f.{png,svg}`.
- **`fig6g/`** — pairwise K–S test heatmap. `KS_test_heatmap_fig6g.csv` holds the
  K–S value / p-value / stars per model pair. → `figures/topology/fig6g/
  KS_test_heatmap_fig6g.{png,svg}`. Needs `seaborn`.
- **`figS10/`** — recurrent weight matrices. `weight_matrices_figS10.npz` holds
  one weight matrix per model. → `figures/topology/figS10/
  weight_matrices_figS10.{png,svg}` plus per-model
  `weight_matrices_figS10_individual/{label}.svg`.

Regenerate all of them via `../../plots/plot_all.sh` (topology section) or run a
single script directly, e.g. `python3 ../../plots/topology/fig6f/plot_cumulative_distribution_curves_fig6f.py`.
