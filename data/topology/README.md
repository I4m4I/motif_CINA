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

## Fig.6 / Fig.S10 connectivity panels

Additional Fig.6 / Fig.S10 panels share this flat layout — inputs in
`data/topology/`, scripts in `../../plots/topology/`, outputs in
`figures/topology/` (same as `modularity` / `smallworld` above).

- **Fig.6a** — community partitioning of the recurrent weight graphs.
  `{FRP,MOp,Average,Vanilla}.pt` are prepared checkpoints; the script
  `community_partitioning.py` reads the `wrec` tensor, binarizes it, runs
  Infomap community detection, and draws `figures/topology/network_{label}.png`.
  Needs `torch` + `infomap`. The Vanilla panel uses a fixed random 25-node
  sample (seed 42) for reproducibility.
- **Fig.6f** — cumulative weight-distribution curves.
  `cumulative_distribution_curves_fig6f.npz` holds per-model CDF (x, y);
  `KS_test_values_for_fig6f_annotation.csv` holds the K–S annotation.
  `plot_cumulative_distribution_curves_fig6f.py` →
  `figures/topology/cumulative_distribution_curves_fig6f.{png,svg}`.
- **Fig.6g** — pairwise K–S test heatmap. `KS_test_heatmap_fig6g.csv` holds the
  K–S value / p-value / stars per model pair. `plot_KS_test_heatmap_fig6g.py` →
  `figures/topology/KS_test_heatmap_fig6g.{png,svg}`. Needs `seaborn`.
- **Fig.S10** — recurrent weight matrices. `weight_matrices_figS10.npz` holds
  one weight matrix per model. `plot_weight_matrices_figS10.py` →
  `figures/topology/weight_matrices_figS10.{png,svg}` plus per-model
  `figures/topology/weight_matrices_figS10_individual/{label}.svg`.

Regenerate all of them via `../../plots/plot_all.sh` (topology section) or run a
single script directly, e.g.
`python3 ../../plots/topology/plot_cumulative_distribution_curves_fig6f.py`.
