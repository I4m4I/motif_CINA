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
