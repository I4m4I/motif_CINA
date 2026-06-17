# Incorporation of single-neuron projectome-based connectivity motifs enhances the cortex-specific performance of artificial neural networks

Reproducible code, run data, and result figures for the paper. A three-node
directed **motif**-frequency prior — derived from the single-neuron mouse
projectome connectome — is built into recurrent / state-space networks, and the
motif variants (**FRP / MOP / Average**, with `_E` "enhanced" forms) are compared
against vanilla baselines across reinforcement learning, noisy classification,
language modeling, brain-signal decoding, and graph-topology analyses.

## What's in here

The repository holds two kinds of content plus the paper's source data:

1. **Analysis / benchmark threads** — organized by artifact type at the top
   level (`data/`, `figures/`, `plots/`, `src/`), each split by sub-project.
   Every figure regenerates from `data/` + `plots/` in seconds, no GPU needed.
2. **Fig. 5 Motif-Mamba experiments** — two self-contained sub-projects
   (`bmi_decoding/`, `language_qa/`) with their own runners and heavier
   dependencies (Mamba / `mamba-ssm`, lm-evaluation-harness).
3. **`network_motif_source_data.zip`** — the paper's source data (see below).

## Layout

```
motif_nature_neuroscience/
├── data/        # run data / inputs        (rl, classification, motif_fre, topology)
├── figures/     # result figures (SVG+PNG) (rl, classification, motif_fre, topology)
├── plots/       # plotting code → figures  (rl, classification, motif_fre, topology) + plot_all.sh
├── src/         # training code            (rl, classification)
│
├── bmi_decoding/   # Fig.5 e-j — MotifMamba brain-signal decoding (Jango / Calcium / mice-lick)
├── language_qa/    # Fig.5 a-d — MotifMamba-130M language QA benchmarks (6 tasks)
│
├── network_motif_source_data.zip   # paper source data (tabular / CSV)
├── environment.yml                 # conda env (PyTorch + mamba-ssm), mainly for Fig.5
└── requirements.txt                # pip deps for the analysis threads
```

## Figures ↔ paper

| Paper | Thread / folder | Figures |
| --- | --- | --- |
| **Fig. 2** (regional motif profiles) | `*/motif_fre/` | `comparison_FRP_chart`, `comparison_MOp_chart`, `nz_score_heatmap`, `cosine_similarity_matrix` |
| **Fig. 4ef** (model benchmarks) | `*/rl/` + `*/classification/` | `walker`, `ant`, `ip`, `idp` (RL) · `smnist`, `tidigits`, `dvs` (noise robustness) |
| **Fig. 5 a-d** (language QA) | `language_qa/results/` | `Fig5b` (radar), `Fig5c` (accuracy table) |
| **Fig. 5 e-j** (BMI decoding) | `bmi_decoding/figures/` | per-dataset decoding accuracy |
| **Fig. 6** (graph topology) | `*/topology/` | `modularity`, `smallworld` |

## Reproduce the analysis figures

All 13 analysis figures (rl 4 + classification 3 + motif_fre 4 + topology 2)
regenerate from the stored data:

```bash
pip install -r requirements.txt        # matplotlib, numpy, scipy, pandas, umap-learn, …
cd plots && bash plot_all.sh           # writes ../figures/<thread>/*.svg (+ .png)
```

Per-thread details (data format, method↔colour maps, training commands) are in
each `data/<thread>/README.md` and `src/<thread>/README.md`. Training code for
`rl` and `classification` lives under `src/` and is optional — the figures are
always reproducible from `data/`.

> **Time budget.** Environment setup ≈ **35 min** (dominated by the `mamba-ssm`
> CUDA build). Full reproduction — training + evaluation across all experiments
> — ≈ **40 h** on a single GPU, ubuntu. Regenerating the figures from the bundled
> `data/` takes seconds.

## Fig. 5 — Motif-Mamba experiments

These are heavier and self-contained (each has its own README / environment):

- **`bmi_decoding/`** — Mamba vs MotifMamba on three brain-signal decoding tasks
  (Jango center-out, Calcium 2-AFC, mice fixed-interval lick). One-click
  `./run_all.sh`; needs `mamba-ssm` (CUDA) and the task datasets.
- **`language_qa/`** — MotifMamba-130M (low-rank PQ motif adapter on a frozen
  Mamba-130M) on six QA benchmarks (LAMBADA, HellaSwag, PIQA, ARC-e/c,
  WinoGrande). Runs through lm-evaluation-harness; bundled result JSON + figures
  in `language_qa/results/`.

`FRP` / `MOP` / `Average` (and `_E`) are **motif-frequency targets, not separate
architectures** — the same backbone with different motif priors.

## Source data

`network_motif_source_data.zip` is the paper's **source-data archive** in tabular
(CSV) form — the underlying numbers behind the figures:

- `connection_data/connection_matrices_5um/` — processed 5 µm neuron-by-neuron
  connection matrices for 50 mouse brain regions.
- `motif_profiles/fig2_region_motif.csv` — regional three-node motif counts and
  normalized motif profiles (Fig. 2).
- `model_evaluation_data/` — benchmark results for Fig. 4ef (RL + classification),
  Fig. 5 h-j (BMI), and Fig. 6 c-f / k (modularity, small-worldness, action
  deviation).

The single-neuron mouse connectome data are from Digital Brain CEBSIT (ION):
<https://www.digital-brain.cn/> · Mouse Projectome Atlas
<https://mouse.digital-brain.cn/projectome>.

## Dependencies

- **Analysis threads** (`plots/`, `src/rl`, `src/classification`):
  `requirements.txt` — matplotlib, numpy, scipy, pandas, umap-learn, torch, gym, …
- **Fig. 5 Motif-Mamba**: `environment.yml` (conda env `motif`, PyTorch +
  `mamba-ssm`) plus each sub-project's own `requirements.txt`.
