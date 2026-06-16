# Noise-robustness run data

Three datasets, one figure each. Each curve is mean accuracy vs Gaussian input
noise, averaged over seeds. The five methods are the motif targets MOP / AVE /
FRP / FRP-E plus the Vanilla baseline.

```
data/
├── smnist/    <METHOD>_<seed>.npy            seeds 0..4   2-D (col0=noise, col1=acc_cont, col2=acc_disc)
├── tidigits/  <METHOD>_<seed>_discrete.npy   seeds 0..4   2-D (same columns)
└── dvs/       <...freclass<METHOD>_seed<seed>...>/evaluation_results_1.csv   seeds 0..9   (noise, accuracy)
```

The plotting reads **col 1** (continuous-path accuracy) for smnist/tidigits, and
the CSV `accuracy` column for dvs.

## Method = file-prefix = legend

Names are normalized: every file/dir prefix is now exactly the method name on
the figure (`MOP`, `AVE`, `FRP`, `FRP-E`, `Vanilla`). Colours:

| method | smnist / tidigits | dvs |
| --- | --- | --- |
| `MOP`     | yellow `#F3CC4F` | blue `#529DCB` |
| `AVE`     | green `#009944`  | green `#009944` |
| `FRP`     | orange `#F18D00` | orange `#F18D00` |
| `FRP-E`   | red `#E60012`    | red `#E60012` |
| `Vanilla` | blue `#529DCB`   | yellow `#F3CC4F` |

> The dvs colour scheme differs from smnist/tidigits (MOP↔Vanilla swap blue/
> yellow); this matches the original dvs figure verbatim and is left as-is.
> Each `plot/draw_*.py` is the source of truth.

## Provenance

The files are byte-for-byte copies of the original training outputs, **renamed**
to the method labels. The original source prefixes were:

- **smnist** — from `classifi/old/motif_SNN/outputs/smnist/noise_eval`, source
  prefixes `MOP_E→MOP`, `MOP→AVE`, `ORBI→FRP`, `ORBI_E→FRP-E`, `Vanilla`.
- **tidigits** — from `classifi/outputs/tidigits/noise_eval` (`_discrete` files),
  source prefixes `12E→MOP`, `12→AVE`, `2E→FRP`, `2→FRP-E`, `Vanilla`.
- **dvs** — `evaluation_results_1.csv` per run under `tmp/logsdisc`, source
  methods `fre12E→MOP`, `fre12→AVE`, `fre2→FRP`, `fre2E→FRP-E`, `Vanilla`
  (training source missing, see `../code/dvs_training/`).

Note: the **training code** in `code/` still emits the original prefixes
(`2`, `12`, `ORBI`, …, per `code/run_*.sh`); align `code/` to these method labels
before dropping freshly trained runs back into `data/`.
