# Run data

One `.npy` per (method, seed). Each file is a 1-D `float64` array of length
`epoch2` (see `code/params.ini`): the mean episode reward at each PPO iteration.

```
data/
├── walker/  <prefix>_<seed>.npy   seeds 1..9   length 8000 (plotted to 6500)
├── ant/     <prefix>_<seed>.npy   seeds 1..8   length 3000
├── ip/      <prefix>_<seed>.npy   seeds 1..10  length 300
└── idp/     <prefix>_<seed>.npy   seeds 1..10  length 1500 (motif) / 2000 (Vanilla)
```

All four use the standard continuous-evaluation reward curves. In `idp` the four
motif methods were trained for 1500 iterations and the Vanilla baseline for
2000, so its curve extends further along the x-axis (this is the original
`idp_2000sad.svg` figure).

## Method = file-prefix = legend

Each `.npy` file's prefix is now exactly the method name shown on the figure,
so `data/<env>/<METHOD>_<seed>.npy` plots as `<METHOD>`. The methods and their
motif targets (from the training config) are:

| method (= file prefix = legend) | motif target |
| --- | --- |
| `FRP`              | motif #2 = 0.25 |
| `AVE`              | motif #2 = 0.40 |
| `MOP`              | motif #12 = 0.25 |
| `MOP_E` (walker/ant) / `MOP-E` (ip/idp) | motif #12 = 0.40 |
| `Vanilla`          | none (plain PPO) |

> The on-figure label `MOP_E` (underscore) is used for walker/ant and `MOP-E`
> (hyphen) for ip/idp, matching each figure's original legend text verbatim.

Note: the **training code** in `code/` still emits the original prefixes
(`2`, `2E`, `12`, `12E`, `ORBI`, … per `code/run_batch_*.sh`). The data files
here have been renamed to the final method labels; align `code/` to these names
(or rename fresh training outputs) before re-plotting newly trained runs.

These `.npy` files are byte-for-byte copies of the original training outputs
(only renamed).
