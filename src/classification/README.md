# classification — noise-robustness experiments

Three noise-robustness figures: a recurrent classifier (with motif-frequency
pre-shaping of the recurrent matrix) is trained on each dataset, then evaluated
under increasing Gaussian input noise. Each figure compares the motif methods
(MOP / AVE / FRP / FRP-E) against Vanilla.

This thread's files live across the top-level artifact dirs (paths from the
repository root):

```
data/classification/      run data — the noise-eval curves
plots/classification/     plotting code — turns data into figures
figures/classification/   result figures (SVG + PNG preview)
src/classification/       training code (this directory)
```

Reproduction is staged: the code runs, and **data + plotting regenerate the
figures exactly** (verified curve-for-curve against the originals — tidigits/dvs
identical, smnist within 8e-6 px = floating-point noise).

## Result figures

| Figure | Dataset | Original |
| --- | --- | --- |
| `figures/classification/smnist.svg` | sequential-MNIST | `smnist_line_ANN_v2.svg` |
| `figures/classification/tidigits.svg` | TIDIGITS (spoken digits) | `tidigits_ANN_v2.svg` |
| `figures/classification/dvs.svg` | DVS128-Gesture | `dvs.svg` |

x = input-noise level, y = accuracy; mean over seeds with a shaded band.

## Run data + plotting

`data/classification/` holds the per-(method, seed) noise sweeps; see
`data/classification/README.md` for the per-dataset label→colour maps.

```bash
cd plots/classification
python3 -m pip install matplotlib numpy
bash plot_all.sh           # writes ../../figures/classification/{smnist,tidigits,dvs}.svg (+ .png)
# or individually: python3 draw_smnist.py / draw_tidigits.py / draw_dvs.py
```

## Training code

`src/classification/classified/` is the smnist + tidigits training/eval package.

```bash
cd src/classification
pip install -r requirements.txt
# train (motif pre-training + classifier); --fre is the 13-D motif target.
# --prefix is just the output filename label, so name it after the method
# (here motif #12 = 0.25 => the "AVE" method):
python -m classified.main train --dataset tidigits --run-id 0 --prefix AVE \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.25 -1
# noise-robustness sweep (writes outputs/<dataset>/noise_eval/<prefix>_<run>[_discrete].npy):
python -m classified.main noise --dataset tidigits --run-id 0 --prefix AVE \
  --max-var 1.0 --steps 26 --discrete --discrete-levels 32
# batch over all methods × seeds:
bash run_tidigits.sh ; bash run_tidigits_noise_discrete.sh
bash run_smnist.sh   ; bash run_smnist_noise_discrete.sh
```

> The bundled `run_*.sh` still use the original prefixes (`2`/`2E`/`12`/`12E`),
> so their outputs need renaming to the method labels (see
> `data/classification/README.md`) before re-plotting.

The `noise` sub-command saves `(variance, acc_cont, acc_disc)` columns — exactly
the `data/classification/` layout. `train` saves a checkpoint under
`outputs/<dataset>/models/`. tidigits ships its dataset in
`src/classification/datasets/TIDIGITS/`; smnist pulls MNIST via Keras (hence the
`tensorflow` dependency); both run on CPU or CUDA.

Core pieces (`src/classification/classified/tools.py`): `motifRegular` (13-class
three-node motif-frequency loss on the recurrent weights), `RNNClassifier`,
`noise_sweep`. Per-dataset hyper-parameters live in
`src/classification/classified/config.py`.

> **dvs training is NOT included.** It used a separate script on a collaborator's
> machine (`dvsgesmotif.py`, DVS128-Gesture SNN). Only its outputs survive, which
> is enough to regenerate `figures/classification/dvs.svg`. See
> `src/classification/dvs_training/README.md`.
