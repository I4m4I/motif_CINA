# rl — Motif-regularized PPO

Four reinforcement-learning experiments (Walker2d, Ant, InvertedPendulum,
InvertedDoublePendulum). Each trains a recurrent PPO policy whose actor recurrent
matrix is pre-shaped by a three-node directed **motif** frequency regularizer,
comparing motif targets (MOP / MOP-E / FRP / AVE) against Vanilla PPO.

This thread's files live across the top-level artifact dirs (paths below are
from the repository root):

```
data/rl/      run data — <METHOD>_<seed>.npy reward curves
plots/rl/     plotting code — turns data/rl into figures/rl
figures/rl/   result figures — the SVGs (+ PNG previews)
src/rl/       training code (this directory)
```

Re-running the training (many seeds × tasks × thousands of PPO iterations on
MuJoCo) is expensive, so reproduction is staged:

1. **The code runs** — `src/rl/` is a complete training entry point (verified to
   compile and execute its core on CPU).
2. **Data + plotting reproduce the figures** — `plots/rl/` reads the stored
   `data/rl/*.npy` and regenerates every figure in `figures/rl/` exactly.

## Result figures

| Figure | Task | Env id |
| --- | --- | --- |
| `figures/rl/walker.svg` | Walker2d | `Walker2d-v2` |
| `figures/rl/ant.svg` | Ant | `Ant-v2` |
| `figures/rl/ip.svg` | InvertedPendulum | `InvertedPendulum-v2` |
| `figures/rl/idp.svg` | InvertedDoublePendulum | `InvertedDoublePendulum-v2` |

Each is a `figsize=(4,3)` reward-vs-steps curve: mean over seeds with a shaded
band (±1 std for walker/ant, ±1 SEM for ip/idp), max-smoothed with window 3.
A `.png` preview sits next to every `.svg`.

> **idp note.** `figures/rl/idp.svg` reproduces the original `idp_2000sad.svg`.
> The source data was recovered from a backup and the regenerated curves match
> the original **exactly** (curve-for-curve RMS = 0). The four motif methods run
> to 1500 steps and the Vanilla baseline to 2000, which is why its curve extends
> further right.

## Run data + plotting

`data/rl/<env>/` holds one `.npy` per (method, seed) — a 1-D array of
per-iteration mean episode reward, named `<METHOD>_<seed>.npy` where `<METHOD>`
is exactly the legend label (`FRP`, `AVE`, `MOP`, `MOP_E`/`MOP-E`, `Vanilla`);
see `data/rl/README.md`.

```bash
cd plots/rl
python3 -m pip install matplotlib numpy   # only deps needed for plotting
bash plot_all.sh                          # writes ../../figures/rl/*.svg (+ .png)
# or one at a time: python3 draw_walker.py / draw_ant.py / draw_ip.py / draw_idp.py
```

Smoothing is configurable, e.g. `python3 draw_ant.py --smooth-mode mean --smooth-window 5`.
`_common.py` holds the shared loading/smoothing/plotting logic; each
`draw_<env>.py` only declares that env's methods, colors, seed count, and axis
limits.

## Training code

`src/rl/` is the training entry point.

```bash
cd src/rl
pip install -r requirements.txt   # needs legacy gym + mujoco-py, see the file
# Vanilla PPO (all motif targets negative => no motif pretraining):
python main.py --env ip --seed 1 --prefix Vanilla --cuda 0 \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
# motif-regularized run (target motif #12 = 0.25 => the "MOP" method;
# --prefix is just the output filename label, so name it after the method):
python main.py --env ip --seed 1 --prefix MOP --cuda 0 \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.25 -1
# batch over all methods × seeds, sharded across GPUs:
bash run_batch_walker.sh   # also run_batch_ant.sh / run_batch_ip.sh / run_batch_idp.sh
```

Outputs land in `src/rl/output/<env>/<prefix>_<seed>.npy` (and
`<prefix>_<seed>_discrete.npy` for the discretized-hidden-state evaluation).
The training `--prefix` names are the original ones (e.g. `2`, `12`, `ORBI`);
the plotting in `data/rl`+`plots/rl` uses the final method labels (`FRP`, `MOP`,
…). To plot freshly trained runs, rename the outputs to the method labels (see
`data/rl/README.md`) and copy them into the matching `data/rl/<env>/`.

Key pieces (`src/rl/tools.py`): `motifRegular` (differentiable 13-class
three-node motif frequency loss on the actor's `weight_hh`), `Actor`/`Critic`
(`nn.RNNCell`-based, with a discretized-hidden-state forward), and `Ppo`
(clipped PPO + GAE). See `main.py` for the two-stage train loop
(motif pretraining → PPO) and `params.ini` for per-env settings.

### `--fre` (13-D motif target vector)
One target frequency per three-node motif class. A value `>= 0` constrains that
motif; `< 0` ignores it. If all 13 are negative, motif pretraining is skipped
(plain PPO). The method↔target mapping used for these figures lives in the
`run_batch_*.sh` scripts.
