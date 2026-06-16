# DVS128-Gesture training — source NOT included

Unlike smnist/tidigits (the `classified/` package here), the DVS128-Gesture
noise-robustness experiment was trained with a separate script that lives on a
collaborator's machine and is **not available in this bundle**:

```
/mnt/hdd/data/haochonghe/motif/dvsgesmotif.py     # training entry point (missing)
/mnt/hdd/data/haochonghe/datasets/DVS             # DVS128-Gesture dataset (missing)
```

What we DO have is the training/eval **output** it produced, which is enough to
regenerate `figures/dvs.svg`:

- `../../data/dvs/<...freclass<method>_seed<seed>...>/evaluation_results_1.csv`
  — per-run noise-vs-accuracy sweeps (5 methods × 10 seeds), consumed by
  `../../plot/draw_dvs.py`.

`args_example.txt` is a verbatim copy of one run's `args.txt`, recording the
exact training configuration so the run can be reproduced once the script is
recovered. Key settings:

- `T=50` timesteps, batch `b=32`, `adam` `lr=0.001`, `epoch1=100` (motif
  pre-training), SNN on DVS128-Gesture.
- `fre=[-1, 0.25, -1, ...]` — the same 13-D motif-frequency target vector used
  by `classified` and by the RL experiments; the directory name encodes the
  method (`freclass<method>`) and seed (`_seed<n>`).
- methods: `fre12E`, `fre12`, `fre2`, `fre2E`, `Vanilla`.

To fully reproduce dvs, recover `dvsgesmotif.py` from the collaborator
(haochonghe) and point it at a local DVS128-Gesture download.
