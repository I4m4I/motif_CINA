#!/usr/bin/env python3
"""Shared plotting helpers for the motif RL reward curves.

Every per-environment script (``draw_walker.py`` etc.) only declares its
configuration (which prefixes/colors/seeds to use) and calls
:func:`plot_means_and_variances`. The data is read from ``../data/<env>`` and
the figure is written to ``../figures/<env>.svg`` relative to this repository,
so the scripts are runnable from anywhere without editing absolute paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Layout: <repo>/plots/rl/_common.py , <repo>/data/rl/ , <repo>/figures/rl/
PLOT_DIR = Path(__file__).resolve().parent          # plots/rl
SUB = PLOT_DIR.name                                 # "rl"
REPO_ROOT = PLOT_DIR.parent.parent                  # repo root
DATA_ROOT = REPO_ROOT / "data" / SUB
FIG_ROOT = REPO_ROOT / "figures" / SUB

SMOOTH_MODES = {"none", "max", "mean"}


def load_series(
    base_dir: Path,
    prefix: str,
    run_ids,
    *,
    suffix: str = "",
    clip: int | None = None,
) -> np.ndarray:
    """Load all 1D runs for a prefix and stack into shape (num_runs, length).

    ``suffix`` lets discrete-evaluation files (``<prefix>_<seed>_discrete.npy``)
    be selected; ``clip`` truncates each run to the first ``clip`` steps.
    """
    series: list[np.ndarray] = []
    target_length: int | None = None
    for run_id in run_ids:
        path = base_dir / f"{prefix}_{run_id + 1}{suffix}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        arr = np.load(path)
        if clip is not None:
            arr = arr[:clip]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
        if target_length is None:
            target_length = arr.shape[0]
        elif arr.shape[0] != target_length:
            raise ValueError(
                f"Inconsistent length for {path}: {arr.shape[0]} vs {target_length}"
            )
        series.append(arr)
    if not series:
        raise RuntimeError(f"No runs loaded for prefix {prefix}")
    return np.vstack(series)


def smooth_series(series: np.ndarray, mode: str, window: int) -> np.ndarray:
    """Apply sliding window smoothing to a 1D array."""
    if window < 1:
        raise ValueError("Smoothing window must be at least 1")
    if mode not in SMOOTH_MODES:
        raise ValueError(f"Unsupported smoothing mode: {mode}")
    adjusted_window = min(window, series.size)
    if mode == "none" or adjusted_window == 1:
        return series.copy()
    pad_left = adjusted_window // 2
    pad_right = adjusted_window - pad_left - 1
    padded = np.pad(series, (pad_left, pad_right), mode="edge")
    windows = sliding_window_view(padded, window_shape=adjusted_window)
    if mode == "mean":
        return windows.mean(axis=-1, dtype=series.dtype)
    if mode == "max":
        return windows.max(axis=-1)
    raise ValueError(f"Unhandled smoothing mode: {mode}")


def apply_smoothing(stacked: np.ndarray, mode: str, window: int) -> np.ndarray:
    if mode == "none" or window <= 1:
        return stacked
    smoothed_runs = [smooth_series(run, mode, window) for run in stacked]
    return np.vstack(smoothed_runs)


def plot_means_and_variances(
    *,
    dataset_name: str,
    prefixes,
    mapped_prefixes,
    mapped_color,
    number_of_seeds: int,
    smooth_mode: str,
    smooth_window: int,
    suffix: str = "",
    clip: int | None = None,
    use_sem: bool = False,
    linewidth: float = 0.5,
    band_linewidth: float = 0.0,
    ylim: tuple[float, float] | None = None,
) -> Path:
    """Reproduce ``<dataset_name>.svg`` from the stored ``.npy`` reward curves."""
    base_dir = DATA_ROOT / dataset_name
    run_ids = list(range(number_of_seeds))

    plt.figure(figsize=(4, 3))
    for prefix in prefixes:
        stacked = load_series(base_dir, prefix, run_ids, suffix=suffix, clip=clip)
        smoothed = apply_smoothing(stacked, smooth_mode, smooth_window)
        mean_vals = smoothed.mean(axis=0)
        std_vals = smoothed.std(axis=0)
        if use_sem:
            std_vals = std_vals / np.sqrt(number_of_seeds)
        x = np.arange(mean_vals.shape[0])
        plt.plot(
            x,
            mean_vals,
            label=mapped_prefixes[prefix],
            color=mapped_color[prefix],
            linewidth=linewidth,
        )
        plt.fill_between(
            x,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.2,
            color=mapped_color[prefix],
            linewidth=band_linewidth,
        )
    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.title("training")
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FIG_ROOT / f"{dataset_name}.svg"
    plt.savefig(out_path, format="svg")
    # Also drop a raster preview next to the vector figure.
    plt.savefig(FIG_ROOT / f"{dataset_name}.png", dpi=130)
    print(f"saved {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward curves with optional smoothing."
    )
    parser.add_argument(
        "--smooth-mode",
        choices=sorted(SMOOTH_MODES),
        default="max",
        help="Smoothing strategy to apply before aggregation",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Sliding window size used for smoothing (ignored when mode is none)",
    )
    return parser.parse_args()
