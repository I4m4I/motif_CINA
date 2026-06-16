#!/usr/bin/env python3
"""Reproduce figures/tidigits.svg (the original tidigits_ANN_v2.svg).

Reads ../data/tidigits/<prefix>_<seed>_discrete.npy. Each file is a 2-D array;
column 1 is the accuracy used here. Curves are averaged over the first 5 seeds
and resampled to 11 points along the noise-step index; the band is +/- std.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATASET_NAME = "tidigits"
_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = _ROOT / "data" / "classification" / DATASET_NAME
FIG_DIR = _ROOT / "figures" / "classification"

PREFIXES = ["MOP", "AVE", "Vanilla", "FRP-E", "FRP"]
MAPPED_PREFIXES = {
    "MOP": "MOP",
    "AVE": "AVE",
    "FRP-E": "FRP-E",
    "FRP": "FRP",
    "Vanilla": "Vanilla",
}
MAPPED_COLOR = {
    "MOP": "#F3CC4F",
    "AVE": "#009944",
    "FRP-E": "#E60012",
    "FRP": "#F18D00",
    "Vanilla": "#529DCB",
}
NUMBER_OF_SEEDS = 5
RUN_IDS = list(range(NUMBER_OF_SEEDS))
SMOOTH_MODES = {"none", "max", "mean"}


def load_series(prefix: str) -> np.ndarray:
    """Load column 1 of all runs for a prefix; stack into (num_runs, length)."""
    series: list[np.ndarray] = []
    target_length: int | None = None
    for run_id in RUN_IDS:
        path = BASE_DIR / f"{prefix}_{run_id}_discrete.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        arr = np.load(path)[:, 1]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
        if target_length is None:
            target_length = arr.shape[0]
        elif arr.shape[0] != target_length:
            raise ValueError(f"Inconsistent length for {path}: {arr.shape[0]} vs {target_length}")
        series.append(arr)
    if not series:
        raise RuntimeError(f"No runs loaded for prefix {prefix}")
    return np.vstack(series)


def smooth_series(series: np.ndarray, mode: str, window: int) -> np.ndarray:
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
    return np.vstack([smooth_series(run, mode, window) for run in stacked])


def plot_means_and_variances(smooth_mode: str, smooth_window: int) -> None:
    plt.figure(figsize=(4, 3))
    for prefix in PREFIXES:
        stacked = load_series(prefix)
        smoothed = apply_smoothing(stacked, smooth_mode, smooth_window)
        mean_vals = smoothed.mean(axis=0)
        std_vals = smoothed.std(axis=0)
        x = np.arange(mean_vals.shape[0])
        num_points = min(11, x.shape[0])
        x_plot = np.linspace(x[0], x[-1], num_points, dtype=float)
        mean_plot = np.interp(x_plot, x, mean_vals)
        std_plot = np.interp(x_plot, x, std_vals)
        plt.plot(
            x_plot, mean_plot,
            label=MAPPED_PREFIXES[prefix], color=MAPPED_COLOR[prefix],
            linewidth=0.5, marker="o", markersize=2, markeredgewidth=0,
        )
        plt.fill_between(
            x_plot, mean_plot - std_plot, mean_plot + std_plot,
            alpha=0.2, color=MAPPED_COLOR[prefix], linewidth=0.2,
        )
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(DATASET_NAME)
    plt.legend()
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{DATASET_NAME}.svg", format="svg")
    plt.savefig(FIG_DIR / f"{DATASET_NAME}.png", dpi=130)
    print(f"saved {FIG_DIR / (DATASET_NAME + '.svg')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tidigits noise-evaluation curves.")
    parser.add_argument("--smooth-mode", choices=sorted(SMOOTH_MODES), default="max")
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_means_and_variances(args.smooth_mode, args.smooth_window)


if __name__ == "__main__":
    main()
