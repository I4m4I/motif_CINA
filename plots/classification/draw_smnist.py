#!/usr/bin/env python3
"""Reproduce figures/smnist.svg (the original smnist_line_ANN_v2.svg).

Reads the noise-sweep curves in ../data/smnist/<prefix>_<seed>.npy. Each file
is a 2-D array whose column 0 is the Gaussian-noise variance (x-axis) and
column 1 is the continuous-path accuracy (y-axis). Curves are averaged over the
first 5 seeds and resampled to 11 points; the shaded band is +/- std/2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATASET_NAME = "smnist"
_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = _ROOT / "data" / "classification" / DATASET_NAME
FIG_DIR = _ROOT / "figures" / "classification"

# File prefix == legend label (names normalized to the on-figure method names).
PREFIXES = ["MOP", "AVE", "Vanilla", "FRP", "FRP-E"]
MAPPED_PREFIXES = {
    "MOP": "MOP",
    "AVE": "AVE",
    "FRP": "FRP",
    "FRP-E": "FRP-E",
    "Vanilla": "Vanilla",
}
MAPPED_COLOR = {
    "MOP": "#F3CC4F",
    "AVE": "#009944",
    "FRP": "#F18D00",
    "FRP-E": "#E60012",
    "Vanilla": "#529DCB",
}
NUMBER_OF_SEEDS = 5
RUN_IDS = list(range(NUMBER_OF_SEEDS))
SMOOTH_MODES = {"none", "max", "mean"}


def load_xy(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load x (column 0) and stack y (column 1) into shape (num_runs, length)."""
    ys: list[np.ndarray] = []
    x_ref: np.ndarray | None = None
    target_length: int | None = None
    for run_id in RUN_IDS:
        path = BASE_DIR / f"{prefix}_{run_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        arr2d = np.load(path)
        if arr2d.ndim != 2 or arr2d.shape[1] < 2:
            raise ValueError(f"Expected 2D array with >=2 columns in {path}, got {arr2d.shape}")
        x, y = arr2d[:, 0], arr2d[:, 1]
        if x_ref is None:
            x_ref = x
        elif x.shape != x_ref.shape or not np.allclose(x, x_ref):
            raise ValueError(f"Inconsistent x values for {path}")
        if target_length is None:
            target_length = y.shape[0]
        elif y.shape[0] != target_length:
            raise ValueError(f"Inconsistent length for {path}: {y.shape[0]} vs {target_length}")
        ys.append(y)
    if not ys:
        raise RuntimeError(f"No runs loaded for prefix {prefix}")
    return x_ref, np.vstack(ys)


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
        x, stacked = load_xy(prefix)
        smoothed = apply_smoothing(stacked, smooth_mode, smooth_window)
        mean_vals = smoothed.mean(axis=0)
        std_vals = smoothed.std(axis=0) / 2
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
            alpha=0.2, color=MAPPED_COLOR[prefix],
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
    parser = argparse.ArgumentParser(description="Plot smnist noise-evaluation curves.")
    parser.add_argument("--smooth-mode", choices=sorted(SMOOTH_MODES), default="max")
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_means_and_variances(args.smooth_mode, args.smooth_window)


if __name__ == "__main__":
    main()
