#!/usr/bin/env python3
"""Reproduce figures/dvs.svg from the DVS128-Gesture noise-eval CSV logs.

Reads ../data/dvs/<...freclass<method>_seed<seed>...>/evaluation_results_1.csv,
each holding (noise, accuracy) rows. Curves are grouped by method, averaged over
seeds, and drawn vs the noise axis; the band is +/- std/2.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATASET_NAME = "dvs"
CSV_FILENAME = "evaluation_results_1.csv"
_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_DIR = _ROOT / "data" / "classification" / DATASET_NAME
DEFAULT_OUTPUT = _ROOT / "figures" / "classification" / "dvs.svg"
SMOOTH_MODES = {"none", "max", "mean"}

PREFIXES = ["MOP", "AVE", "Vanilla", "FRP", "FRP-E"]
# Directory method token == prefix == legend label (names normalized).
RAW_METHOD_TO_PREFIX = {
    "MOP": "MOP",
    "AVE": "AVE",
    "Vanilla": "Vanilla",
    "FRP": "FRP",
    "FRP-E": "FRP-E",
}
LABELS = {
    "MOP": "MOP",
    "AVE": "AVE",
    "FRP": "FRP",
    "FRP-E": "FRP-E",
    "Vanilla": "Vanilla",
}
COLORS = {
    "MOP": "#529DCB",
    "AVE": "#009944",
    "FRP": "#F18D00",
    "FRP-E": "#E60012",
    "Vanilla": "#F3CC4F",
}


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


def parse_experiment_name(experiment_dir_name: str) -> tuple[str, int] | None:
    match = re.search(r"freclass(?P<method>[^_]+)_seed(?P<seed>\d+)", experiment_dir_name)
    if not match:
        return None
    raw_method = match.group("method")
    mapped_prefix = RAW_METHOD_TO_PREFIX.get(raw_method)
    if mapped_prefix is None:
        return None
    return mapped_prefix, int(match.group("seed"))


def load_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    noises: list[float] = []
    accs: list[float] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noises.append(float(row["noise"]))
            accs.append(float(row["accuracy"]))
    if not noises:
        raise ValueError(f"Empty CSV: {csv_path}")
    noise_arr = np.asarray(noises, dtype=np.float64)
    acc_arr = np.asarray(accs, dtype=np.float64)
    order = np.argsort(noise_arr)
    return noise_arr[order], acc_arr[order]


def collect_runs(log_dir: Path) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for csv_path in sorted(log_dir.rglob(CSV_FILENAME)):
        parsed = parse_experiment_name(csv_path.parent.name)
        if parsed is None:
            continue
        prefix, seed = parsed
        noises, accs = load_curve(csv_path)
        grouped[prefix].append((seed, noises, accs))
    return grouped


def get_aligned_series(prefix, grouped_runs):
    runs = sorted(grouped_runs.get(prefix, []), key=lambda item: item[0])
    if not runs:
        raise FileNotFoundError(f"No runs found for prefix '{prefix}'")
    ref_noise = runs[0][1]
    stacked: list[np.ndarray] = []
    for seed, noise, acc in runs:
        if noise.shape != ref_noise.shape or not np.allclose(noise, ref_noise):
            raise ValueError(f"Inconsistent noise axis for prefix '{prefix}', seed={seed}")
        stacked.append(acc)
    return ref_noise, np.vstack(stacked)


def plot_dvs(log_dir: Path, output: Path, smooth_mode: str, smooth_window: int, show: bool) -> None:
    grouped_runs = collect_runs(log_dir)
    missing = [prefix for prefix in PREFIXES if prefix not in grouped_runs]
    if missing:
        raise FileNotFoundError(f"Missing method groups: {missing}")

    plt.figure(figsize=(4, 3))
    for prefix in PREFIXES:
        noise_axis, stacked = get_aligned_series(prefix, grouped_runs)
        smoothed = apply_smoothing(stacked, smooth_mode, smooth_window)
        mean_vals = smoothed.mean(axis=0)
        std_vals = smoothed.std(axis=0)
        plt.plot(
            noise_axis, mean_vals,
            label=LABELS[prefix], color=COLORS[prefix],
            linewidth=0.5, marker="o", markersize=2, markeredgewidth=0,
        )
        plt.fill_between(
            noise_axis, mean_vals - std_vals / 2, mean_vals + std_vals / 2,
            alpha=0.2, color=COLORS[prefix], linewidth=0.2,
        )

    plt.xlabel("Noise")
    plt.ylabel("Accuracy")
    plt.title(DATASET_NAME)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="svg")
    plt.savefig(output.with_suffix(".png"), dpi=130)
    print(f"saved {output}")
    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DVS noise-evaluation curves from CSV logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smooth-mode", choices=sorted(SMOOTH_MODES), default="none")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dvs(args.log_dir, args.output, args.smooth_mode, args.smooth_window, args.show)


if __name__ == "__main__":
    main()
