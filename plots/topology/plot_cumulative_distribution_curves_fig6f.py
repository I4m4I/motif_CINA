#!/usr/bin/env python3
"""Plot the CDF figure from stored CDF arrays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CDF_COLORS = {
    "FRP": "#f39c12",
    "MOp": "#d4b000",
    "Average": "#2ca02c",
    "Vanilla": "#7ec7ee",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_ks_lookup(csv_path: Path) -> dict[tuple[str, str], tuple[float, str]]:
    if not csv_path.exists():
        return {}
    out: dict[tuple[str, str], tuple[float, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(row["row_model"], row["column_model"])] = (
                float(row["ks_value"]),
                row.get("stars", ""),
            )
    return out


def cdf_at(x_sorted: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.searchsorted(x_sorted, xq, side="right") / len(x_sorted)


def largest_ks_pair(labels: list[str], xs: dict[str, np.ndarray]) -> tuple[str, str, float, float]:
    best = ("", "", -1.0, 0.0)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            lx = xs[left]
            rx = xs[right]
            grid = np.linspace(min(lx.min(), rx.min()), max(lx.max(), rx.max()), 2000)
            diff = np.abs(cdf_at(lx, grid) - cdf_at(rx, grid))
            idx = int(np.argmax(diff))
            value = float(diff[idx])
            if value > best[2]:
                best = (left, right, value, float(grid[idx]))
    return best


def plot_cdf(labels: list[str], cdf_data: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path, ks_csv: Path) -> None:
    xs = {label: cdf_data[label][0] for label in labels}
    left, right, ks_value, x_star = largest_ks_pair(labels, xs)
    ks_lookup = load_ks_lookup(ks_csv)
    ks_value, stars = ks_lookup.get((left, right), (ks_value, ""))

    fig, ax = plt.subplots(figsize=(5.0, 4.1), dpi=300)
    for label in labels:
        x, y = cdf_data[label]
        ax.plot(x, y, label=label, linewidth=1.2, color=CDF_COLORS.get(label), alpha=0.95)

    ax.axvline(x_star, color="black", linewidth=0.8, alpha=0.75)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    ax.text(
        0.98,
        0.08,
        f"{left} vs {right}\nK-S={ks_value:.2f} {stars}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_distribution_curves_fig6f.svg", dpi=500)
    fig.savefig(output_dir / "cumulative_distribution_curves_fig6f.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdf-data", type=Path, default=repo_root() / "data" / "topology" / "cumulative_distribution_curves_fig6f.npz")
    parser.add_argument("--ks-values", type=Path, default=repo_root() / "data" / "topology" / "KS_test_values_for_fig6f_annotation.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "figures" / "topology")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cdf_npz = np.load(args.cdf_data)
    labels = [str(x) for x in cdf_npz["labels"].tolist()]
    cdf_data = {label: (cdf_npz[f"{label}_x"], cdf_npz[f"{label}_y"]) for label in labels}

    plot_cdf(labels, cdf_data, args.output_dir, args.ks_values)
    print(f"Saved cumulative distribution curves figure to {args.output_dir}")


if __name__ == "__main__":
    main()
