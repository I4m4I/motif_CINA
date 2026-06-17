#!/usr/bin/env python3
"""Plot the K-S heatmap from precomputed K-S and p values.

This script is for lightweight figure reproduction when the model checkpoint
files are not distributed. It reproduces only the K-S heatmap; use
plot_weight_and_stats.py when recomputing the statistics from model weights.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_ORDER = ["FRP", "MOp", "Average", "Vanilla"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_values(csv_path: Path, order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {label: i for i, label in enumerate(order)}
    ks_matrix = np.full((len(order), len(order)), np.nan, dtype=float)
    p_matrix = np.full((len(order), len(order)), np.nan, dtype=float)
    stars = np.empty((len(order), len(order)), dtype=object)
    stars[:, :] = ""

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            i = idx[row["row_model"]]
            j = idx[row["column_model"]]
            ks_matrix[i, j] = float(row["ks_value"])
            p_matrix[i, j] = float(row["p_value"])
            stars[i, j] = row.get("stars", "")

    if np.isnan(ks_matrix).any() or np.isnan(p_matrix).any():
        raise ValueError(f"Missing entries in {csv_path}")
    return ks_matrix, p_matrix, stars


def plot_heatmap(order: list[str], ks_matrix: np.ndarray, stars: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=300)
    vmax = max(0.01, float(np.nanmax(ks_matrix)))
    sns.heatmap(
        ks_matrix,
        square=True,
        cmap="Reds",
        vmin=0,
        vmax=vmax,
        xticklabels=order,
        yticklabels=order,
        cbar_kws={"label": "K-S test value"},
        ax=ax,
        annot=False,
    )
    for i in range(len(order)):
        for j in range(len(order)):
            value = ks_matrix[i, j]
            suffix = stars[i, j]
            text = f"{value:.2f}" if not suffix else f"{value:.2f}\n{suffix}"
            color = "white" if value > 0.55 * vmax else "black"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=color, fontsize=10)
    ax.tick_params(axis="x", labelrotation=0, labelsize=11)
    ax.tick_params(axis="y", labelrotation=90, labelsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if out_path.suffix.lower() != ".svg":
        fig.savefig(out_path.with_suffix(".svg"), dpi=500)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=Path, default=repo_root() / "data" / "topology" / "fig6g" / "KS_test_heatmap_fig6g.csv")
    parser.add_argument("--output", type=Path, default=repo_root() / "figures" / "topology" / "fig6g" / "KS_test_heatmap_fig6g.png")
    parser.add_argument("--order", nargs="+", default=DEFAULT_ORDER)
    args = parser.parse_args()

    ks_matrix, _p_matrix, stars = load_values(args.values, args.order)
    plot_heatmap(args.order, ks_matrix, stars, args.output)
    print(f"Saved K-S test heatmap to {args.output}")


if __name__ == "__main__":
    main()
