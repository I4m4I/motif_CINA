#!/usr/bin/env python3
"""Plot weight matrices from stored plotting arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def plot_weight_panel(labels: list[str], matrices: list[np.ndarray], output_dir: Path, vlim: float) -> None:
    fig, axes = plt.subplots(1, len(labels), figsize=(15, 4.2), dpi=300, constrained_layout=False)
    if len(labels) == 1:
        axes = [axes]

    last_image = None
    for idx, (ax, label, matrix) in enumerate(zip(axes, labels, matrices)):
        last_image = ax.imshow(matrix, cmap="bwr", vmin=-vlim, vmax=vlim, interpolation="nearest")
        ax.set_title(
            label,
            fontsize=15,
            pad=8,
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "square,pad=0.2"},
        )
        ax.set_xticks([0, matrix.shape[1] // 2, matrix.shape[1] - 1])
        ax.set_xticklabels(["1", "......", str(matrix.shape[1])], fontsize=11)
        ax.set_yticks([0, matrix.shape[0] // 2, matrix.shape[0] - 1])
        if idx == 0:
            ax.set_yticklabels([str(matrix.shape[0]), "......", "1"], fontsize=11)
            ax.set_ylabel("Source neurons", fontsize=14)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Target neurons", fontsize=14)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.16, top=0.82, wspace=0.06)
    if last_image is not None:
        cax = fig.add_axes([0.946, 0.19, 0.012, 0.58])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_ticks([-vlim, vlim])
        cbar.set_ticklabels([f"-{vlim:.1f}", f"+{vlim:.1f}"])
        cbar.set_label("Connectivity weight", fontsize=14)
    fig.savefig(output_dir / "weight_matrices_figS10.svg", dpi=500)
    fig.savefig(output_dir / "weight_matrices_figS10.png", dpi=300)
    plt.close(fig)


def plot_individual_weight_matrices(labels: list[str], matrices: list[np.ndarray], output_dir: Path, vlim: float) -> None:
    for idx, (label, matrix) in enumerate(zip(labels, matrices)):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
        im = ax.imshow(matrix, cmap="bwr", vmin=-vlim, vmax=vlim, interpolation="nearest")
        ax.set_title(label)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "weight_matrices_figS10_individual" / f"{label}.svg", dpi=500)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-data", type=Path, default=repo_root() / "data" / "topology" / "weight_matrices_figS10.npz")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "figures" / "topology")
    parser.add_argument("--vlim", type=float, default=0.4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "weight_matrices_figS10_individual").mkdir(parents=True, exist_ok=True)
    weight_npz = np.load(args.weight_data)
    labels = [str(x) for x in weight_npz["labels"].tolist()]
    matrices = [weight_npz[label] for label in labels]

    plot_individual_weight_matrices(labels, matrices, args.output_dir, args.vlim)
    plot_weight_panel(labels, matrices, args.output_dir, args.vlim)
    print(f"Saved weight matrices figure to {args.output_dir}")


if __name__ == "__main__":
    main()
