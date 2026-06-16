#!/usr/bin/env python3
"""Recreate the NZ comparison charts (FRP_E vs FRP, MOp_E vs MOp) from the CSV.

Same workflow as the original notebook, but the input vectors come from
``data/motif_fre/fig2_region_motif_counts_log10_1p8NZ_5um_thr0.1.csv`` instead
of being hard-coded:

1. Read each region's M01..M13 ``log10(1 + 8*NZ)`` values from the CSV.
2. Invert the display transform to the raw NZ vector:  NZ = (10**x - 1) / 8.
3. Build the enhanced vector (natural-log biasing + L2 renormalization).
4. Re-apply the display transform ``log10(1 + 8*NZ)`` to the enhanced vector.
5. Plot grouped bars (enhanced vs original) and save SVG (+ PNG preview).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = _ROOT / "data" / "motif_fre" / "fig2_region_motif_counts_log10_1p8NZ_5um_thr0.1.csv"
FIG_DIR = _ROOT / "figures" / "motif_fre"

COLORS = {
    "MOp_E": "#601986",
    "MOp": "#F3CC4F",
    "FRP_E": "#E60012",
    "FRP": "#F18D00",
}
LOG_COLS = [f"M{idx:02d}_log10_1p8NZ" for idx in range(1, 14)]


def read_region_transformed(region: str) -> np.ndarray:
    """Return the 13-D log10(1+8*NZ) vector for the first row of `region`."""
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["region"] == region:
                return np.array([float(row[c]) for c in LOG_COLS], dtype=np.float64)
    raise KeyError(f"region {region!r} not found in {CSV_PATH}")


def invert_transform(x: np.ndarray) -> np.ndarray:
    """Invert log10(1 + 8*NZ) back to the raw NZ vector."""
    return (np.power(10.0, x) - 1.0) / 8.0


def make_enhanced_nz(nz: np.ndarray, pos_scale: float = 5.0, neg_scale: float = 3.0) -> np.ndarray:
    enhanced = nz.astype(np.float64).copy()
    pos = nz > 0
    neg = nz < 0
    enhanced[pos] = np.log1p(pos_scale * nz[pos])
    enhanced[neg] = -np.log1p(neg_scale * np.abs(nz[neg]))
    norm = np.linalg.norm(enhanced)
    if norm == 0:
        raise ValueError("Cannot normalize an all-zero vector.")
    return enhanced / norm


def visualization_transform(nz: np.ndarray) -> np.ndarray:
    argument = 1.0 + 8.0 * nz
    if np.any(argument <= 0):
        bad = nz[argument <= 0]
        raise ValueError(f"log10(1 + 8 * NZ) is undefined for values: {bad}")
    return np.log10(argument)


def plot_pair(first, second, first_label, second_label, output_path: Path) -> None:
    index = np.arange(1, 14)
    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(index, first, bar_width, label=first_label, color=COLORS[first_label])
    ax.bar(index + bar_width, second, bar_width, label=second_label, color=COLORS[second_label])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(range(1, 14))
    ax.set_ylim(-0.2, 1)
    ax.legend()
    ax.set_title("Comparison of MOp, FRP and their Average")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> None:
    argparse.ArgumentParser(description="Recreate NZ comparison charts from CSV.").parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for region in ["FRP", "MOp"]:
        transformed = read_region_transformed(region)       # = log10(1+8*NZ), the "original"
        nz = invert_transform(transformed)
        enhanced_transformed = visualization_transform(make_enhanced_nz(nz))
        plot_pair(
            enhanced_transformed, transformed,
            f"{region}_E", region,
            FIG_DIR / f"comparison_{region}_chart.svg",
        )


if __name__ == "__main__":
    main()
