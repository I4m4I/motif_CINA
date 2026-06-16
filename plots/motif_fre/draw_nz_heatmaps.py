#!/usr/bin/env python3
"""Recreate the Fig.2 NZ-score heatmap and region cosine-similarity matrix.

Reads ``../../data/motif_fre/fig2_region_NZ_matrix.csv`` — a (50 region × 13
motif) matrix of normalized-z-score (NZ) motif profiles, with regions already
ordered by the manuscript's 4-cluster custom order — and draws:

  - ``figures/motif_fre/nz_score_heatmap.svg``          (original "3.nz_score_heatmap")
  - ``figures/motif_fre/cosine_similarity_matrix.svg``  (original "4.cosine_similarity_matrix_RdBu_r")

The heatmap shows ``log10(1 + 8*NZ)`` (undefined entries clamped low); the
similarity matrix is the cosine similarity between regions over the first 12
motif NZ values.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = _ROOT / "data" / "motif_fre" / "fig2_region_NZ_matrix.csv"
FIG_DIR = _ROOT / "figures" / "motif_fre"

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def load_nz() -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    rows: list[list[float]] = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            names.append(r[0])
            rows.append([float(v) for v in r[1:14]])
    return np.asarray(rows, dtype=float), names


def plot_nz_heatmap(NZ: np.ndarray, names: list[str]) -> None:
    plt.figure(figsize=(9, 8))
    im = plt.imshow(
        np.nan_to_num(np.log10(1 + 8 * NZ), nan=-100.0, posinf=-100.0, neginf=-100.0),
        aspect="auto", cmap="bwr", vmin=-1, vmax=1,
    )
    plt.colorbar(im, label="NZ-score")
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xticks(range(13), [f"M{i + 1}" for i in range(13)], rotation=45)
    plt.title("log10(1+8*NZ) heatmap (custom region order)")
    plt.tight_layout()
    out = FIG_DIR / "nz_score_heatmap.svg"
    plt.savefig(out, bbox_inches="tight", dpi=300, transparent=True)
    plt.savefig(FIG_DIR / "nz_score_heatmap.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"saved {out}")


def plot_cosine_similarity(NZ: np.ndarray, names: list[str]) -> None:
    X = NZ[:, :12]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos_sim = X_norm @ X_norm.T
    plt.figure(figsize=(7, 6))
    im = plt.imshow(cos_sim, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Cosine similarity of NZ-score")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.yticks(range(len(names)), names, fontsize=7)
    plt.title("Cosine similarity matrix (cmap=RdBu_r)")
    plt.tight_layout()
    out = FIG_DIR / "cosine_similarity_matrix.svg"
    plt.savefig(out, bbox_inches="tight", dpi=300, transparent=True)
    plt.savefig(FIG_DIR / "cosine_similarity_matrix.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"saved {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NZ, names = load_nz()
    plot_nz_heatmap(NZ, names)
    plot_cosine_similarity(NZ, names)


if __name__ == "__main__":
    main()
