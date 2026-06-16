#!/usr/bin/env python3
"""Reproduce figures/ip.svg from data/ip/*.npy (InvertedPendulum-v2)."""

from __future__ import annotations

from _common import parse_args, plot_means_and_variances

DATASET_NAME = "ip"
NUMBER_OF_SEEDS = 10
PREFIXES = ["FRP", "AVE", "MOP", "MOP-E", "Vanilla"]
MAPPED_PREFIXES = {
    "FRP": "FRP",
    "AVE": "AVE",
    "MOP": "MOP",
    "MOP-E": "MOP-E",
    "Vanilla": "Vanilla",
}
MAPPED_COLOR = {
    "FRP": "#F18D00",
    "AVE": "#009944",
    "MOP": "#F3CC4F",
    "MOP-E": "#601986",
    "Vanilla": "#529DCB",
}


def main() -> None:
    args = parse_args()
    plot_means_and_variances(
        dataset_name=DATASET_NAME,
        prefixes=PREFIXES,
        mapped_prefixes=MAPPED_PREFIXES,
        mapped_color=MAPPED_COLOR,
        number_of_seeds=NUMBER_OF_SEEDS,
        smooth_mode=args.smooth_mode,
        smooth_window=args.smooth_window,
        clip=None,
        use_sem=True,
        linewidth=1.0,
        band_linewidth=1.0,
        ylim=None,
    )


if __name__ == "__main__":
    main()
