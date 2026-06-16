#!/usr/bin/env python3
"""Reproduce figures/ant.svg from data/ant/*.npy (Ant-v2)."""

from __future__ import annotations

from _common import parse_args, plot_means_and_variances

DATASET_NAME = "ant"
NUMBER_OF_SEEDS = 8
PREFIXES = ["FRP", "AVE", "MOP", "MOP_E", "Vanilla"]
MAPPED_PREFIXES = {
    "FRP": "FRP",
    "AVE": "AVE",
    "MOP": "MOP",
    "MOP_E": "MOP_E",
    "Vanilla": "Vanilla",
}
MAPPED_COLOR = {
    "MOP_E": "#601986",
    "MOP": "#F3CC4F",
    "FRP": "#F18D00",
    "AVE": "#009944",
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
        use_sem=False,
        linewidth=0.5,
        band_linewidth=0.0,
        ylim=(0, 3500),
    )


if __name__ == "__main__":
    main()
