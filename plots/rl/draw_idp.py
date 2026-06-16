#!/usr/bin/env python3
"""Reproduce figures/idp.svg from data/idp/*.npy (InvertedDoublePendulum-v2).

Based on the original ``idp_2000sad.svg`` figure. Its data lives in
``data/idp`` as continuous-evaluation reward curves: the four motif methods
(MOP-E/MOP/FRP/AVE) were trained for 1500 iterations while the Vanilla
baseline ran for 2000. To keep the curves comparable, every series is
truncated to the common length 1500 (``clip=1500``) so all five end at the
same x.
"""

from __future__ import annotations

from _common import parse_args, plot_means_and_variances

DATASET_NAME = "idp"
NUMBER_OF_SEEDS = 10
PREFIXES = ["MOP-E", "MOP", "FRP", "AVE", "Vanilla"]
MAPPED_PREFIXES = {
    "MOP-E": "MOP-E",
    "MOP": "MOP",
    "FRP": "FRP",
    "AVE": "AVE",
    "Vanilla": "Vanilla",
}
MAPPED_COLOR = {
    "MOP-E": "#601986",
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
        suffix="",
        clip=1500,  # truncate all curves to the common length (Vanilla/AVE runs to 2000)
        use_sem=True,
        linewidth=0.5,
        band_linewidth=0.0,
        ylim=None,
    )


if __name__ == "__main__":
    main()
