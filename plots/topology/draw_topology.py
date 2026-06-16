#!/usr/bin/env python3
"""Recreate the Fig.6 modularity (Q) and small-worldness charts.

Reads ``../../data/topology/{Q_Qer_best.jsonl, swER_all.jsonl}``, picks one
connectivity threshold per network, aggregates mean ± SEM across model
instances, and draws:

  - ``figures/topology/modularity.svg`` — modularity Q vs an ER null model
  - ``figures/topology/smallworld.svg`` — clustering C, path length L, sigma

Paired / Welch / one-sample t-tests are printed to stdout.

Note: the **Vanilla** bars are placeholder values — the original notebook used
Q = 0 and C/L/sigma = 1 for Vanilla (no real Vanilla data was available).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = _ROOT / "data" / "topology"
FIG = _ROOT / "figures" / "topology"

COLORS = {
    "FRP": "#d79036",
    "MOp": "#d8bf5a",
    "Average": "#4a9650",
    "Vanilla": "#8fbad3",
    "ER": "#b3b3b3",
}
NET_ORDER = ["FRP", "MOp", "Average"]


# ---------------------------------------------------------------- stats helpers
def p_to_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_ttest_report(name, a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"name": name, "n": len(a), "t": np.nan, "p": np.nan, "star": "NA"}
    t, p = stats.ttest_rel(a, b)
    return {"name": name, "n": len(a), "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)), "t": float(t), "p": float(p), "star": p_to_star(p)}


def welch_ttest_report(name, a, b):
    a = np.asarray(a, dtype=float)[np.isfinite(a)]
    b = np.asarray(b, dtype=float)[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"name": name, "n1": len(a), "n2": len(b), "t": np.nan, "p": np.nan, "star": "NA"}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"name": name, "n1": len(a), "n2": len(b), "mean_1": float(np.mean(a)),
            "mean_2": float(np.mean(b)), "t": float(t), "p": float(p), "star": p_to_star(p)}


def onesample_ttest_report(name, a, popmean=1.0):
    a = np.asarray(a, dtype=float)[np.isfinite(a)]
    if len(a) < 2:
        return {"name": name, "n": len(a), "t": np.nan, "p": np.nan, "star": "NA"}
    t, p = stats.ttest_1samp(a, popmean=popmean)
    return {"name": name, "n": len(a), "mean": float(np.mean(a)), "mu0": float(popmean),
            "t": float(t), "p": float(p), "star": p_to_star(p)}


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n))


def _save(fig, name):
    out = FIG / f"{name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", transparent=True)
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------- small-worldness
def plot_smallworld():
    chosen_p = {"FRP": 0.2, "MOp": 0.2, "Average": 0.2}
    records_by_net = {k: [] for k in NET_ORDER}
    with open(DATA / "swER_all.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            net = rec["net_type"]
            if net in chosen_p and abs(float(rec["p_thre"]) - chosen_p[net]) < 1e-9:
                records_by_net[net].append(rec)

    C_mean, C_sem, C_er_mean, C_er_sem = [], [], [], []
    L_mean, L_sem, L_er_mean, L_er_sem = [], [], [], []
    S_mean, S_sem = [], []
    for net in NET_ORDER:
        recs = records_by_net[net]
        for src, dst_m, dst_s in [
            ([r["sw_er"]["obs"]["C"] for r in recs], C_mean, C_sem),
            ([r["sw_er"]["rand"]["C_mean"] for r in recs], C_er_mean, C_er_sem),
            ([r["sw_er"]["obs"]["L"] for r in recs], L_mean, L_sem),
            ([r["sw_er"]["rand"]["L_mean"] for r in recs], L_er_mean, L_er_sem),
            ([r["sw_er"]["ratio"]["sigma_CL"] for r in recs], S_mean, S_sem),
        ]:
            m, s = mean_sem(src)
            dst_m.append(m)
            dst_s.append(s)

    # Vanilla placeholders (no real data; clustering/path/sigma == 1)
    van_C = van_L = van_sigma = 1.0
    van_sem = 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    group_labels = ["FRP", "MOp", "Average", "Vanilla"]
    group_colors = [COLORS[g] for g in group_labels]
    x = np.arange(len(group_labels))
    barw = 0.34
    xt = ["FRP\nER$_{FRP}$", "MOp\nER$_{MOp}$", "Average\nER$_{Average}$", "Vanilla\nER$_{Vanilla}$"]

    for ax, obs_m, obs_s, er_m, er_s, ylab in [
        (axes[0], C_mean, C_sem, C_er_mean, C_er_sem, "Clustering coefficient"),
        (axes[1], L_mean, L_sem, L_er_mean, L_er_sem, "Average path length"),
    ]:
        obs_all = [obs_m[0], obs_m[1], obs_m[2], (van_C if ylab[0] == "C" else van_L)]
        obs_sem_all = [obs_s[0], obs_s[1], obs_s[2], van_sem]
        er_all = [er_m[0], er_m[1], er_m[2], (van_C if ylab[0] == "C" else van_L)]
        er_sem_all = [er_s[0], er_s[1], er_s[2], van_sem]
        ax.bar(x - barw / 2, obs_all, barw, yerr=obs_sem_all, capsize=3, color=group_colors, edgecolor="none")
        ax.bar(x + barw / 2, er_all, barw, yerr=er_sem_all, capsize=3, color=COLORS["ER"], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=-25)
        ax.set_ylabel(ylab)

    sigma_vals = [S_mean[0], S_mean[1], S_mean[2], van_sigma]
    sigma_errs = [S_sem[0], S_sem[1], S_sem[2], van_sem]
    axes[2].bar(np.arange(4), sigma_vals, 0.58, yerr=sigma_errs, capsize=3,
                color=[COLORS[k] for k in group_labels], edgecolor="none")
    axes[2].set_xticks(np.arange(4))
    axes[2].set_xticklabels(group_labels, rotation=-25)
    axes[2].set_ylabel("Small-world coefficient")

    for ax, lab in zip(axes, ["d", "e", "f"]):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=1.2)
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    fig.tight_layout()
    _save(fig, "smallworld")

    print("\n===== Paired t-test: observed vs ER (small-world) =====")
    for net in NET_ORDER:
        recs = records_by_net[net]
        print(paired_ttest_report(f"{net}: C vs ER",
              [r["sw_er"]["obs"]["C"] for r in recs], [r["sw_er"]["rand"]["C_mean"] for r in recs]))
        print(paired_ttest_report(f"{net}: L vs ER",
              [r["sw_er"]["obs"]["L"] for r in recs], [r["sw_er"]["rand"]["L_mean"] for r in recs]))
    print("\n===== Small-world coefficient tests =====")
    sigma_by_net = {net: np.array([r["sw_er"]["ratio"]["sigma_CL"] for r in records_by_net[net]], dtype=float)
                    for net in NET_ORDER}
    for a, b in combinations(NET_ORDER, 2):
        print(welch_ttest_report(f"sigma: {a} vs {b}", sigma_by_net[a], sigma_by_net[b]))
    for net in NET_ORDER:
        print(onesample_ttest_report(f"sigma: {net} vs Vanilla(=1)", sigma_by_net[net], popmean=1.0))


# ---------------------------------------------------------------- modularity
def plot_modularity():
    chosen_p = {"FRP": 0.23, "MOp": 0.13, "Average": 0.21}
    q_dict = {}
    with open(DATA / "Q_Qer_best.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if abs(float(rec.get("gamma", 1.0)) - 1.0) > 1e-9:
                continue
            key = (rec["net_type"], int(rec["n"]), round(float(rec["p_thre"]), 6))
            q_dict[key] = rec

    records_by_net = {k: [] for k in NET_ORDER}
    for net in NET_ORDER:
        p = round(float(chosen_p[net]), 6)
        ns = sorted({k[1] for k in q_dict if k[0] == net and abs(k[2] - p) < 1e-9})
        for n in ns:
            if (net, n, p) in q_dict:
                records_by_net[net].append(q_dict[(net, n, p)])

    Q_mean, Q_sem, Q_er_mean, Q_er_sem = [], [], [], []
    for net in NET_ORDER:
        recs = records_by_net[net]
        m, s = mean_sem([r["Q_best"] for r in recs]); Q_mean.append(m); Q_sem.append(s)
        m, s = mean_sem([r["Q_ER_mean"] for r in recs]); Q_er_mean.append(m); Q_er_sem.append(s)

    van_q = 0.0
    van_sem = 1e-4
    group_labels = ["FRP", "MOp", "Average", "Vanilla"]
    x = np.arange(len(group_labels))
    barw = 0.34
    Q_obs_all = [Q_mean[0], Q_mean[1], Q_mean[2], van_q]
    Q_obs_sem_all = [Q_sem[0], Q_sem[1], Q_sem[2], van_sem]
    Q_er_all = [Q_er_mean[0], Q_er_mean[1], Q_er_mean[2], van_q]
    Q_er_sem_all = [Q_er_sem[0], Q_er_sem[1], Q_er_sem[2], van_sem]

    for net in NET_ORDER:
        print(net, "threshold =", chosen_p[net], "n_samples =", len(records_by_net[net]))

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8))
    ax.bar(x - barw / 2, Q_obs_all, barw, yerr=Q_obs_sem_all, capsize=3,
           color=[COLORS[g] for g in group_labels], edgecolor="none", label="Q")
    ax.bar(x + barw / 2, Q_er_all, barw, yerr=Q_er_sem_all, capsize=3,
           color=COLORS["ER"], edgecolor="none", label="ER")
    ax.set_xticks(x)
    ax.set_xticklabels(["FRP\nER$_{FRP}$", "MOp\nER$_{MOp}$", "Average\nER$_{Average}$", "Vanilla\nER$_{Vanilla}$"],
                       rotation=-25)
    ax.set_ylabel("Modularity (Q)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2)
    fig.tight_layout()
    _save(fig, "modularity")

    print("\n===== Paired t-test: Q vs ER =====")
    for net in NET_ORDER:
        recs = records_by_net[net]
        print(paired_ttest_report(f"{net}: Q vs ER", [r["Q_best"] for r in recs], [r["Q_ER_mean"] for r in recs]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_modularity()
    plot_smallworld()


if __name__ == "__main__":
    main()
