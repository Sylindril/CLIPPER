"""Figure 2 (all-baselines.png): SLIPPER among the discrete rule and tree learners.

One stripe per benchmark, a sub-column per training size; within a
sub-column every seed is a faint point and each learner's mean a solid
diamond. Disjoint benchmarks share columns (stroke/heloc, bank/water), and a
last band averages each seed over the seven datasets other than College.
Jitter uses a seeded RNG over the rows in their stored order, so the figure
is reproducible.
"""
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FIGURES, RESULTS
from .style import SLIPPER_FULL

SLIPPER = "slipper"
READABLE = ["ripper", "greedy_rule_list", "skope_rules", "fpskope",
            "imodels_slipper", "tao_tree"]
CORELS = "corels"
ENSEMBLES = ["rulefit", "boosted_rules", "hstree", "figs"]

PURPLE = SLIPPER_FULL          # SLIPPER
LIGHT_BLUE = "#8A98A7"         # other rule/tree learners: neutral slate
GREY = "#c9c9c9"               # rule/tree ensembles
GREEN = "#2c9e56"              # CORELS
DATASETS = ["stroke", "bank_marketing", "adult", "college", "telco_churn",
            "compas", "heloc", "water_potability"]
ABBREV = {"bank_marketing": "bank", "telco_churn": "telco",
          "water_potability": "water", "macro": "average (-college)"}
PAIRS = [("stroke", "heloc"), ("bank_marketing", "water_potability")]
MACRO = "macro"
MACRO_DROP = {"college"}       # College stops at n=600
MACRO_BG = "#EEF5FB"
MACRO_RULE = "#000000"
SIZES = [100, 250, 600, 1200, 2000]


def load(path, labels, sizes):
    out = collections.defaultdict(list)
    by_seed = collections.defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["label"] in labels and int(r["size"]) in sizes and r["err"]:
                key = (r["dataset"], int(r["size"]), r["label"])
                out[key].append(float(r["err"]))
                by_seed[(int(r["size"]), r["label"], int(r["seed"]))][r["dataset"]] = float(r["err"])
    return out, by_seed


def add_macro(data, by_seed, datasets, sizes):
    """Per (size, learner, seed): that seed's mean error over the benchmarks."""
    for z in sizes:
        want = [d for d in datasets if d not in MACRO_DROP and (d, z, SLIPPER) in data]
        if not want:
            continue
        for (zz, l, _seed), per in by_seed.items():
            if zz == z and all(d in per for d in want):
                data[(MACRO, z, l)].append(float(np.mean([per[d] for d in want])))


def swarm(ax, values, x, spread, colour, size, alpha, z, rng):
    v = np.asarray(values, dtype=float)
    ax.plot(x + rng.uniform(-spread / 2, spread / 2, size=len(v)), v, "o",
            ms=size, color=colour, alpha=alpha, zorder=z, ls="", markeredgewidth=0)


def mean_marker(ax, values, x, colour, size, z, marker="o"):
    ax.plot([x], [float(np.mean(values))], marker, ms=size, color=colour, zorder=z,
            markeredgecolor="white", markeredgewidth=0.7, ls="")


def build(dpi=160):
    sizes = SIZES
    labels = [SLIPPER, CORELS] + READABLE + ENSEMBLES
    data, by_seed = load(RESULTS / "tabular/symbolic-baselines.csv", set(labels), set(sizes))
    datasets = [d for d in DATASETS if any((d, z, SLIPPER) in data for z in sizes)]

    groups, placed = [], set()
    pair_of = {a: (a, b) for a, b in PAIRS}
    pair_of.update({b: (a, b) for a, b in PAIRS})
    for d in datasets:
        pair = pair_of.get(d)
        if pair and pair[0] in datasets and pair[1] in datasets:
            if pair in placed:
                continue
            placed.add(pair)
            groups.append(list(pair))
        else:
            groups.append([d])
    add_macro(data, by_seed, datasets, sizes)
    groups.append([MACRO])

    col_w, gap = 1.0, 0.8
    x, ticks, ticklabels, bands, cols = 0.0, [], [], [], {}
    for g in groups:
        left = x
        for z in sizes:
            if any((d, z, l) in data for d in g for l in labels):
                for d in g:
                    cols[(d, z)] = x
                ticks.append(x)
                ticklabels.append("%d" % z)
                x += col_w
        bands.append((g, left - col_w / 2, x - col_w / 2))
        x += gap

    fig, ax = plt.subplots(figsize=(0.42 * len(ticks) + 2.2, 6.4))
    for i, (g, left, right) in enumerate(bands):
        if g == [MACRO]:
            ax.axvspan(left, right, color=MACRO_BG, zorder=0)
            ax.axvline(left - gap / 2, color=MACRO_RULE, lw=1.6, zorder=1)
        elif i % 2 == 0:
            ax.axvspan(left, right, color="#f4f6fa", zorder=0)
        mid = (left + right) / 2
        ax.text(mid, 1.012, ABBREV.get(g[-1], g[-1]), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
        if len(g) > 1:
            ax.text(mid, 0.012, ABBREV.get(g[0], g[0]), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    spread = col_w * 0.72
    for (d, z), xx in sorted(cols.items()):
        for group, colour in ((ENSEMBLES, GREY), (READABLE, LIGHT_BLUE)):
            for l in group:
                if (d, z, l) in data:
                    swarm(ax, data[(d, z, l)], xx, spread, colour, 2.2, 0.28, 2, rng)
        if (d, z, CORELS) in data:
            swarm(ax, data[(d, z, CORELS)], xx, spread, GREEN, 2.4, 0.40, 3, rng)
        if (d, z, SLIPPER) in data:
            swarm(ax, data[(d, z, SLIPPER)], xx, spread, PURPLE, 2.6, 0.55, 4, rng)
        for l in ENSEMBLES:
            if (d, z, l) in data:
                mean_marker(ax, data[(d, z, l)], xx, GREY, 5.5, 5, marker="D")
        for l in READABLE:
            if (d, z, l) in data:
                mean_marker(ax, data[(d, z, l)], xx, LIGHT_BLUE, 5.5, 6, marker="D")
        if (d, z, CORELS) in data:
            mean_marker(ax, data[(d, z, CORELS)], xx, GREEN, 6.0, 7, marker="D")
        if (d, z, SLIPPER) in data:
            mean_marker(ax, data[(d, z, SLIPPER)], xx, PURPLE, 7.0, 8, marker="D")

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=12, rotation=90)
    ax.set_ylabel("test error (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 55)
    ax.set_xlim(-0.8, max(cols.values()) + 0.8)
    ax.grid(axis="y", color="white", lw=1.0, zorder=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="D", ms=6, color=PURPLE,
                          label="SLIPPER, mean of 10 seeds",
                          markeredgecolor="white", markeredgewidth=0.7)]
    handles += [plt.Line2D([], [], ls="", marker="D", ms=5.5, color=c, label=t,
                           markeredgecolor="white", markeredgewidth=0.7)
                for c, t in [(GREEN, "CORELS, mean of 10 seeds"),
                             (LIGHT_BLUE, "other rule/tree learners, mean"),
                             (GREY, "rule/tree ensembles, mean")]]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.992, 0.025),
              frameon=True, fontsize=12, facecolor="white", edgecolor="#999999",
              framealpha=1.0, borderpad=0.7, labelspacing=0.55).set_zorder(10)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.115)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "all-baselines.png", dpi=dpi)
    fig.savefig(FIGURES / "all-baselines.pdf")
    plt.close(fig)
