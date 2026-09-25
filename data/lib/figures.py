"""Every figure in the paper, redrawn from the per-seed results in data/results.

Outputs go to output/figures/<name>.pdf and .png, using the paper's file names.
Each builder starts from matplotlib's defaults, so figures do not affect each other.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy import stats

from . import figure_baselines
from .common import (EIGHT, FIGURES, RESULTS, SEVEN, load_agents, load_anonymized,
                     load_calculator, load_symbolic_baselines, load_tabular, macro_table, sem)
from .style import (CLIPPER, CLIPPER_HINTS, CLIPPER_TOOLS, CORELS, RULE_BASELINE,
                    RULECHEF, SLIPPER, SLIPPER_FULL)
from .tables import anonymization_pairs

INK, MUTED, GRID = "#26343C", "#64717B", "#E2E6EA"


def save(fig, name, **kwargs):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        options = dict(kwargs)
        if suffix == "pdf":
            options["metadata"] = {"CreationDate": None}
        fig.savefig(FIGURES / f"{name}.{suffix}", **options)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: the learning loop, and three learned rules applied to real inputs
# ---------------------------------------------------------------------------

def learned_rule_cards(card, width, height):
    """(b) Three learned rules, each applied to one input: tabular, text, extraction.

    The input excerpts and rule text are as printed in the figure; the weights
    come from results/figure1/learned-rules.json.
    """
    shown = json.loads((RESULTS / "figure1/learned-rules.json").read_text())["rules"]
    alpha = {r["panel"]: r["alpha"] for r in shown}
    ink, muted, positive = "#1F2933", "#5B6570", "#2E7D5B"
    mark, box_fill, stroke = "#FCE8A6", "#F5F7F9", "#C9CFD5"
    renderer = card.figure.canvas.get_renderer()
    inches = card.transData.inverted()

    card.add_patch(FancyBboxPatch((.01, .01), width - .02, height - .02,
                                  boxstyle="round,pad=0,rounding_size=.04",
                                  facecolor="white", edgecolor="#B9C0C7", linewidth=.7))
    card.text(.12, height - .17, "(b) Learned rules at work", fontsize=9, va="center", color=ink)
    card.text(2.04, height - .175,
              "Each rule is code that runs without an LLM; a matching rule adds its weight.",
              fontsize=7.2, color=muted, va="center")

    def segments(x, y, parts, size=6.5, mono=False):
        # Draw text pieces left to right; a piece marked True is highlighted.
        for text, hot in parts:
            t = card.text(x, y, text, fontsize=size, va="center", color=ink,
                          family="DejaVu Sans Mono" if mono else "DejaVu Sans",
                          bbox=dict(boxstyle="square,pad=.05", facecolor=mark,
                                    edgecolor="none") if hot else None)
            box = t.get_window_extent(renderer)
            x = inches.transform((box.x1, box.y0))[0]
        return x

    def rule_text(x, y, parts):
        # Blue monospace text is copied from the learned rule.
        for text, kind in parts:
            mono = kind == "re"
            t = card.text(x, y, text, fontsize=6.3 if mono else 6.6, va="center",
                          family="DejaVu Sans Mono" if mono else "DejaVu Sans",
                          color=CLIPPER if mono else ink)
            box = t.get_window_extent(renderer)
            x = inches.transform((box.x1, box.y0))[0]
        return x

    def down(x, y0, y1):
        card.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                       mutation_scale=7, linewidth=1, color="#5B6570"))

    gap, pad = .1, .12
    col = (width - 2 * pad - 2 * gap) / 3
    top = height - .38
    columns = [
        dict(title="Tabular classification", task="Adult: income > $50K",
             input=[[("married", True), (" husband, ", False), ("age 40", True), (",", False)],
                    [("education-num 13", True), (", ", False), ("45 hours/week", True)]],
             mono=False,
             rule=[[("married ∧ education-num ≥ 13", "txt")],
                   [("∧ hours ≥ 40 ∧ age ≥ 30", "txt")]],
             out=f"adds {alpha['Tabular classification']:+.2f} toward >$50K"),
        dict(title="Text classification", task="MedCalc: which calculator?",
             input=[[("…blood urea nitrogen 24.8 mg/dL,", False)],
                    [("serum ", False), ("creatinine 4.1 mg/dL", True), (", sodium…", False)]],
             mono=False,
             rule=[[(r"creatinine…(\d+\.\d+)\s*mg/dL", "re")],
                   [("matches, and the value ≥ 2.5", "txt")]],
             out=f"adds {alpha['Text classification']:+.2f} toward MDRD GFR"),
        dict(title="Value extraction", task="MedCalc: bicarbonate",
             input=[[("Cl-: 102 mEq/L", False)], [("HCO3-: 12 mEq/L", True)],
                    [("BUN: 15 mg/dL", False)]],
             mono=True,
             rule=[[(r"HCO3[^\d.]*([\d.]+)\s*(mEq/L|mmol/L)", "re")],
                   [("returns the number and unit", "txt")]],
             out="returns [12, mEq/L]"),
    ]
    for k, c in enumerate(columns):
        x = pad + k * (col + gap)
        card.text(x, top, c["title"], fontsize=7.4, color=ink, va="center")
        card.text(x, top - .14, c["task"], fontsize=6.4, color=muted, va="center")
        # Input excerpt with the matched text highlighted.
        in_top, line = top - .27, .115
        in_h = line * 3 + .08
        card.add_patch(FancyBboxPatch((x, in_top - in_h), col, in_h,
                                      boxstyle="round,pad=0,rounding_size=.03",
                                      facecolor=box_fill, edgecolor=stroke, linewidth=.5))
        n = len(c["input"])
        y0 = in_top - in_h / 2 + (n - 1) * line / 2
        for j, parts in enumerate(c["input"]):
            segments(x + .07, y0 - j * line, parts, mono=c["mono"])
        down(x + col / 2, in_top - in_h - .02, in_top - in_h - .14)
        ry = in_top - in_h - .25
        for j, parts in enumerate(c["rule"]):
            rule_text(x + .02, ry - j * .12, parts)
        down(x + col / 2, ry - .2, ry - .32)
        oy = ry - .43
        card.text(x + col / 2, oy, c["out"], fontsize=6.8, ha="center", va="center",
                  color=positive, bbox=dict(boxstyle="round,pad=.25", facecolor="#EAF3EE",
                                            edgecolor="none"))


def figure1_overview():
    """tabular-main: (a) the shared training loop, (b) learned rules at work."""
    plt.rcdefaults()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         # Unhinted glyph advances keep PNG and PDF text widths equal.
                         "text.hinting": "none", "text.hinting_factor": 1}):
        card_height = 1.95
        width, height = 6.65, card_height + 1.88
        fig = plt.figure(figsize=(width, height))
        diagram = fig.add_axes([.1 / width, (card_height + .16) / height,
                                (width - .2) / width, 1.65 / height])
        diagram.set(xlim=(0, width - .2), ylim=(0, 1.65))
        diagram.set_axis_off()
        ink, stroke = "#1F2933", "#9AA1A8"
        muted = "#5B6570"
        accent, accent_fill = CLIPPER, "#E4EEF6"
        diagram.text(.01, 1.51, "(a) Learn one rule at a time", fontsize=9, va="bottom", color=ink)

        def box(x, box_width, lines, accent_box=False):
            diagram.add_patch(FancyBboxPatch(
                (x, .42), box_width, .63, boxstyle="round,pad=0,rounding_size=.035",
                facecolor=accent_fill if accent_box else "#F5F7F9",
                edgecolor=accent if accent_box else stroke,
                linewidth=.9 if accent_box else .7))
            for label, y, size in lines:
                diagram.text(x + box_width / 2, y, label, ha="center", va="center",
                             fontsize=size, color=ink, linespacing=1.25)

        def arrow(points, color="#5B6570", dashed=False):
            diagram.add_patch(FancyArrowPatch(
                path=PlotPath(points), arrowstyle="-|>", mutation_scale=7, linewidth=.9,
                color=color, linestyle=(0, (4, 2.5)) if dashed else "-", clip_on=False))

        box(.01, .82, [("Training data", .85, 7.5), (r"$\{(x_i,y_i)\}_{i=1}^{n}$", .62, 8.3)])
        box(1.01, 1.40, [(r"Select all / up to $M$", .82, 7.4),
                         (r"examples with weights $D_t$", .63, 7.2)])
        box(2.57, 1.33, [(r"Propose rule $r_t$", .89, 8), ("CLIPPER: coding LLM", .70, 7.1),
                         ("SLIPPER: discrete search", .53, 6.8)], accent_box=True)
        box(4.09, 1.17, [(r"Score on all $n$", .89, 7.8), (r"accept $(r_t,\alpha_t)$", .70, 7.2),
                         (r"or update default ($b$)", .53, 7.2)])
        box(5.44, 1.0, [("Ensemble rules", .87, 7.4), (r"$b+\sum_t\alpha_t r_t$", .62, 7.1)])
        for start, end in [(.84, 1.0), (2.42, 2.56), (3.91, 4.08), (5.27, 5.43)]:
            arrow([(start, .735), (end, .735)])
        arrow([(.42, 1.06), (.42, 1.39), (4.675, 1.39), (4.675, 1.06)])
        diagram.text(2.55, 1.41, "All training examples", ha="center", va="bottom",
                     fontsize=7.2, color=muted)
        arrow([(.84, .93), (.92, .93), (.92, 1.18), (3.235, 1.18), (3.235, 1.06)],
              color=accent, dashed=True)
        diagram.text(2.1, 1.20, "Condition hints (optional)", ha="center", va="bottom",
                     fontsize=7, color=accent)
        arrow([(4.675, .41), (4.675, .15), (1.71, .15), (1.71, .41)])
        diagram.text(3.19, .15, r"Reweight examples: $D_t\,\rightarrow\,D_{t+1}$",
                     ha="center", va="center", fontsize=7.5, color=muted,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1))
        arrow([(5.94, .41), (5.94, -.07)])
        diagram.text(5.86, .16, "Outputs the\nlearned classifier", ha="right", va="center",
                     fontsize=6.8, color=muted, linespacing=1.25)

        card = fig.add_axes([.1 / width, .08 / height, (width - .2) / width,
                             card_height / height])
        card.set(xlim=(0, width - .2), ylim=(0, card_height))
        card.set_axis_off()
        learned_rule_cards(card, width - .2, card_height)
        save(fig, "tabular-main", dpi=220, bbox_inches="tight", pad_inches=.035)


# ---------------------------------------------------------------------------
# Figure 3: tabular results. (a) equal examples, (b) condition hints, (c) rule learners
# ---------------------------------------------------------------------------

CURVE_SIZES = [100, 175, 250, 400, 500, 600, 800, 1200, 2000]
RULE_SIZES = [100, 250, 600, 1200, 2000]
TITLE_SIZE, NOTE_SIZE, LEGEND_SIZE = 8, 7, 6.6
FIG_RC = {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
          "xtick.labelsize": 7.3, "ytick.labelsize": 7.5, "text.color": INK,
          "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
          "font.weight": "normal", "axes.titleweight": "normal",
          "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"}
# (legend text, color, marker, linestyle); only full-data SLIPPER is dashed.
# Tools: slopper_notools and candcond_notools run without tools; slopper uses
# the coverage tools.
TABULAR_STYLES = {
    "slopper_notools": ("CLIPPER (no tools)", CLIPPER, "^", "-"),
    "candcond_notools": ("CLIPPER (no tools) + hints", CLIPPER_HINTS, "v", "-"),
    "slopper": ("CLIPPER (tools)", CLIPPER_TOOLS, "<", "-"),
    "slipper_M100": ("SLIPPER, $M=100$", SLIPPER, "s", "-"),
    "slipper": ("SLIPPER, full data", SLIPPER_FULL, "o", "--"),
    "corels": ("CORELS", CORELS, "D", "-"),
    "ripper": ("RIPPER", RULE_BASELINE, "v", "-"),
    "greedy_rule_list": ("Greedy rule list", RULE_BASELINE, ">", "-"),
    "skope_rules": ("SkopeRules", RULE_BASELINE, "P", "-"),
    "fpskope": ("FPSkope", RULE_BASELINE, "X", "-"),
}
PANEL_A = ["slopper_notools", "slipper_M100", "slipper"]
PANEL_B = ["slopper_notools", "candcond_notools", "slipper"]
PANEL_C = ["slopper", "slipper", "slipper_M100", "corels", "ripper",
           "greedy_rule_list", "skope_rules", "fpskope"]
SHORT = {"adult": "Adult", "bank_marketing": "Bank", "college": "College",
         "compas": "COMPAS", "heloc": "HELOC", "stroke": "Stroke", "telco_churn": "Telco",
         "water_potability": "Water", "average": "Average"}


def curve_frame(tab, labels):
    """Macro error on the seven datasets at the nine learning-curve sizes."""
    mm = macro_table(tab[tab["size"].isin(CURVE_SIZES)], labels)
    frame = pd.concat([mm[mm.label == l].sort_values("size") for l in labels])
    assert len(frame) == 9 * len(labels)
    return frame.rename(columns={"mean": "mean_error"})


def rule_learner_summary():
    """Macro error on the seven datasets; RIPPER comes from Figure 2's baseline runs."""
    tab = load_tabular()
    ripper = load_symbolic_baselines()
    frame = pd.concat([tab[tab.label.isin(PANEL_C)], ripper[ripper.label == "ripper"]])
    out = macro_table(frame[frame["size"].isin(RULE_SIZES)], PANEL_C)
    assert len(out) == len(PANEL_C) * len(RULE_SIZES)
    return out.rename(columns={"mean": "mean_error"})


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#BAC3CB")
        ax.spines[side].set_linewidth(.6)
    ax.tick_params(length=2.5, width=.6, pad=3)


def draw_lines(ax, frame, labels, styles):
    """Mean error with SEM bars; frame has label, size, mean_error, sem (percent)."""
    handles = {}
    for label in labels:
        name, color, marker, line = styles[label]
        group = frame[frame.label == label].sort_values("size")
        baseline = color == RULE_BASELINE
        handles[label] = ax.errorbar(
            group["size"], group.mean_error, yerr=group["sem"], label=name, color=color,
            marker=marker, linestyle=line, markersize=3.1, linewidth=.95 if baseline else 1.35,
            markeredgewidth=.65, elinewidth=.65, capsize=1.6,
            zorder=2 if baseline or label == "slipper" else 3)
    style_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    return handles


def panel_titles(fig, entries):
    for x, title, note in entries:
        fig.text(x, .972, title, fontsize=TITLE_SIZE, ha="left", va="top")
        fig.text(x, .918, note, fontsize=NOTE_SIZE, color=MUTED, ha="left", va="top")


def figure3_tabular():
    """figure3-tabular."""
    tab = load_tabular()
    equal = curve_frame(tab, PANEL_A)
    hints = curve_frame(tab, PANEL_B)
    learners = rule_learner_summary()
    plt.rcdefaults()
    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(5.5, 3.0))
        axes_y, axes_h = .35, .52
        ax_a = fig.add_axes([.075, axes_y, .255, axes_h])
        ax_b = fig.add_axes([.365, axes_y, .255, axes_h], sharey=ax_a)
        ax_c = fig.add_axes([.73, axes_y, .255, axes_h])
        panel_titles(fig, [(.075, "(a) Equal examples", "$M=100$ per proposal"),
                           (.365, "(b) Condition hints", "Sample plus full-data hints"),
                           (.73, "(c) Rule learners", "Seven datasets")])
        handles = draw_lines(ax_a, equal, PANEL_A, TABULAR_STYLES)
        handles.update(draw_lines(ax_b, hints, PANEL_B, TABULAR_STYLES))
        handles.update(draw_lines(ax_c, learners, PANEL_C, TABULAR_STYLES))
        for ax in (ax_a, ax_b, ax_c):
            ax.set_xscale("log")
            ax.set_xlim(88, 2260)
            ax.set_xticks([100, 250, 600, 1200, 2000], ["100", "250", "600", "1.2k", "2k"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("Training examples", labelpad=3)
        ax_a.set_ylim(20.8, 27)
        ax_a.set_yticks(range(21, 28))
        ax_a.set_ylabel("Macro test error (%)", labelpad=3)
        ax_b.tick_params(axis="y", labelleft=False)
        ax_c.set_ylim(20.7, 29.3)
        ax_c.set_yticks([22, 24, 26, 28])
        order = ["slopper_notools", "candcond_notools", "slopper", "slipper_M100",
                 "slipper", "corels", "ripper", "greedy_rule_list", "skope_rules", "fpskope"]
        fig.legend([handles[l] for l in order], [TABULAR_STYLES[l][0] for l in order],
                   loc="upper left", bbox_to_anchor=(.005, .165), ncol=4, frameon=False,
                   handlelength=1.4, handletextpad=.4, columnspacing=1.0, labelspacing=.4,
                   borderaxespad=0, fontsize=LEGEND_SIZE)
        save(fig, "figure3-tabular", dpi=300)


# ---------------------------------------------------------------------------
# Figure 4: (a) MedCalc calculators, (b) text classification, (c) without feature names
# ---------------------------------------------------------------------------

CAPS = [1, 4, 16, 64]
# Calculators with fewer training notes than the cap contribute all they have,
# so the realized training sizes are 55, 220, 870, and 3,302 notes; the axis
# plots each at its nearest multiple of 55.
PLOTTED_SIZE = {1: 55, 4: 220, 16: 880, 64: 3300}
MEDCALC_STYLES = {
    "slopper_raw_M20": ("CLIPPER (tools), $M=20$", CLIPPER, "^", "-"),
    "slipper_M20": ("SLIPPER, $M=20$", SLIPPER, "s", "-"),
    "slipper_full": ("SLIPPER, full data", SLIPPER_FULL, "o", "--"),
}
# Text: both CLIPPER lines use no tools and M=100; `slopper_text` reads the raw
# text (regular expressions allowed), `slopper` the set of words SLIPPER sees.
TEXT_DATASETS = ["sms_spam", "sentiment_amazon", "sentiment_imdb", "sentiment_yelp",
                 "youtube_spam"]
TEXT_SIZES = [50, 100, 200, 300, 400, 500, 750]
TEXT_STYLES = {
    "slopper_text": ("CLIPPER, raw text (no tools)", CLIPPER, "^", "-"),
    "slopper": ("CLIPPER, word sets (no tools)", CLIPPER_TOOLS, "v", "-"),
    "rulechef": ("RuleChef", RULECHEF, "s", "-"),
    "slipper": ("SLIPPER, full data", SLIPPER_FULL, "o", "--"),
    "hstree": ("HS-Tree", RULE_BASELINE, "D", "-"),
}


def load_text():
    """Per-seed results on the five text datasets, sizes 50-750. RuleChef rows also
    carry rules (final rule-set size) and coverage (% of test documents matched)."""
    rulechef = pd.read_csv(RESULTS / "text/rulechef-runs.csv")
    frame = pd.concat([pd.read_csv(RESULTS / "text/text-runs.csv"),
                       rulechef[["label", "dataset", "size", "seed", "err", "rules", "coverage"]]],
                      ignore_index=True)
    return frame[frame["size"].isin(TEXT_SIZES)]


def text_summary(labels=None):
    """Macro error over the five text datasets, per learner and training size."""
    labels = labels or list(TEXT_STYLES)
    out = macro_table(load_text(), labels, datasets=TEXT_DATASETS)
    assert len(out) == len(labels) * len(TEXT_SIZES)
    return out.rename(columns={"mean": "mean_error"})


def calculator_error_summary(split="final"):
    """100 − exact 55-way accuracy, mean and SEM over ten seeds."""
    frame = load_calculator()
    frame = frame[frame.split == split].assign(error=lambda f: 100 - f.accuracy)
    out = frame.groupby(["condition", "cap"]).error.agg(mean_error="mean", sem=sem).reset_index()
    out["size"] = out.cap.map(PLOTTED_SIZE)
    return out.rename(columns={"condition": "label"})


def draw_anonymization(ax, paired):
    """Paired per-seed error changes (anonymized − natural), boxes colored by a
    one-sample t-test against zero (green: significant increase, red: decrease)."""
    green, red, gray, point = "#1BAF7A", "#CC2B2B", "#8C969E", "#888888"
    alpha = .05
    deltas = {d: g.sort_values("seed").delta.to_numpy() for d, g in paired.groupby("dataset")}
    order = sorted(deltas, key=lambda d: np.median(deltas[d]))
    deltas["average"] = paired.groupby("seed").delta.mean().sort_index().to_numpy()
    order = ["average"] + order
    values = [deltas[d] for d in order]
    pvalues = [stats.ttest_1samp(v, 0.0).pvalue for v in values]
    colors = [gray if p >= alpha else green if np.mean(v) > 0 else red
              for v, p in zip(values, pvalues)]
    positions = np.arange(1, len(order) + 1, dtype=float)
    positions[1:] += .4
    boxes = ax.boxplot(values, positions=positions, orientation="horizontal", widths=.6,
                       patch_artist=True, showfliers=False, whis=1.5, zorder=2)
    for box, color in zip(boxes["boxes"], colors):
        box.set(facecolor=to_rgba(color, .20), edgecolor=color, linewidth=.85)
    for key in ["whiskers", "caps", "medians"]:
        for index, artist in enumerate(boxes[key]):
            color = colors[index if key == "medians" else index // 2]
            artist.set(color=color, linewidth=1.3 if key == "medians" else .8)
    rng = np.random.default_rng(0)
    for position, v in zip(positions, values):
        ax.plot(v, position + rng.uniform(-.16, .16, size=len(v)), "o", ms=2.4,
                color=point, alpha=.72, markeredgewidth=0, zorder=3)
    ax.axhline(1.7, color=GRID, lw=.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=.85, ls="--", zorder=1)
    stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < alpha else ""
    ax.set_yticks(positions, [(SHORT[d] + " " + stars(p)).strip() for d, p in zip(order, pvalues)])
    ax.set_ylim(.35, positions[-1] + .65)
    ax.set_xlim(-8.7, 9.0)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("Error change (pp)\n(anonymized − natural)", labelpad=3, linespacing=1.15)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)


def figure4_text():
    """figure4-text."""
    medcalc = calculator_error_summary("final")
    text = text_summary()
    paired = anonymization_pairs(load_agents(), load_anonymized())
    plt.rcdefaults()
    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(5.5, 3.0))
        axes_y, axes_h = .44, .43
        ax_a = fig.add_axes([.075, axes_y, .235, axes_h])
        ax_b = fig.add_axes([.375, axes_y, .235, axes_h])
        ax_c = fig.add_axes([.76, axes_y, .225, axes_h])
        panel_titles(fig, [(.075, "(a) MedCalc calculators", "Official test split"),
                           (.375, "(b) Text classification", "Five datasets"),
                           (.655, "(c) Without feature names", "CLIPPER (tools), $n=250$")])
        med_handles = draw_lines(ax_a, medcalc, list(MEDCALC_STYLES), MEDCALC_STYLES)
        text_handles = draw_lines(ax_b, text, list(TEXT_STYLES), TEXT_STYLES)
        draw_anonymization(ax_c, paired)
        ax_c.set_xlabel("Anonymized − natural\nerror (pp)", labelpad=3, linespacing=1.15)
        ax_a.set_ylabel("Test error (%)", labelpad=3)
        ax_a.set_xscale("log", base=2)
        ax_a.set_xlim(43, 4200)
        ax_a.set_xticks(list(PLOTTED_SIZE.values()), ["55", "220", "880", "3.3k"])
        ax_a.xaxis.set_minor_locator(NullLocator())
        ax_a.set_ylim(50, 92)
        ax_a.set_yticks([50, 60, 70, 80, 90])
        ax_a.set_xlabel("Training examples", labelpad=3)
        ax_b.set_xscale("log")
        ax_b.set_xlim(42, 880)
        ax_b.set_xticks([50, 100, 200, 400, 750], ["50", "100", "200", "400", "750"])
        ax_b.xaxis.set_minor_locator(NullLocator())
        ax_b.set_ylim(12, 37)
        ax_b.set_yticks([15, 20, 25, 30, 35])
        ax_b.set_xlabel("Training examples", labelpad=3)
        legend = dict(loc="upper left", frameon=False, handlelength=1.4, handletextpad=.4,
                      columnspacing=.7, labelspacing=.4, borderaxespad=0, fontsize=LEGEND_SIZE)
        fig.legend([med_handles[l] for l in MEDCALC_STYLES],
                   [v[0] for v in MEDCALC_STYLES.values()],
                   bbox_to_anchor=(.005, .255), ncol=1, **legend)
        fig.legend([text_handles[l] for l in TEXT_STYLES], [v[0] for v in TEXT_STYLES.values()],
                   bbox_to_anchor=(.33, .255), ncol=1, **legend)
        save(fig, "figure4-text", dpi=300)


# ---------------------------------------------------------------------------
# Appendix figure: Sol with and without the explicit no-learners instruction
# ---------------------------------------------------------------------------

SOL_ORDER = ["college", "water_potability", "stroke", "telco_churn", "compas", "heloc",
             "adult", "bank_marketing"]
SOL_PRETTY = dict(zip(SOL_ORDER, ["College", "Water pot.", "Stroke", "Telco", "COMPAS",
                                  "HELOC", "Adult", "Bank"]))


def sol_pairs(raw):
    """Paired seed differences: Sol (no learners) − Sol (spec gaming), seeds 1-3."""
    q = raw[raw.seed.between(1, 3)]
    a = q[q.label == "agent_gpt56_sol_norun"]
    b = q[q.label == "agent_gpt56_sol"]
    z = a.merge(b, on=["dataset", "size", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
    assert len(z) == 24
    z["delta"] = z.err_a - z.err_b
    return z


def paired_macro(z, datasets):
    g = z[z.dataset.isin(datasets)].groupby("dataset").delta.agg(["mean", sem])
    return float(g["mean"].mean()), float(np.linalg.norm(g["sem"]) / len(g))


def sol_behavior():
    """new-sol-behavior (appendix only)."""
    z = sol_pairs(load_agents())
    purple, ink, gray, grid = "#73558f", "#26343c", "#8a949b", "#e5e9eb"
    plt.rcdefaults()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#aeb7bd", "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"}):
        labels = SOL_ORDER + ["macro_eight", "macro_seven"]
        fig, ax = plt.subplots(figsize=(6, 3.15))
        for y, label in enumerate(labels):
            if label.startswith("macro"):
                pool = EIGHT if label == "macro_eight" else SEVEN
                vals = z[z.dataset.isin(pool)].groupby("seed").delta.mean().to_numpy()
                mean, error = paired_macro(z, pool)
            else:
                vals = z[z.dataset == label].sort_values("seed").delta.to_numpy()
                mean, error = vals.mean(), sem(vals)
            ax.scatter(vals, y + np.linspace(-.1, .1, len(vals)), s=11, color=gray,
                       alpha=.45, edgecolors="none")
            ax.errorbar(mean, y, xerr=error, fmt="D" if label.startswith("macro") else "o",
                        color=ink if label.startswith("macro") else purple,
                        ms=4.2, capsize=2.5, lw=1.2)
        ax.set_yticks(range(10), [SOL_PRETTY[x] for x in SOL_ORDER] +
                      ["Macro, all eight", "Macro, fixed seven"])
        ax.set_ylim(9.5, -.5)
        ax.axvline(0, color="#aab3b9", lw=.8)
        ax.axhline(7.5, color="#c8ced2", lw=.7)
        ax.set_xlabel("Sol − Sol (spec gaming) error (percentage points)")
        ax.grid(axis="x", color=grid, lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        fig.subplots_adjust(left=.225, right=.99, bottom=.18, top=.98)
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "new-sol-behavior.pdf", bbox_inches="tight", pad_inches=.02,
                    metadata={"CreationDate": None})
        fig.savefig(FIGURES / "new-sol-behavior.png", dpi=240, bbox_inches="tight", pad_inches=.02)
        plt.close(fig)


def all_baselines():
    """Figure 2 (all-baselines.png)."""
    plt.rcdefaults()
    figure_baselines.build()


def build_all():
    """Build every figure, in paper order; returns the output file stems."""
    builders = [("tabular-main", figure1_overview), ("all-baselines", all_baselines),
                ("figure3-tabular", figure3_tabular), ("figure4-text", figure4_text),
                ("new-sol-behavior", sol_behavior)]
    for _, build in builders:
        build()
    return [name for name, _ in builders]
