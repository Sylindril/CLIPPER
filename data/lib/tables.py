"""Every table in the paper, rebuilt from the per-seed results in data/results.

Each builder writes output/tables/<name>.tex (the same file name the paper
inputs from its tables/ directory) and returns the LaTeX text.
"""

import itertools
import statistics

import numpy as np
import pandas as pd

from .common import (DATASET_NAMES, EIGHT, METHOD_NAMES, RESULTS, SEVEN, fmt,
                     load_agents, load_anonymized, load_calculator, load_extraction,
                     load_tabular, lowest, macro, macro_table, per_dataset, pm, sem,
                     write_tabular, write_tex)


# ---------------------------------------------------------------------------
# Tabular benchmark (ten seeds, fixed seven datasets)
# ---------------------------------------------------------------------------

def fixed_seven_summary(tab):
    """Appendix table: macro error of all fifteen main methods at three sizes."""
    labels = ["slipper", "slipper_M100", "slopper_notools", "slopper", "candcond_notools",
              "candcond_tool_T2", "rulefit", "boosted_rules", "figs", "corels",
              "tao_tree", "hstree", "greedy_rule_list", "fpskope", "skope_rules"]
    sizes = [250, 1200, 2000]
    mm = macro_table(tab, labels).set_index(["label", "size"])
    best = {n: lowest([mm.loc[(l, n), "mean"] for l in labels]) for n in sizes}
    rows = [["Method", "$n=250$", "$n=1{,}200$", "$n=2{,}000$"], r"\midrule"]
    for i, label in enumerate(labels):
        rows.append([METHOD_NAMES[label]] + [
            pm(mm.loc[(label, n), "mean"], mm.loc[(label, n), "sem"], best[n][i]) for n in sizes])
    return write_tabular("fixed-seven-summary", "lrrr", rows)


def outer_loop_main(tab):
    """Main-text table: boosting vs. set cover vs. one-shot, LLM vs. discrete search."""
    sizes = [100, 250, 600, 1200, 2000]
    blocks = [[("Boosting", "Gemini 3.1 Flash-Lite", "slopper"),
               ("Set cover", "Gemini 3.1 Flash-Lite", "slopper_cover"),
               ("One-shot", "Gemini 3.1 Flash-Lite", "oneshot_slopper")],
              [("Boosting", "discrete search", "slipper"),
               ("Set cover", "discrete search", "discrete_cover_repaired")]]
    mm = macro_table(tab, [l for b in blocks for *_, l in b]).set_index(["label", "size"])
    rows = [["Harness", "Model / rule search"] + [f"$n={n:,}$".replace(",", "{,}") for n in sizes],
            r"\midrule"]
    for k, block in enumerate(blocks):
        if k:
            rows.append(r"\midrule")
        best = {n: lowest([mm.loc[(l, n), "mean"] for *_, l in block]) for n in sizes}
        for i, (harness, model, label) in enumerate(block):
            rows.append([harness, model] + [
                pm(mm.loc[(label, n), "mean"], mm.loc[(label, n), "sem"], best[n][i])
                for n in sizes])
    return write_tabular("outer-loop-main", "llccccc", rows)


def harness_main(tab):
    """Appendix table: alternative outer loops at n=250 and n=1,200."""
    rows_spec = [("slipper", "Boosting / SLIPPER"),
                 ("discrete_cover", "Legacy covering / SLIPPER"),
                 ("slopper", "Boosting / CLIPPER coverage"),
                 ("slopper_cover", "Covering / CLIPPER, patience 1"),
                 ("slopper_cover_p3", "Covering / CLIPPER, patience 3"),
                 ("oneshot_slopper", "One-shot / CLIPPER coverage"),
                 ("oneshot_candcond", "One-shot / CLIPPER hints"),
                 ("discrete_cover_repaired", "Repaired covering / SLIPPER")]
    labels = [l for l, _ in rows_spec]
    mm = macro_table(tab, labels).set_index(["label", "size"])
    best = {n: lowest([mm.loc[(l, n), "mean"] for l in labels]) for n in [250, 1200]}
    rows = [["Harness / rule proposer", "$n=250$", "$n=1{,}200$"], r"\midrule"]
    for i, (label, display) in enumerate(rows_spec):
        if label == "discrete_cover_repaired":
            rows.append(r"\midrule")
        rows.append([display] + [pm(mm.loc[(label, n), "mean"], mm.loc[(label, n), "sem"],
                                    best[n][i]) for n in [250, 1200]])
    return write_tabular("harness-main", "lrr", rows)


# Dataset-level appendix tables ("hero" tables), one longtable per panel.
HERO_NAMES = {
    "slopper": r"CLIPPER\\tools", "slopper_notools": r"CLIPPER\\no tools",
    "candcond_notools": r"CLIPPER\\suggestions",
    "candcond_tool_T2": r"CLIPPER\\suggestions\\+ tools",
    "slipper": "SLIPPER", "slipper_M100": r"SLIPPER\\$M=100$",
    "slopper_cover": r"CLIPPER\\set cover", "discrete_cover": r"Discrete\\set cover",
    "candcond_cover": r"Suggestions\\set cover",
    "oneshot_slopper": r"One-shot\\CLIPPER", "oneshot_candcond": r"One-shot\\suggestions",
    "boosted_rules": r"Boosted\\stumps", "rulefit": "RuleFit", "figs": "FIGS",
    "tao_tree": "TAO tree", "hstree": "HS tree", "corels": "CORELS",
    "greedy_rule_list": r"Greedy\\rule list", "fpskope": "FPSkope",
    "skope_rules": "SkopeRules",
}
SYMBOLIC = ["slipper", "rulefit", "boosted_rules", "figs", "corels", "tao_tree",
            "hstree", "greedy_rule_list", "fpskope", "skope_rules"]
SLOPSLIP = ["slipper", "slipper_M100", "slopper_notools", "slopper",
            "candcond_notools", "candcond_tool_T2"]
HARNESSES = ["slipper", "slopper", "candcond_tool_T2", "slopper_cover", "discrete_cover",
             "candcond_cover", "oneshot_slopper", "oneshot_candcond"]
PUNCHLINE = ["slipper", "slopper_notools", "slopper", "candcond_notools", "candcond_tool_T2"]
HERO = [
    ("hero-250-symbolic", "Discrete baselines", SYMBOLIC, [250]),
    ("hero-1200-symbolic-corrected", "Discrete baselines", SYMBOLIC, [1200]),
    ("hero-250-slopslip", "CLIPPER and SLIPPER", SLOPSLIP, [250]),
    ("hero-1200-slopslip", "CLIPPER and SLIPPER", SLOPSLIP, [1200]),
    ("hero-250-harness", "Alternative harnesses", HARNESSES, [250]),
    ("hero-1200-harness", "Alternative harnesses", HARNESSES, [1200]),
    ("hero-punchline-curve", "Complete summary learning-curve results", PUNCHLINE,
     [100, 175, 250, 400, 500, 600, 800, 1200, 2000]),
]


def fractional_ranks(errors):
    """{label: error} -> {label: rank}; 1 is lowest error, ties share the average rank."""
    return {label: rank for label, rank in
            pd.Series(errors).rank(method="average").items()}


def hero_comparison(cells, labels, size):
    """Per-dataset cells, macro, and mean rank for one comparison at one size.

    Macro and rank use the datasets every compared method has at this size
    (College is available only up to n=600); ranks use unrounded means.
    """
    at = cells[(cells["size"] == size) & cells.label.isin(labels)]
    datasets = [d for d in EIGHT if set(at[at.dataset == d].label) == set(labels)]
    at = at[at.dataset.isin(datasets)].set_index(["label", "dataset"])
    summary = {}
    for label in labels:
        summary[label] = dict(zip(["mean", "sem"], macro(at.loc[label])))
    ranks = {label: [] for label in labels}
    for d in datasets:
        for label, rank in fractional_ranks({l: at.loc[(l, d), "mean"] for l in labels}).items():
            ranks[label].append(rank)
    for label in labels:
        summary[label]["rank"] = float(np.mean(ranks[label]))
    order = sorted(labels, key=lambda l: summary[l]["mean"])
    return at, datasets, summary, order


def hero_table(name, title, labels, sizes, cells):
    parts = []
    for size in sizes:
        at, datasets, summary, order = hero_comparison(cells, labels, size)
        panels = [order[i:i + 3] for i in range(0, len(order), 3)]
        for k, panel in enumerate(panels, start=1):
            bold = len(panel) > 1
            header = ("Dataset / statistic & " + " & ".join(
                r"\shortstack{" + HERO_NAMES[l] + "}" for l in panel) + r" \\")
            cols = "l" + "r" * len(panel)
            lines = [
                r"\begin{longtable}{@{}" + cols + r"@{}}",
                rf"\caption{{{title} at $n={size}$, panel {k} of {len(panels)}.}}"
                rf"\label{{tab:{name}-{size}-{k}}}\\",
                r"\toprule", header, r"\midrule", r"\endfirsthead",
                rf"\caption[]{{{title} at $n={size}$ (continued).}}\\",
                r"\toprule", header, r"\midrule", r"\endhead",
                r"\bottomrule", r"\endfoot"]

            def row(first, values, errors=None):
                flags = lowest(values) if bold else [False] * len(values)
                out = []
                for v, e, b in zip(values, errors or [None] * len(values), flags):
                    body = fmt(v) if e is None else f"{fmt(v)}\\pm {fmt(e)}"
                    out.append(f"$\\boldsymbol{{{body}}}$" if b else f"${body}$")
                return first + " & " + " & ".join(out) + r" \\"

            for d in datasets:
                lines.append(row(DATASET_NAMES[d], [at.loc[(l, d), "mean"] for l in panel],
                                 [at.loc[(l, d), "sem"] for l in panel]))
            lines.append(r"\midrule")
            lines.append(row("Macro", [summary[l]["mean"] for l in panel],
                             [summary[l]["sem"] for l in panel]))
            lines.append(row("Mean rank", [summary[l]["rank"] for l in panel]))
            lines.append(r"\end{longtable}")
            parts.append("\n".join(lines) + "\n")
    return write_tex(name, "\n".join(parts))


def hero_tables(tab):
    cells = per_dataset(tab)
    return {name: hero_table(name, title, labels, sizes, cells)
            for name, title, labels, sizes in HERO}


def model_complexity(tab):
    """Appendix table: retained rules (tabular) and Python extraction source size."""
    def number(value):
        return f"{value:.1f}".rstrip("0").rstrip(".")

    def quantiles(values):
        # Linear interpolation at (n-1)p, as numpy/pandas do by default.
        return [float(np.quantile(values, p)) for p in (.25, .5, .75)]

    labels = ["slipper", "slipper_M100", "slopper_notools", "slopper",
              "candcond_notools", "candcond_tool_T2"]
    lines = [r"\multicolumn{4}{@{}l}{(a) Tabular retained rules: mean (median)} \\",
             ["Method", "$n=250$", "$n=1{,}200$", "$n=2{,}000$"], r"\midrule"]
    fits = tab[tab.label.isin(labels) & tab.dataset.isin(SEVEN)]
    for label in labels:
        cells = []
        for size in [250, 1200, 2000]:
            rules = fits[(fits.label == label) & (fits["size"] == size)].rules
            assert len(rules) == 70
            cells.append(f"{rules.mean():.1f} ({number(rules.median())})")
        lines.append([METHOD_NAMES[label]] + cells)
    rules = pd.read_csv(RESULTS / "model-complexity/python-extraction-rules.csv")
    ensembles = rules.groupby(["sample_cap", "seed", "key"]).size().rename("rules").reset_index()
    lines += [r"\midrule",
              r"\multicolumn{4}{@{}l}{(b) Python extraction: median [first, third quartile]} \\",
              ["Sample cap", "Rules per key", "Code lines per rule", "Tests per rule"],
              r"\midrule"]
    for cap in [20, 55, 110]:
        cells = []
        for values in [ensembles[ensembles.sample_cap == cap].rules,
                       rules[rules.sample_cap == cap].code_lines,
                       rules[rules.sample_cap == cap].if_tests]:
            q25, q50, q75 = quantiles(values)
            cells.append(f"{number(q50)} [{number(q25)}, {number(q75)}]")
        lines.append([f"$M={cap}$"] + cells)
    return write_tabular("model-complexity", "lrrr", lines)


def setcover_development():
    """Inline table in the development appendix: discrete covering repairs."""
    frame = pd.read_csv(RESULTS / "tabular/set-cover-development.csv")
    spec = [("discrete_cover", "Legacy covering"),
            ("discrete_cover_fix1", "Gate-aware sizing"),
            ("discrete_cover_linear_only", "Linear growth"),
            ("discrete_cover_fix12_linear", "Gate-aware sizing + linear growth"),
            ("discrete_coverlist", "Decision list, FOIL growth"),
            ("discrete_coverlist_linear", "Decision list, linear growth")]
    means = [frame[frame.label == l].err.mean() for l, _ in spec]
    best = lowest(means)
    rows = [["Condition", r"Pooled error (\%)", "Zero-rule cells / 440"], r"\midrule"]
    for (label, name), mean, b in zip(spec, means, best):
        q = frame[frame.label == label]
        assert len(q) == 440
        cell = rf"\textbf{{{fmt(mean)}}}" if b else fmt(mean)
        rows.append([name, cell, str(int((q.rules == 0).sum()))])
    return write_tabular("setcover-development", "lrr", rows)


def development_configurations():
    """Inline table in the development appendix: configurations on the ten-dataset
    development suite (full-data comparison), macro over ten seeds per dataset."""
    frame = pd.read_csv(RESULTS / "tabular/development-configurations.csv")
    spec = [("discrete", "Discrete SLIPPER, full data"),
            ("table_repr", "CLIPPER, table sample, no tools (earlier prompt)"),
            ("table_repr_valtools", "Per-attribute SLIPPER condition tools"),
            ("tv2_lambda", "One-conjunction framing with coverage tool"),
            ("tv2_lambda_toolsv1", "Revised coverage-tool prompt"),
            ("candcond_tool_T2", "Condition-hints table with iterative queries")]
    means = []
    for label, _ in spec:
        q = frame[frame.label == label]
        assert len(q) == 100 and q.dataset.nunique() == 10, label
        means.append(q.groupby("dataset").err.mean().mean())
    best = lowest(means)
    rows = [["Configuration", "Macro error"], r"\midrule"]
    for (_, name), mean, b in zip(spec, means, best):
        rows.append([name, rf"\textbf{{{fmt(mean)}}}" if b else fmt(mean)])
    return write_tabular("development-configurations", "lr", rows)


# ---------------------------------------------------------------------------
# MedCalc-Bench-Verified: calculator prediction and value extraction
# ---------------------------------------------------------------------------

CALCULATOR = ["slipper_full", "slipper_M20", "slipper_M100", "slopper_raw_M20",
              "slopper_raw_M100"]


def calculator_summary(split="final"):
    frame = load_calculator()
    frame = frame[frame.split == split]
    out = frame.groupby(["condition", "cap"]).accuracy.agg(["mean", sem, "count"]).reset_index()
    assert (out["count"] == 10).all()
    return out.rename(columns={"sem": "sem"})


def calculator_final_repeats():
    s = calculator_summary().set_index(["condition", "cap"])
    rows = [["Cap", r"\multicolumn{3}{c}{SLIPPER}", r"\multicolumn{2}{c}{CLIPPER (raw notes)}"],
            r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
            ["", "Full", "$M=20$", "$M=100$", "$M=20$", "$M=100$"], r"\midrule"]
    for cap in [1, 4, 16, 64]:
        rows.append([str(cap)] + [pm(s.loc[(c, cap), "mean"], s.loc[(c, cap), "sem"])
                                  for c in CALCULATOR])
    return write_tabular("calculator-final-repeats", "rrrrrr", rows)


EXTRACTION = ["extract_pool"] + [f"extract_{b}_M{m}" for b in ["python", "regex"]
                                 for m in [20, 55, 110]]
EXTRACTION_METRICS = ["micro_p", "micro_r", "micro_f1", "exact_set"]


def extraction_summary(split="final"):
    frame = load_extraction()
    frame = frame[frame.split == split]
    agg = frame.groupby("condition")[EXTRACTION_METRICS].agg(["mean", sem])
    return agg


def extraction_final_repeats():
    s = extraction_summary()
    rows = [["Rule source", "Precision", "Recall", "F1", "All selected"], r"\midrule"]
    for condition in EXTRACTION:
        if condition == "extract_pool":
            rows.append(["Hand-written pool"] + [f"${fmt(s.loc[condition, (m, 'mean')])}$"
                                                 for m in EXTRACTION_METRICS])
        else:
            _, body, m = condition.split("_")
            rows.append([f"{body.title()}, ${m[0]}={m[1:]}$"] + [
                pm(s.loc[condition, (k, "mean")], s.loc[condition, (k, "sem")])
                for k in EXTRACTION_METRICS])
    return write_tabular("extraction-final-repeats", "lrrrr", rows)


def extraction_f1_main():
    s = extraction_summary()
    f1 = lambda c: pm(s.loc[c, ("micro_f1", "mean")], s.loc[c, ("micro_f1", "sem")])
    rows = [["$M$", "Python", "Regex"], r"\midrule"]
    for m in [20, 55, 110]:
        rows.append([str(m), f1(f"extract_python_M{m}"), f1(f"extract_regex_M{m}")])
    rows += [r"\midrule", ["Pool", r"\multicolumn{2}{c}{$"
                           + fmt(s.loc["extract_pool", ("micro_f1", "mean")]) + "$}"]]
    return write_tabular("extraction-f1-main", "lcc", rows)


# ---------------------------------------------------------------------------
# n=250 agents, backends, and anonymization (three or ten matched seeds)
# ---------------------------------------------------------------------------

AGENT_NAMES = {
    "agent_gpt56_sol": "Sol (spec gaming)",
    "agent_gpt56_sol_norun": "Sol",
    "agent_opus5": "Opus 5",
    "agent_gemini35_flash": "Gemini 3.5 CLI",
    "agent_gpt56_terra": "Terra",
    "agent_sonnet5": "Sonnet 5",
    "agent_gpt56_luna": "Luna",
    "agent_gemini31_flashlite": "Gemini 3.1 CLI",
    "agent_haiku45": "Haiku 4.5",
    "slopper": "CLIPPER (gemini 3.1 flash-lite)",
    "slipper": "SLIPPER",
}
T0 = ["slopper_sonnet_T0", "slopper_notools", "slopper_g38_T0", "slopper_g25fl_T0",
      "slopper_qwen9b_T0", "slopper_haiku_T0"]
TOOLS = ["slopper_g38", "slopper", "slopper_haiku", "slopper_luna", "slopper_g25fl"]
BACKEND_NAMES = {
    "slopper_sonnet_T0": "Sonnet 5", "slopper_notools": "Gemini 3.1 Flash-Lite",
    "slopper_g38_T0": "Gemini 3.8", "slopper_g25fl_T0": "Gemini 2.5 Flash-Lite",
    "slopper_qwen9b_T0": "Qwen 3.5 9B", "slopper_haiku_T0": "Haiku 4.5",
    "slopper_g38": "Gemini 3.8", "slopper": "Gemini 3.1 Flash-Lite",
    "slopper_haiku": "Haiku 4.5", "slopper_luna": "Luna",
    "slopper_g25fl": "Gemini 2.5 Flash-Lite",
}


def three_seed(raw, label):
    """Seeds 1-3 of one condition on all eight datasets."""
    q = raw[(raw.label == label) & raw.seed.between(1, 3)]
    assert len(q) == 24, label
    return q


def n250_summary(raw, label):
    q = three_seed(raw, label)
    cells = q.groupby("dataset").err.agg(mean="mean", sem=sem)
    return dict(cells=cells["mean"], eight=macro(cells), seven=macro(cells.loc[SEVEN]))


def new_harness_full(raw):
    order = ["agent_gpt56_sol_norun", "agent_opus5", "agent_gpt56_terra", "slopper",
             "agent_sonnet5", "agent_gemini35_flash", "agent_gpt56_luna", "slipper",
             "agent_gemini31_flashlite", "agent_haiku45"]
    columns = ["college", "water_potability", "stroke", "telco_churn", "compas", "heloc",
               "adult", "bank_marketing"]
    s = {label: n250_summary(raw, label) for label in order + ["agent_gpt56_sol"]}
    order.sort(key=lambda l: s[l]["seven"][0])
    bold_cell = {d: lowest([s[l]["cells"][d] for l in order]) for d in columns}
    bold_eight = lowest([s[l]["eight"][0] for l in order])
    bold_seven = lowest([s[l]["seven"][0] for l in order])
    gaming = s["agent_gpt56_sol"]
    lines = [r"{\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
             r"System & College & Water & Stroke & Telco & COMPAS & HELOC & Adult & Bank & Macro 8 & Macro 7 \\",
             r"\midrule",
             # The scikit-learn-fitting Sol run is kept out of the table (Appendix discussion).
             "%Sol (spec gaming) & " + " & ".join(f"{gaming['cells'][d]:.2f}" for d in columns)
             + " & " + rf"${gaming['eight'][0]:.2f}\pm{gaming['eight'][1]:.2f}$"
             + " & " + rf"${gaming['seven'][0]:.2f}\pm{gaming['seven'][1]:.2f}$" + r" \\"]
    for i, label in enumerate(order):
        cells = [rf"\textbf{{{fmt(s[label]['cells'][d])}}}" if bold_cell[d][i]
                 else fmt(s[label]["cells"][d]) for d in columns]
        lines.append(AGENT_NAMES[label] + " & " + " & ".join(cells) + " & "
                     + pm(*s[label]["eight"], bold_eight[i]) + " & "
                     + pm(*s[label]["seven"], bold_seven[i]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", ""]
    return write_tex("new-harness-full", "\n".join(lines))


def agents_main(raw):
    blocks = [[("Codex", "agent_gpt56_sol_norun", "Sol", "default"),
               ("", "agent_gpt56_terra", "Terra", "default"),
               ("", "agent_gpt56_luna", "Luna", "default"),
               ("Claude Code", "agent_opus5", "Opus 5", "default"),
               ("", "agent_sonnet5", "Sonnet 5", "default"),
               ("", "agent_haiku45", "Haiku 4.5", "default"),
               ("Gemini CLI", "agent_gemini35_flash", "Gemini 3.5 Flash", "default"),
               ("", "agent_gemini31_flashlite", "Gemini 3.1 Flash-Lite", "default")],
              [("Boosting", "slopper", "Gemini 3.1 Flash-Lite", "coverage"),
               ("Boosting", "slipper", "discrete search", "---")]]
    rows = [["Harness", "Model / rule search", "Tools", r"Error (\%)"], r"\midrule"]
    for k, block in enumerate(blocks):
        if k:
            rows.append(r"\midrule")
        values = [n250_summary(raw, label)["seven"] for _, label, _, _ in block]
        best = lowest([v[0] for v in values])
        for i, ((harness, label, model, tools), value) in enumerate(zip(block, values)):
            if k == 0 and i in (3, 6):
                rows.append(r"\addlinespace[2pt]")
            name = rf"\textbf{{{model}}}" if best[i] else model
            rows.append([harness, name, tools, pm(*value, best[i])])
    return write_tabular("agents-main", "lllc", rows)


def new_backend_full(raw):
    rows = [["Backend", "Macro, all eight", "Macro, fixed seven"], r"\midrule"]
    for k, (heading, labels) in enumerate([("Without coverage tools", T0),
                                           ("With coverage tools", TOOLS)]):
        if k:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{3}{@{}l}{\textit{" + heading + r"}} \\")
        s = {label: n250_summary(raw, label) for label in labels}
        labels = sorted(labels, key=lambda l: s[l]["eight"][0])
        b8 = lowest([s[l]["eight"][0] for l in labels])
        b7 = lowest([s[l]["seven"][0] for l in labels])
        for i, label in enumerate(labels):
            rows.append([BACKEND_NAMES[label], pm(*s[label]["eight"], b8[i]),
                         pm(*s[label]["seven"], b7[i])])
    return write_tabular("new-backend-full", "lrr", rows)


def backends_main(raw):
    s = {label: n250_summary(raw, label)["seven"] for label in T0}
    labels = sorted(T0, key=lambda l: s[l][0])
    names = {**BACKEND_NAMES, "slopper_g38_T0": "Gemini 3.8 Flash"}
    best = lowest([s[l][0] for l in labels])
    rows = [["Model", r"Error (\%)"], r"\midrule"]
    for label, b in zip(labels, best):
        rows.append([rf"\textbf{{{names[label]}}}" if b else names[label], pm(*s[label], b)])
    return write_tabular("backends-main", "lc", rows)


def anonymization_pairs(raw, anon):
    """Paired (natural, anonymized) CLIPPER-with-tools errors, ten seeds, n=250."""
    natural = raw[raw.label == "slopper"]
    z = anon.merge(natural, on=["dataset", "size", "seed"], suffixes=("_anon", "_natural"),
                   validate="one_to_one")
    assert len(z) == 80 and (z.bestj_anon == z.bestj_natural).all()
    z["delta"] = z.err_anon - z.err_natural
    return z


def new_anonymization_full(raw, anon):
    z = anonymization_pairs(raw, anon)
    order = ["college", "water_potability", "stroke", "telco_churn", "compas", "heloc",
             "adult", "bank_marketing"]
    short = dict(zip(order, ["College", "Water pot.", "Stroke", "Telco", "COMPAS",
                             "HELOC", "Adult", "Bank"]))
    cells = z.groupby("dataset").agg(nat=("err_natural", "mean"), nat_sem=("err_natural", sem),
                                     an=("err_anon", "mean"), an_sem=("err_anon", sem),
                                     d=("delta", "mean"), d_sem=("delta", sem))
    rows = [["Dataset", "Original", "Anonymized", "Paired change (pp)"], r"\midrule"]

    def row(name, nat, an, delta):
        b = lowest([nat[0], an[0]])
        return [name, pm(*nat, b[0]), pm(*an, b[1]), pm(*delta)]

    for d in order:
        c = cells.loc[d]
        rows.append(row(short[d], (c.nat, c.nat_sem), (c.an, c.an_sem), (c.d, c.d_sem)))
    rows.append(r"\midrule")
    for name, pool in [("Macro, all eight", EIGHT), ("Macro, fixed seven", SEVEN)]:
        c = cells.loc[pool]
        rows.append(row(name, macro(c[["nat", "nat_sem"]].set_axis(["mean", "sem"], axis=1)),
                        macro(c[["an", "an_sem"]].set_axis(["mean", "sem"], axis=1)),
                        macro(c[["d", "d_sem"]].set_axis(["mean", "sem"], axis=1))))
    return write_tabular("new-anonymization-full", "lrrr", rows)


# ---------------------------------------------------------------------------
# Text classification appendix (CLIPPER without tools, RuleChef, SLIPPER, HS-Tree)
# ---------------------------------------------------------------------------

TEXT_NAMES = {"sms_spam": "SMS spam", "sentiment_amazon": "Amazon",
              "sentiment_imdb": "IMDb", "sentiment_yelp": "Yelp",
              "youtube_spam": "YouTube spam"}
TEXT_METHODS = {"slopper_text": "CLIPPER (no tools)", "rulechef": "RuleChef",
                "slipper": "SLIPPER, full data", "hstree": "HS-Tree"}


def text_suite_error():
    """Test error per text dataset and training size; bold is the lowest per size."""
    from .figures import TEXT_DATASETS, TEXT_SIZES, load_text
    frame = load_text()
    frame = frame[frame.label.isin(TEXT_METHODS)]
    stats = frame.groupby(["dataset", "label", "size"]).err.agg(mean="mean", sem=sem)
    blocks = [(TEXT_NAMES[d], stats.loc[d]) for d in TEXT_DATASETS]
    blocks.append(("Macro", stats.groupby(["label", "size"]).apply(macro).apply(
        pd.Series).set_axis(["mean", "sem"], axis=1)))
    rows = [["Method"] + [f"$n={n}$" for n in TEXT_SIZES]]
    for name, block in blocks:
        rows += [r"\midrule", rf"\multicolumn{{{len(TEXT_SIZES) + 1}}}{{@{{}}l}}{{\textit{{{name}}}}} \\"]
        best = {n: lowest([block.loc[(l, n), "mean"] for l in TEXT_METHODS]) for n in TEXT_SIZES}
        for i, (label, method) in enumerate(TEXT_METHODS.items()):
            cells = []
            for n in TEXT_SIZES:
                body = f"{fmt(block.loc[(label, n), 'mean'])}{{\\pm}}{fmt(block.loc[(label, n), 'sem'])}"
                cells.append(f"$\\boldsymbol{{{body}}}$" if best[n][i] else f"${body}$")
            rows.append([method] + cells)
    return write_tabular("text-suite-error", "l" + "r" * len(TEXT_SIZES), rows)


def text_suite_rulechef():
    """RuleChef's mean rule count and test coverage (%) per dataset at n=50, 200, 750."""
    from .figures import TEXT_DATASETS, load_text
    frame = load_text()
    stats = frame[frame.label == "rulechef"].groupby(["dataset", "size"])[["rules", "coverage"]].mean()
    shown = [50, 200, 750]
    k = len(shown)
    rows = [rf" & \multicolumn{{{k}}}{{c}}{{Rules}} & "
            rf"\multicolumn{{{k}}}{{c}}{{Test documents matched (\%)}} \\",
            rf"\cmidrule(lr){{2-{1 + k}}}\cmidrule(l){{{2 + k}-{1 + 2 * k}}}",
            ["Dataset"] + [f"$n={n}$" for n in shown * 2], r"\midrule"]
    for d in TEXT_DATASETS + ["mean"]:
        if d == "mean":
            rows.append(r"\midrule")
            s_ = stats.groupby(level="size").mean()
        else:
            s_ = stats.loc[d]
        rows.append(["Mean" if d == "mean" else TEXT_NAMES[d]]
                    + [fmt(s_.loc[n, "rules"]) for n in shown]
                    + [fmt(s_.loc[n, "coverage"], 0) for n in shown])
    return write_tabular("text-suite-rulechef", "l" + "r" * (2 * k), rows)


def banking77_error():
    """Banking77 test error per size, mean and SEM over ten seeds; bold is the lowest."""
    from .figures import BANKING77_SIZES, BANKING77_STYLES, banking77_summary
    stats = banking77_summary().set_index(["label", "size"])
    labels = list(BANKING77_STYLES)
    best = {n: lowest([stats.loc[(l, n), "mean_error"] for l in labels]) for n in BANKING77_SIZES}
    rows = [["Method"] + [f"$n={n}$" for n in BANKING77_SIZES], r"\midrule"]
    for i, label in enumerate(labels):
        cells = []
        for n in BANKING77_SIZES:
            body = f"{fmt(stats.loc[(label, n), 'mean_error'])}{{\\pm}}{fmt(stats.loc[(label, n), 'sem'])}"
            cells.append(f"$\\boldsymbol{{{body}}}$" if best[n][i] else f"${body}$")
        rows.append([BANKING77_STYLES[label][0]] + cells)
    return write_tabular("banking77-error", "l" + "r" * len(BANKING77_SIZES), rows)


# ---------------------------------------------------------------------------

def build_all():
    """Build every table; returns {name: LaTeX} in paper order."""
    tab, raw, anon = load_tabular(), load_agents(), load_anonymized()
    tables = {
        "outer-loop-main": outer_loop_main(tab),
        "agents-main": agents_main(raw),
        "backends-main": backends_main(raw),
        "extraction-f1-main": extraction_f1_main(),
        "calculator-final-repeats": calculator_final_repeats(),
        "extraction-final-repeats": extraction_final_repeats(),
        "development-configurations": development_configurations(),
        "harness-main": harness_main(tab),
        "setcover-development": setcover_development(),
        "fixed-seven-summary": fixed_seven_summary(tab),
    }
    tables.update(hero_tables(tab))
    tables.update({
        "text-suite-error": text_suite_error(),
        "text-suite-rulechef": text_suite_rulechef(),
        "banking77-error": banking77_error(),
        "new-harness-full": new_harness_full(raw),
        "new-anonymization-full": new_anonymization_full(raw, anon),
        "new-backend-full": new_backend_full(raw),
        "model-complexity": model_complexity(tab),
    })
    return tables
