"""In-text numbers that do not appear verbatim in a table.

Each entry is (paper location, statement, value string). Everything is
recomputed from data/results.
"""
import numpy as np
import pandas as pd

from .common import (EIGHT, RESULTS, SEVEN, fmt, load_agents, load_anonymized,
                     load_calculator, load_extraction, load_symbolic_baselines,
                     load_tabular, macro, macro_table, sem)
from .figures import (BANKING77_SIZES, CURVE_SIZES, banking77_summary,
                      calculator_error_summary, load_banking77, rule_learner_summary,
                      sol_pairs)
from .tables import anonymization_pairs, n250_summary


def pm(mean_sem):
    if isinstance(mean_sem, pd.Series):
        mean_sem = (mean_sem["mean"], mean_sem["sem"])
    return f"{fmt(mean_sem[0])} ± {fmt(mean_sem[1])}"


def paired_macro_difference(frame, a, b, datasets):
    """Macro of per-dataset paired seed differences a - b, with propagated SEM."""
    z = frame[frame.label == a].merge(frame[frame.label == b], on=["dataset", "size", "seed"],
                                      suffixes=("_a", "_b"), validate="one_to_one")
    z = z[z.dataset.isin(datasets)].assign(delta=lambda f: f.err_a - f.err_b)
    assert len(z) == 10 * len(datasets)
    return macro(z.groupby("dataset").delta.agg(mean="mean", sem=sem))


def tabular_numbers():
    fig3a = "Sec. (LLMs build better rules than discrete search), Fig. 3a"
    fig3b = "Sec. (LLMs build better rules than discrete search), Fig. 3b"
    fig3c = "Sec. (LLMs build better rules than discrete search), Fig. 3c"
    out = []
    tab = load_tabular()
    mm = macro_table(tab[tab["size"].isin(CURVE_SIZES)],
                     ["slopper_notools", "slipper_M100", "slipper",
                      "candcond_notools"]).set_index(["label", "size"])
    get = lambda l, n: (mm.loc[(l, n), "mean"], mm.loc[(l, n), "sem"])
    wins = sum(get("slopper_notools", n)[0] < get("slipper_M100", n)[0] for n in CURVE_SIZES)
    out.append((fig3a, "Sizes where CLIPPER (no tools, M=100) beats SLIPPER (M=100)",
                f"{wins} of {len(CURVE_SIZES)}"))
    at1200 = tab[tab["size"] == 1200]
    out.append((fig3a, "n=1,200: SLIPPER (M=100) minus CLIPPER, paired; the two macro errors",
                f"{pm(paired_macro_difference(at1200, 'slipper_M100', 'slopper_notools', SEVEN))}"
                f" ({fmt(get('slopper_notools', 1200)[0])} vs {fmt(get('slipper_M100', 1200)[0])})"))
    beats_full = [n for n in CURVE_SIZES if get("slopper_notools", n)[0] < get("slipper", n)[0]]
    out.append((fig3a, "Sizes where CLIPPER (M=100) beats full-data SLIPPER",
                ", ".join(map(str, beats_full))))
    hint_wins = sum(get("candcond_notools", n)[0] < get("slipper", n)[0] for n in CURVE_SIZES)
    out.append((fig3b, "Sizes where CLIPPER + hints beats full-data SLIPPER",
                f"{hint_wins} of {len(CURVE_SIZES)}"))
    out.append((fig3b, "n=1,200: hints / sample-only CLIPPER / full-data SLIPPER",
                f"{fmt(get('candcond_notools', 1200)[0])} / {fmt(get('slopper_notools', 1200)[0])}"
                f" / {fmt(get('slipper', 1200)[0])}"))

    rl = rule_learner_summary().set_index(["label", "size"])
    sizes = sorted(set(rl.index.get_level_values("size")))
    beaten = [l for l in ["corels", "ripper", "greedy_rule_list", "skope_rules", "fpskope",
                          "slipper_M100"]
              if all(rl.loc[("slopper", n), "mean_error"] < rl.loc[(l, n), "mean_error"]
                     for n in sizes)]
    out.append((fig3c, "Learners CLIPPER (tools) beats at all five sizes", ", ".join(beaten)))
    lower = [n for n in sizes
             if rl.loc[("slipper", n), "mean_error"] < rl.loc[("slopper", n), "mean_error"]]
    out.append((fig3c, "Sizes where full-data SLIPPER is lower than CLIPPER (tools)",
                ", ".join(map(str, lower))))
    clipper = tab[tab.label.isin(["slopper_notools", "slopper", "candcond_notools",
                                  "candcond_tool_T2"]) & tab.dataset.isin(SEVEN)
                  & tab["size"].isin([250, 1200, 2000])]
    medians = clipper.groupby(["label", "size"]).rules.median()
    out.append((fig3c + "; App. model complexity",
                "Median retained rules, CLIPPER conditions and sizes",
                f"{medians.min():g} to {medians.max():g}"))
    ht = clipper[(clipper.label == "candcond_tool_T2") & (clipper["size"] == 2000)].rules
    out.append(("App. model complexity", "Hints+tools rule-count range at n=2,000",
                f"{ht.min():g} to {ht.max():g}"))

    harness = "Sec. (the harness matters more than the model), one rule at a time"
    h = macro_table(tab, ["slopper", "oneshot_slopper"]).set_index(["label", "size"])
    at250 = tab[tab["size"] == 250]
    out.append((harness, "n=250: one-shot minus boosted CLIPPER, paired; the two macro errors",
                f"{pm(paired_macro_difference(at250, 'oneshot_slopper', 'slopper', SEVEN))} "
                f"({fmt(h.loc[('oneshot_slopper', 250), 'mean'])} vs "
                f"{fmt(h.loc[('slopper', 250), 'mean'])})"))
    out.append((harness, "n=1,200: one-shot vs boosted CLIPPER",
                f"{fmt(h.loc[('oneshot_slopper', 1200), 'mean'])} vs "
                f"{fmt(h.loc[('slopper', 1200), 'mean'])}"))
    return out


def setcover_numbers():
    frame = pd.read_csv(RESULTS / "tabular/set-cover-development.csv")
    key = ["dataset", "size", "seed"]
    a = frame[frame.label == "discrete_cover_fix12_linear"].set_index(key).err
    b = frame[frame.label == "discrete_cover"].set_index(key).err
    d = (a - b.loc[a.index]).round(9)
    at800 = frame[frame["size"] == 800].groupby("label").err.mean()
    return [("App. repairs to discrete covering", "Combined repair vs legacy: better / worse / tied cells",
             f"{(d < 0).sum()} / {(d > 0).sum()} / {(d == 0).sum()}"),
            ("App. repairs to discrete covering", "Size 800 pooled error: combined repair vs legacy",
             f"{fmt(at800['discrete_cover_fix12_linear'])} vs {fmt(at800['discrete_cover'])}")]


def agent_numbers():
    raw, anon = load_agents(), load_anonymized()
    out = []
    z = anonymization_pairs(raw, anon)
    cells = z.groupby("dataset").agg(nm=("err_natural", "mean"), ns=("err_natural", sem),
                                     am=("err_anon", "mean"), as_=("err_anon", sem),
                                     dm=("delta", "mean"), ds=("delta", sem))
    part = lambda c, m, s: macro(c[[m, s]].set_axis(["mean", "sem"], axis=1))
    out.append(("Sec. (background knowledge), Fig. 4d", "Anonymization, all eight: original -> anonymized; paired change",
                f"{pm(part(cells, 'nm', 'ns'))} -> {pm(part(cells, 'am', 'as_'))}; "
                f"{pm(part(cells, 'dm', 'ds'))}"))
    seven = cells.loc[SEVEN]
    out.append(("Sec. (background knowledge)", "Anonymized CLIPPER, fixed seven",
                pm(part(seven, "am", "as_"))))

    s = {l: n250_summary(raw, l) for l in raw.label.unique()}
    seven = lambda l: s[l]["seven"]
    out.append(("Sec. (the harness matters more than the model)", "Flash-Lite in Gemini CLI; Haiku 4.5 in Claude Code vs in CLIPPER (tools)",
                f"{pm(seven('agent_gemini31_flashlite'))}; {pm(seven('agent_haiku45'))} vs "
                f"{pm(seven('slopper_haiku'))}"))
    agents = ["agent_gpt56_sol_norun", "agent_opus5", "agent_gpt56_terra", "agent_sonnet5",
              "agent_gemini35_flash", "agent_gpt56_luna"]
    lo = min(seven(l)[0] for l in agents)
    hi = max(seven(l)[0] for l in agents)
    out.append(("Sec. (the harness matters more than the model)", "CLIPPER Flash-Lite (tools); range of six strongest agents; full-data SLIPPER",
                f"{pm(seven('slopper'))}; {fmt(lo)}-{fmt(hi)}; {pm(seven('slipper'))}"))
    cost = raw[raw.seed.between(1, 3) & raw.dataset.isin(SEVEN)].groupby("label").cost_usd.mean()
    out.append(("Sec. (the harness matters more than the model)", "Mean cost per fit (USD), CLIPPER Flash-Lite without tools",
                f"{cost['slopper_notools']:.2f}"))
    out.append(("Sec. (the harness matters more than the model)", "Mean cost per fit (USD): Terra, Sonnet 5, Gemini 3.5, Opus 5",
                ", ".join(f"{cost[l]:.2f}" for l in ["agent_gpt56_terra", "agent_sonnet5",
                                                      "agent_gemini35_flash", "agent_opus5"])))
    t0 = ["slopper_sonnet_T0", "slopper_notools", "slopper_g38_T0", "slopper_g25fl_T0",
          "slopper_qwen9b_T0", "slopper_haiku_T0"]
    out.append(("Sec. (the harness matters more than the model)", "Backends without tools: span of macro error",
                f"{fmt(min(seven(l)[0] for l in t0))}-{fmt(max(seven(l)[0] for l in t0))}"))

    # Appendix: Sol (spec gaming) vs Sol with the explicit no-learners instruction.
    zs = sol_pairs(raw)
    for name, pool in [("fixed seven", SEVEN), ("all eight", EIGHT)]:
        g = zs[zs.dataset.isin(pool)].groupby("dataset")
        before = macro(g.err_b.agg(mean="mean", sem=sem))
        after = macro(g.err_a.agg(mean="mean", sem=sem))
        change = macro(g.delta.agg(mean="mean", sem=sem))
        out.append(("App. Sol with and without the no-learners instruction", f"Sol (spec gaming) -> Sol, {name}; paired change",
                    f"{pm(before)} -> {pm(after)}; {pm(change)}"))
    gap_before = seven("slopper")[0] - seven("agent_gpt56_sol")[0]
    gap_after = seven("slopper")[0] - seven("agent_gpt56_sol_norun")[0]
    out.append(("App. Sol with and without the no-learners instruction", "Sol's advantage over CLIPPER (tools), before -> after",
                f"{fmt(gap_before)} -> {fmt(gap_after)}"))
    per = zs.groupby("dataset").delta.mean()
    out.append(("App. Sol with and without the no-learners instruction", "Largest paired changes: Water potability, HELOC",
                f"+{fmt(per['water_potability'])}, +{fmt(per['heloc'])}"))
    return out


def medcalc_numbers():
    where = "Sec. (text classification), Fig. 4b"
    err = calculator_error_summary("final").set_index(["label", "cap"])
    get = lambda l, c: (err.loc[(l, c), "mean_error"], err.loc[(l, c), "sem"])
    out = []
    for cap in [1, 4, 16, 64]:
        out.append((where, f"Cap {cap}: test error CLIPPER M=20 / SLIPPER M=20 / full SLIPPER",
                    f"{pm(get('slopper_raw_M20', cap))} / {pm(get('slipper_M20', cap))} / "
                    f"{pm(get('slipper_full', cap))}"))
    gaps = [get("slipper_M20", c)[0] - get("slopper_raw_M20", c)[0] for c in [4, 16, 64]]
    out.append((where, "SLIPPER M=20 minus CLIPPER M=20 at caps 4, 16, 64",
                ", ".join(fmt(g) for g in gaps)))
    ext = load_extraction()
    ext = ext[ext.split == "final"].groupby("condition")[["micro_p", "micro_r", "micro_f1"]]
    m = ext.mean()
    out.append(("Sec. (value-returning rules), Fig. 4a", "Python M=55 micro F1 vs pool",
                f"{fmt(m.loc['extract_python_M55', 'micro_f1'])} vs "
                f"{fmt(m.loc['extract_pool', 'micro_f1'])}"))
    return out


def complexity_numbers():
    rules = pd.read_csv(RESULTS / "model-complexity/python-extraction-rules.csv")
    ens = rules.groupby(["sample_cap", "seed", "key"]).code_lines.sum()
    m55 = ens.loc[55]
    q = np.quantile(m55, [.25, .5, .75])
    return [("App. model complexity", "M=55 ensemble code lines: median (IQR)",
             f"{q[1]:g} ({q[0]:.1f}-{q[2]:g})"),
            ("App. model complexity", "Retained Python rules at M=20, 55, 110",
             ", ".join(str((rules.sample_cap == c).sum()) for c in [20, 55, 110]))]


def banking77_numbers():
    """CLIPPER and RuleChef on the five BANKING77 intents."""
    where = "Sec. (text classification), Fig. 4c; App. BANKING77"
    runs = load_banking77()
    wide = runs.pivot_table(index=["size", "seed"], columns="label", values="err")
    out = []
    for label, name in [("clipper_raw_M100_tools", "CLIPPER (tools)"),
                        ("clipper_raw_M100", "CLIPPER (no tools)")]:
        gap = wide["rulechef"] - wide[label]
        out.append((where, f"RuleChef minus {name} error, mean over paired runs; runs where "
                    f"{name} is lower", f"{fmt(gap.mean())}; {(gap > 0).sum()} of {len(gap)}"))
    s = banking77_summary().set_index(["label", "size"]).mean_error
    later = [s[("rulechef", n)] for n in BANKING77_SIZES[1:]]
    out.append((where, "RuleChef error from n=100 on (min-max)",
                f"{fmt(min(later))}-{fmt(max(later))}"))
    rc = runs[runs.label == "rulechef"]
    strict = rc.groupby("size").err_strict.mean()
    out += [(where, "Share of test queries labeled by RuleChef's rules (%)", fmt(rc.coverage.mean(), 0)),
            ("App. Banking77 and RuleChef", "RuleChef error counting unlabeled queries as errors, "
             "per-size means (min-max)", f"{fmt(strict.min(), 0)}-{fmt(strict.max(), 0)}")]
    return out


def all_numbers():
    return (tabular_numbers() + agent_numbers() + medcalc_numbers() + banking77_numbers()
            + complexity_numbers() + setcover_numbers())
