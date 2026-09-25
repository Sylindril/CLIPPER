"""Paths, loaders, summary statistics, and formatting shared by tables and figures."""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

# The seven datasets of every main-text tabular comparison (College excluded).
SEVEN = ["adult", "bank_marketing", "compas", "heloc", "stroke",
         "telco_churn", "water_potability"]
EIGHT = ["adult", "bank_marketing", "college", "compas", "heloc", "stroke",
         "telco_churn", "water_potability"]
DATASET_NAMES = {"adult": "Adult", "bank_marketing": "Bank marketing",
                 "college": "College", "compas": "COMPAS", "heloc": "HELOC",
                 "stroke": "Stroke", "telco_churn": "Telco churn",
                 "water_potability": "Water potability"}

# Condition identifiers in the CSVs are experiment IDs. "slopper" is CLIPPER,
# "candcond" is CLIPPER with condition hints, and "slipper" is SLIPPER.
METHOD_NAMES = {
    "slipper": "SLIPPER full",
    "slipper_M100": "SLIPPER M100",
    "slopper_notools": "CLIPPER sample",
    "slopper": "CLIPPER coverage",
    "candcond_notools": "CLIPPER hints",
    "candcond_tool_T2": "CLIPPER hints+tools",
    "rulefit": "RuleFit", "boosted_rules": "Boosted stumps",
    "figs": "FIGS", "corels": "CORELS", "tao_tree": "TAO tree",
    "hstree": "HS tree", "greedy_rule_list": "Greedy rule list",
    "fpskope": "FPSkope", "skope_rules": "SkopeRules",
    "discrete_cover": "SLIPPER, legacy cover",
    "discrete_cover_repaired": "SLIPPER, repaired cover",
    "slopper_cover": "CLIPPER, cover (patience 1)",
    "slopper_cover_p3": "CLIPPER, cover (patience 3)",
    "oneshot_slopper": "CLIPPER, one-shot",
    "oneshot_candcond": "CLIPPER hints, one-shot",
    "candcond_cover": "CLIPPER hints, cover",
}


def load_tabular():
    """Per-seed test error (%) and retained rules for every tabular run."""
    return pd.read_csv(RESULTS / "tabular/tabular-runs.csv")


def load_symbolic_baselines():
    """Figure 2's per-seed rows, in their original plotting order."""
    return pd.read_csv(RESULTS / "tabular/symbolic-baselines.csv")


def load_calculator():
    return pd.read_csv(RESULTS / "medcalc/calculator-prediction-seeds.csv")


def load_extraction():
    return pd.read_csv(RESULTS / "medcalc/value-extraction-seeds.csv")


def load_agents():
    """n=250 agent, backend, and reference runs (natural dataset names)."""
    return pd.read_csv(RESULTS / "agents/n250-runs.csv")


def load_anonymized():
    frame = pd.read_csv(RESULTS / "agents/n250-anonymized-runs.csv")
    return frame.drop(columns="anonymized_dataset").rename(
        columns={"original_dataset": "dataset"})


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def macro(per_dataset):
    """Equal-weight macro of per-dataset (mean, sem) rows: the mean of the
    dataset means, with SEM sqrt(sum SEM_d^2) / D."""
    return (float(per_dataset["mean"].mean()),
            float(np.sqrt(np.square(per_dataset["sem"]).sum()) / len(per_dataset)))


def per_dataset(frame, value="err"):
    """Mean and SEM over seeds for every (label, size, dataset) cell."""
    return (frame.groupby(["label", "size", "dataset"])[value]
            .agg(mean="mean", sem=sem, seeds="count").reset_index())


def macro_table(frame, labels, datasets=SEVEN):
    """Macro error per (label, size) over a fixed dataset pool."""
    cells = per_dataset(frame[frame.label.isin(labels) & frame.dataset.isin(datasets)])
    rows = []
    for (label, size), group in cells.groupby(["label", "size"]):
        if set(group.dataset) != set(datasets):
            continue
        assert (group.seeds == 10).all(), (label, size)
        mean, error = macro(group)
        rows.append(dict(label=label, size=int(size), mean=mean, sem=error))
    return pd.DataFrame(rows)


def fmt(value, digits=1):
    """Round half up, as in the paper's tables.

    Values are first rounded to nine decimals to remove floating-point noise, so
    a mean such as (30.98 + 30.94 + 30.63) / 3 = 30.85 rounds up to 30.9.
    """
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(f"{float(value):.9f}").quantize(quantum, rounding=ROUND_HALF_UP))


def pm(mean, error, bold=False, space=""):
    body = f"{fmt(mean)}\\pm{space}{fmt(error)}"
    return f"$\\boldsymbol{{{body}}}$" if bold else f"${body}$"


def lowest(values):
    """Flags for the lowest displayed (rounded) value, including ties."""
    shown = [float(fmt(v)) for v in values]
    best = min(shown)
    return [v == best for v in shown]


def write_tabular(name, columns, rows, header_lines=()):
    """Write a booktabs tabular; rows are lists of cells or raw LaTeX lines."""
    lines = list(header_lines) + [r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule"]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_tex(name, "\n".join(lines) + "\n")


def write_tex(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(text)
    return text


def latex_to_text(tex):
    """Render a generated LaTeX table as readable plain text for the console."""
    import re
    rows = []
    for line in tex.splitlines():
        line = line.strip()
        if (not line or line.startswith("%") or line.startswith(r"\begin")
                or line.startswith(r"\end") or line.startswith(r"\caption")
                or line in {r"{\scriptsize", "}"} or line.startswith(r"\setlength")):
            continue
        if line in {r"\toprule", r"\midrule", r"\bottomrule"} or line.startswith(r"\cmidrule") \
                or line.startswith(r"\addlinespace"):
            rows.append(None)
            continue
        line = line.rstrip("\\").strip()
        cells = [c.strip() for c in line.split("&")]
        clean, title = [], len(cells) == 1
        for c in cells:
            span = re.match(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{\})*\}\{(.*)\}$", c)
            pad = int(span.group(1)) - 1 if span else 0
            c = span.group(2) if span else c
            c = re.sub(r"\\shortstack\{(.*)\}", lambda m: m.group(1).replace("\\\\", " "), c)
            c = re.sub(r"\\(boldsymbol|textbf|textit|texttt)\{", "{", c)
            c = c.replace(r"\pm", " ± ").replace("$", "").replace("{,}", ",")
            c = c.replace("{", "").replace("}", "").replace(r"\%", "%").replace("---", "-")
            c = c.replace(r"\dagger", "†").replace(r"\ddagger", "‡").replace("^", "")
            c = re.sub(r"\s+", " ", c).replace("± ", "±").replace(" ±", "±").strip()
            clean += [c] + [""] * pad
        rows.append(clean[0] if title else clean)
    width = max(len(r) for r in rows if isinstance(r, list))
    widths = [0] * width
    for r in rows:
        if isinstance(r, list):
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
    total = sum(widths) + 2 * (width - 1)
    out = []
    for r in rows:
        if r is None:
            if not out or out[-1] != "-" * total:
                out.append("-" * total)
        elif isinstance(r, str):
            out.append(r)
        else:
            out.append("  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                                 for i, c in enumerate(r)).rstrip())
    return "\n".join(out)


