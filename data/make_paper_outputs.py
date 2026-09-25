#!/usr/bin/env python3
"""Print every table and redraw every figure of the CLIPPER paper.

Reads only the per-seed results under data/results. Writes
  output/tables/<name>.tex     LaTeX tabular bodies, named as the paper's tables/*.tex
  output/figures/<name>.pdf    figures, named as the paper's figures/*
  output/figures/<name>.png
and prints each table as plain text, followed by in-text numbers that are
not shown verbatim in a table.

Usage:
  python make_paper_outputs.py                 # everything
  python make_paper_outputs.py --only tables   # tables only (plus in-text numbers)
  python make_paper_outputs.py --only figures
  python make_paper_outputs.py --only numbers

Requires numpy, pandas, matplotlib, and scipy.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import figures, numbers, tables  # noqa: E402
from lib.common import FIGURES, TABLES, latex_to_text  # noqa: E402

# Where each table appears in the paper.
TABLE_TITLES = {
    "outer-loop-main": "Table (main): iterative construction vs. one-shot; LLM vs. discrete search",
    "agents-main": "Table (main): general-purpose coding agents, n=250, fixed seven",
    "backends-main": "Table (main): changing CLIPPER's model, no coverage tools, n=250",
    "extraction-f1-main": "Table (main): MedCalc value extraction, official test micro F1",
    "calculator-final-repeats": "Appendix: MedCalc calculator prediction, test accuracy (%)",
    "extraction-final-repeats": "Appendix: MedCalc value extraction, official test",
    "development-configurations": "Appendix (inline table): development configurations, full-data comparison",
    "harness-main": "Appendix: outer-loop comparison, fixed seven",
    "setcover-development": "Appendix (inline table): discrete covering repairs, development suite",
    "fixed-seven-summary": "Appendix: fixed seven-dataset macro error",
    "hero-250-symbolic": "Appendix: discrete baselines, n=250 (per dataset)",
    "hero-1200-symbolic-corrected": "Appendix: discrete baselines, n=1,200 (per dataset)",
    "hero-250-slopslip": "Appendix: CLIPPER and SLIPPER, n=250 (per dataset)",
    "hero-1200-slopslip": "Appendix: CLIPPER and SLIPPER, n=1,200 (per dataset)",
    "hero-250-harness": "Appendix: alternative harnesses, n=250 (per dataset)",
    "hero-1200-harness": "Appendix: alternative harnesses, n=1,200 (per dataset)",
    "hero-punchline-curve": "Appendix: complete learning-curve results (per dataset)",
    "text-suite-error": "Appendix: text classification test error by dataset",
    "text-suite-rulechef": "Appendix: RuleChef rule sets (rules, % test documents matched)",
    "new-harness-full": "Appendix: complete coding-agent comparison, n=250",
    "new-anonymization-full": "Appendix: anonymization on all eight datasets, n=250",
    "new-backend-full": "Appendix: complete backend results, n=250",
    "model-complexity": "Appendix: saved model sizes",
}


def print_tables():
    built = tables.build_all()
    for name, tex in built.items():
        print("=" * 100)
        print(f"{TABLE_TITLES[name]}   [{name}.tex]")
        print("=" * 100)
        if "longtable" in tex:
            # One block per panel; each panel's caption names its size.
            for panel in tex.split(r"\end{longtable}")[:-1]:
                caption = panel.split(r"\caption{", 1)[1].split("}", 1)[0]
                print(caption.replace("$", ""))
                header = next(l for l in panel.splitlines() if l.startswith("Dataset"))
                body = panel.split(r"\endfoot", 1)[1]
                print(latex_to_text(r"\toprule" + "\n" + header + "\n" + r"\midrule" + body))
                print()
        else:
            print(latex_to_text(tex))
            print()
    print("LaTeX written to output/tables/")


def print_numbers():
    print("=" * 100)
    print("In-text numbers not shown verbatim in a table")
    print("=" * 100)
    for where, what, value in numbers.all_numbers():
        print(f"[{where}] {what}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["tables", "figures", "numbers"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if args.only in (None, "tables"):
        print_tables()
    if args.only in (None, "tables", "numbers"):
        print_numbers()
    if args.only in (None, "figures"):
        names = figures.build_all()
        print("\nFigures written to output/figures/: " + ", ".join(names))


if __name__ == "__main__":
    main()
