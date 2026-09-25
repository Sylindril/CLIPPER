#!/usr/bin/env python3
"""Paired significance test between two bestj-suite labels.

bestj-out sibling of paired_lc.py (which has the extra `size` axis).  Cells are
paired on (dataset, seed) -- both labels are run at each dataset's fixed bestj
budget over the same seeds, so a seed indexes the same stochastic draws for
both conditions.  compare_labels.py prints the same deltas without p-values;
this adds the paired t-tests.

    uv run python scripts/paired_bestj.py --new candcond_tool_T2 --base tv2_lambda_toolsv1

Two levels are reported, matching the development-suite comparisons:

  - per dataset: paired t-test over the seeds, the fine-grained view.
  - macro: paired t-test over the per-dataset means (n = #datasets), plus a
    sign count (how many datasets NEW wins).

Significance marks follow run_format_grid._paired: *** p<.001, ** p<.01,
* p<.05, else n.s.  Negative delta = NEW has lower test error = NEW is better.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import DATASETS, REPO

DEFAULT_CSV = os.path.join(REPO, "scripts", "bestj-out", "results.csv")


def load(csv_path, label):
    """label -> {(dataset, seed): err}."""
    out = {}
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            if r["label"] == label:
                out[(r["dataset"], int(r["seed"]))] = float(r["err"])
    return out


def paired(a, b):
    """Paired t-test of a vs b -> (p, sig); mirrors run_format_grid._paired."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), "?"
    if np.allclose(a, b):
        return 1.0, "n.s."
    p = float(stats.ttest_rel(a, b).pvalue)
    return p, ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s.")


def fmt_p(p):
    return "  —  " if np.isnan(p) else ("<.001" if p < .001 else "%.3f" % p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="label under test")
    ap.add_argument("--base", required=True, help="baseline label")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    new, base = load(args.csv, args.new), load(args.csv, args.base)
    if not new or not base:
        sys.exit("no rows for %s" % (args.new if not new else args.base))
    common = set(new) & set(base)
    if not common:
        sys.exit("labels share no (dataset,seed) cells")

    dropped = (len(new) - len(common), len(base) - len(common))
    if any(dropped):
        print("note: %d NEW / %d BASE cells unpaired and ignored\n" % dropped)

    print("paired test: NEW=%s vs BASE=%s   (delta = NEW - BASE, negative = NEW better)\n"
          % (args.new, args.base))
    print("%-26s %9s %9s %8s %7s %s"
          % ("dataset", "NEW", "BASE", "delta", "p", "sig"))
    print("-" * 68)
    dmeans_new, dmeans_base = [], []
    for ds in DATASETS:
        seeds = sorted(s for (d, s) in common if d == ds)
        if not seeds:
            continue
        a = np.array([new[(ds, s)] for s in seeds])
        b = np.array([base[(ds, s)] for s in seeds])
        p, sig = paired(a, b)
        dmeans_new.append(a.mean())
        dmeans_base.append(b.mean())
        print("%-26s %9.2f %9.2f %+8.2f %7s %s"
              % (ds, a.mean(), b.mean(), a.mean() - b.mean(), fmt_p(p), sig))
    if len(dmeans_new) < 2:
        return
    a, b = np.array(dmeans_new), np.array(dmeans_base)
    p, sig = paired(a, b)
    wins = int((a < b).sum())
    print("-" * 68)
    print("%-26s %9.2f %9.2f %+8.2f %7s %s   (NEW better on %d/%d datasets)"
          % ("macro (paired over datasets)", a.mean(), b.mean(),
             a.mean() - b.mean(), fmt_p(p), sig, wins, len(a)))


if __name__ == "__main__":
    main()
