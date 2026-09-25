#!/usr/bin/env python3
"""Paired significance test between two learning-curve labels.

Sibling of compare_labels.py (which reads bestj-out and prints no p-values) for
the extra `size` axis of scripts/lc-out/results.csv.  Cells are paired on
(dataset, size, seed), so both labels must have been run on the same grid.

    uv run python scripts/paired_lc.py --new slopper_notools_M200_mb25 \
                                       --base slopper_notools_M200

Two levels are reported, matching how conditions are written up in
the development-suite comparisons:

  - per (size, dataset): paired t-test over the seeds, the fine-grained view.
  - per size: paired t-test over the per-dataset means (n = #datasets), the
    "macro delta" quoted in the condition sections, plus a sign count (how many
    datasets NEW wins) since a single high-variance dataset can carry a macro.

Significance marks follow run_format_grid._paired: *** p<.001, ** p<.01,
* p<.05, else n.s.  Negative delta = NEW has lower test error = NEW is better.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import REPO, SUITES

DEFAULT_CSV = os.path.join(REPO, "scripts", "lc-out", "results.csv")


def load(csv_path, label):
    """label -> {(dataset, size, seed): err}."""
    out = {}
    with open(csv_path) as fp:
        for r in csv.DictReader(fp):
            if r["label"] == label:
                out[(r["dataset"], int(r["size"]), int(r["seed"]))] = float(r["err"])
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
    ap.add_argument("--sizes", help="comma-separated subset of sizes (default: all common)")
    args = ap.parse_args()

    new, base = load(args.csv, args.new), load(args.csv, args.base)
    if not new or not base:
        sys.exit("no rows for %s" % (args.new if not new else args.base))
    common = set(new) & set(base)
    if not common:
        sys.exit("labels share no (dataset,size,seed) cells")
    sizes = sorted({s for _, s, _ in common})
    if args.sizes:
        want = {int(x) for x in args.sizes.split(",")}
        sizes = [s for s in sizes if s in want]

    dropped = (len(new) - len(common), len(base) - len(common))
    if any(dropped):
        print("note: %d NEW / %d BASE cells unpaired and ignored\n" % dropped)

    print("paired test: NEW=%s vs BASE=%s   (delta = NEW - BASE, negative = NEW better)"
          % (args.new, args.base))

    # Order datasets by whichever registered suite they belong to, then append
    # any strays alphabetically.  Iterating one suite's fixed list silently
    # printed empty tables for every other suite's results.
    present = {d for (d, _, _) in common}
    order = [d for s in SUITES.values() for d in s.datasets if d in present]
    order += sorted(present - set(order))
    if not order:
        sys.exit("no cells common to %r and %r in %s -- check the labels and "
                 "that both were run on the same suite" % (args.new, args.base, args.csv))

    macro = []
    for size in sizes:
        print("\n## n=%d\n" % size)
        print("%-26s %9s %9s %8s %7s %s"
              % ("dataset", "NEW", "BASE", "delta", "p", "sig"))
        print("-" * 68)
        dmeans_new, dmeans_base = [], []
        for ds in order:
            seeds = sorted(s for (d, z, s) in common if d == ds and z == size)
            if not seeds:
                continue
            a = np.array([new[(ds, size, s)] for s in seeds])
            b = np.array([base[(ds, size, s)] for s in seeds])
            p, sig = paired(a, b)
            dmeans_new.append(a.mean())
            dmeans_base.append(b.mean())
            print("%-26s %9.2f %9.2f %+8.2f %7s %s"
                  % (ds, a.mean(), b.mean(), a.mean() - b.mean(), fmt_p(p), sig))
        if len(dmeans_new) < 2:
            continue
        a, b = np.array(dmeans_new), np.array(dmeans_base)
        p, sig = paired(a, b)
        wins = int((a < b).sum())
        print("-" * 68)
        print("%-26s %9.2f %9.2f %+8.2f %7s %s   (NEW better on %d/%d datasets)"
              % ("macro (paired over datasets)", a.mean(), b.mean(),
                 a.mean() - b.mean(), fmt_p(p), sig, wins, len(a)))
        macro.append((size, a, b))

    if len(macro) > 1:
        a = np.concatenate([x for _, x, _ in macro])
        b = np.concatenate([x for _, _, x in macro])
        p, sig = paired(a, b)
        print("\n## all sizes pooled  (%d dataset x size pairs)\n" % len(a))
        print("%-26s %9.2f %9.2f %+8.2f %7s %s   (NEW better on %d/%d)"
              % ("macro", a.mean(), b.mean(), a.mean() - b.mean(), fmt_p(p), sig,
                 int((a < b).sum()), len(a)))


if __name__ == "__main__":
    main()
