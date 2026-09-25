#!/usr/bin/env python3
"""Summarize bestj-suite results as a clean label x dataset table with stderrs.

Invoke with one or more globs of *selector* files; each selects a condition by
name.  Two kinds are recognized:

  - ``<label>.log``          (a run log)
  - ``<label>_config.yaml``  (the run-config every run_bestj writes)

The ``_config.yaml`` form is the robust one: run_bestj always writes it, even
for runs whose stdout went to a shared/combined log (so they have no
``<label>.log``).  The numbers themselves come from the sibling ``results.csv``
(authoritative, per-seed) -- NOT the selector file.  So:

    uv run python scripts/summarize_results.py "scripts/bestj-out/*_config.yaml"
    uv run python scripts/summarize_results.py "scripts/bestj-out/*.log"

quote the glob so the script expands it (and so it can't be pre-expanded to a
different set by the shell).

A **learning-curve** results.csv (``scripts/lc-out/``, ``scripts/test-lc-out/``)
has an extra ``size`` column and one row per (label, dataset, seed, size), so
``--size N`` is required to get a meaningful table -- it restricts the summary to
the cells at training size N:

    uv run python scripts/summarize_results.py --size 250 \
        "scripts/test-lc-out/*_config.yaml"

Without it the repeated seeds collapse last-row-wins (each cell is then an
arbitrary size), so the script warns.  For the whole curve at once use
``summarize_learning_curve.py``, which tabulates every size.

Table: rows are labels, columns are datasets, each cell is ``mean +/- SEM``
over that condition's seeds (SEM = sample sd / sqrt(n), the standard error of
the mean).  Missing cells show ``--``.  ``--min-seeds N`` drops any cell backed
by fewer than N seeds (and any label left with no cells), so half-finished or
single-seed probes don't clutter the table.

The final **macro** column is the unweighted mean of that row's per-dataset
means, with the error propagated from the per-dataset SEMs: treating the macro
as sum_d w_d * mean_d with equal weights w_d = 1/D, its variance is
sum_d w_d^2 * SEM_d^2, and the reported stderr is the square root of that
("weighted stderr = sqrt of the weighted variance", weights 1/D).  Rows are
sorted by macro ascending (lower test error first).

A final **rank** column (after macro) gives each label's mean rank (1 = best,
lowest error) over the datasets where at least two labels are present, with ties
averaged -- a scale-robust score that a single easy or hard dataset can't
dominate, mirroring summarize_learning_curve.py's ``avg rank``.

Rendering: both formats give every numeric column the same width, so the table
reads as a grid in the source as well as when rendered.  ``--markdown`` emits a
GitHub table and **bolds the smallest value in each column** (datasets, macro
and rank alike), comparing on the 2-decimal rounding that is displayed, so a
visible tie bolds every cell in it.
"""
import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import DATASETS  # canonical column order

# Short column headers so a 10-dataset table stays readable; unknown datasets
# fall back to a truncated name.
ABBREV = {
    "hypothyroid": "hypothy", "breast_cancer_wisconsin": "breast",
    "cylinder_bands": "cylinder", "pima_diabetes": "pima", "titanic": "titanic",
    "abalone": "abalone", "german_credit": "german", "obesity": "obesity",
    "student_por": "student", "move1": "move1",
}


def abbrev(ds):
    return ABBREV.get(ds, ds[:8])


def label_of(path):
    """The condition label a selector file names, or None if it isn't one.
    Accepts ``<label>_config.yaml`` (preferred) and ``<label>.log``."""
    b = os.path.basename(path)
    if b.endswith("_config.yaml"):
        return b[:-len("_config.yaml")]
    if b.endswith(".log"):
        return b[:-len(".log")]
    return None


def load_csv(path, size=None):
    """(label, dataset) -> {seed: err} from a results.csv, plus the set of
    training sizes seen.

    A bestj-suite results.csv has one row per (label, dataset, seed), so the
    seed key is unique.  A learning-curve results.csv carries an extra ``size``
    column and repeats that triple once per size -- without a selection the
    later row silently wins.  ``size`` keeps only rows at that training size;
    the returned size set lets the caller warn when a multi-size csv is read
    without one."""
    out = {}
    sizes = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            sz = r.get("size")
            if sz:
                sizes.add(int(sz))
                if size is not None and int(sz) != size:
                    continue
            elif size is not None:      # no size column -> nothing to select
                continue
            out.setdefault((r["label"], r["dataset"]), {})[int(r["seed"])] = float(r["err"])
    return out, sizes


def mean_sem(errs):
    """(mean, SEM, n) for a list of per-seed errors; SEM = sd/sqrt(n)."""
    n = len(errs)
    if n == 0:
        return None
    m = sum(errs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((e - m) ** 2 for e in errs) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", metavar="GLOB",
                    help="glob(s) of selector files; each <label>.log or "
                         "<label>_config.yaml selects a condition")
    ap.add_argument("--min-seeds", type=int, default=1, metavar="N",
                    help="drop cells backed by fewer than N seeds (default 1)")
    ap.add_argument("--min-cols", type=int, default=1, metavar="N",
                    help="drop labels covering fewer than N datasets (default 1); "
                         "skips low-coverage experiments like single-dataset probes")
    ap.add_argument("--size", type=int, metavar="N",
                    help="learning-curve results.csv only: summarize the cells "
                         "at training size N.  Without it a multi-size csv is "
                         "collapsed by last-row-wins (see load_csv), so pass "
                         "this whenever the selectors point at a *lc-out/ dir")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a GitHub-markdown table instead of aligned plain text")
    args = ap.parse_args()

    # Expand globs -> selector files -> (label, results.csv path). The label is
    # the selector basename minus its .log / _config.yaml suffix; the data lives
    # in results.csv beside the selector.
    selectors = []
    for pat in args.globs:
        selectors.extend(glob.glob(pat))
    if not selectors:
        sys.exit("no selector files matched: %s" % " ".join(args.globs))

    labels = []            # preserve first-seen; dedupe
    label_csv = {}
    seen = set()
    for sel in sorted(selectors):
        label = label_of(sel)
        if label is None:
            continue
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        label_csv[label] = os.path.join(os.path.dirname(sel), "results.csv")

    # Load every referenced results.csv once.
    csv_cache = {}
    for path in set(label_csv.values()):
        if os.path.exists(path):
            csv_cache[path], sizes = load_csv(path, args.size)
            if args.size is not None and not sizes:
                sys.exit("%s has no `size` column -- --size only applies to a "
                         "learning-curve results.csv" % path)
            if args.size is not None and args.size not in sizes:
                sys.exit("no size %d in %s (have: %s)"
                         % (args.size, path, ", ".join(str(s) for s in sorted(sizes))))
            if args.size is None and len(sizes) > 1:
                print("// warning: %s holds %d training sizes (%s) and no --size "
                      "was given; each (label,dataset,seed) collapses to its "
                      "last row, which is NOT a per-size or pooled mean"
                      % (path, len(sizes), ", ".join(str(s) for s in sorted(sizes))),
                      file=sys.stderr)
        else:
            print("// warning: no results.csv at %s" % path, file=sys.stderr)
            csv_cache[path] = {}

    # Per (label, dataset) stats; collect which datasets actually appear.
    stats = {}             # label -> {dataset -> (mean, sem, n)}
    present = set()
    missing_labels = []
    for label in labels:
        data = csv_cache[label_csv[label]]
        row = {}
        for (lab, ds), seedmap in data.items():
            if lab != label:
                continue
            s = mean_sem(list(seedmap.values()))
            if s is not None and s[2] >= args.min_seeds:   # s[2] == n seeds
                row[ds] = s
                present.add(ds)
        if row:
            stats[label] = row
        else:
            missing_labels.append(label)

    if not stats:
        sys.exit("no results found for any selected label (%s)" % ", ".join(labels))
    for lab in missing_labels:
        why = ("no cells with >= %d seeds" % args.min_seeds
               if args.min_seeds > 1 else "no rows in results.csv")
        print("// warning: label %r skipped (%s)" % (lab, why), file=sys.stderr)

    # Drop low-coverage experiments (fewer than --min-cols datasets), then
    # recompute the column set from the survivors so a column contributed only
    # by a dropped row doesn't linger as an all-'--' stripe.
    if args.min_cols > 1:
        for lab in [l for l in stats if len(stats[l]) < args.min_cols]:
            print("// note: label %r skipped (covers %d < %d datasets)"
                  % (lab, len(stats[lab]), args.min_cols), file=sys.stderr)
            del stats[lab]
        if not stats:
            sys.exit("no label covers >= %d datasets" % args.min_cols)
    present = set()
    for row in stats.values():
        present.update(row)

    # Column order: canonical datasets first, then any extras alphabetically.
    cols = [d for d in DATASETS if d in present] + \
           sorted(d for d in present if d not in DATASETS)

    # Macro per label: unweighted mean of dataset means; stderr propagated from
    # per-dataset SEMs with equal weights 1/D.
    macro = {}
    for label, row in stats.items():
        means = [row[d][0] for d in cols if d in row]
        sems = [row[d][1] for d in cols if d in row]
        D = len(means)
        mmean = sum(means) / D
        mvar = sum((sem / D) ** 2 for sem in sems)   # sum_d (w_d*SEM_d)^2, w=1/D
        macro[label] = (mmean, math.sqrt(mvar), D)

    order = sorted(stats, key=lambda l: macro[l][0])   # best (lowest) first

    # Average rank: for each dataset rank the labels present there by mean error
    # (1 = best/lowest, ties averaged); a label's rank is the mean of its
    # per-dataset ranks. Datasets with fewer than two labels are skipped (a lone
    # label can't be ranked), so a label ranked nowhere shows '--'. Scale-robust
    # complement to macro -- a single easy/hard dataset can't dominate it.
    def ranks_in(pairs):   # pairs: [(label, mean)] -> {label: rank}, ties averaged
        out = {}
        ordered = sorted(pairs, key=lambda x: x[1])
        i, n = 0, len(ordered)
        while i < n:
            j = i
            while j < n and ordered[j][1] == ordered[i][1]:
                j += 1
            avg = sum(range(i + 1, j + 1)) / (j - i)   # mean of 1-based positions
            for k in range(i, j):
                out[ordered[k][0]] = avg
            i = j
        return out

    rank_lists = {label: [] for label in stats}
    for d in cols:
        here = [(label, stats[label][d][0]) for label in stats if d in stats[label]]
        if len(here) < 2:
            continue
        for label, rk in ranks_in(here).items():
            rank_lists[label].append(rk)
    avg_rank = {label: (sum(v) / len(v) if v else None)
                for label, v in rank_lists.items()}

    # ---- build the string matrix once, then render plain or markdown ----
    def cell(s):
        return "--" if s is None else "%.2f±%.2f" % (s[0], s[1])

    headers = ["label"] + [abbrev(d) for d in cols] + ["macro", "rank"]
    matrix = []            # rows of cell strings
    values = []            # parallel rows of the numbers those cells show
    for label in order:
        row = stats[label]
        mm, ms, D = macro[label]
        macro_cell = "%.2f±%.2f" % (mm, ms) + ("" if D == len(cols) else " (%d)" % D)
        rk = avg_rank[label]
        rank_cell = "--" if rk is None else "%.2f" % rk
        matrix.append([label] + [cell(row.get(d)) for d in cols]
                      + [macro_cell, rank_cell])
        values.append([None] + [row[d][0] if d in row else None for d in cols]
                      + [mm, rk])

    # Per-column winner = the smallest value, compared on the DISPLAYED
    # 2-decimal rounding rather than the raw float, so two cells that print
    # identically are both marked (bolding one of a visible tie reads as a bug).
    # The label column (index 0) has no value and is never marked.  A row
    # covering only D < len(cols) datasets still competes for the macro column;
    # its macro is a mean over fewer datasets, which the `(D)` suffix flags.
    best = []
    for c in range(len(headers)):
        col = [round(r[c], 2) for r in values if r[c] is not None]
        best.append(min(col) if col else None)
    is_best = [[v is not None and best[c] is not None and round(v, 2) == best[c]
                for c, v in enumerate(r)] for r in values]

    legend = (("" if args.size is None else "training size n=%d.  " % args.size)
              + "cell = mean±SEM over seeds (SEM = sd/sqrt(n)). "
              "macro = mean of dataset means +/- sqrt(sum (SEM_d/D)^2); "
              "(D) marks a row covering only D of %d datasets. "
              "rank = mean rank (1=best) over datasets with >=2 labels present."
              % len(cols))

    # Markdown marks the per-column winners; the bold markers are part of the
    # cell text, so they count toward the column width below and the raw table
    # stays aligned in the source as well as when rendered.
    body = matrix if not args.markdown else \
        [["**%s**" % c if b else c for c, b in zip(r, bold)]
         for r, bold in zip(matrix, is_best)]

    # One width for every numeric column (the widest cell or header anywhere in
    # them) so the columns line up as a grid and the eye can scan across a row;
    # the label column keeps its own width, left-aligned.  Both renderers share
    # this -- only the separator between cells differs.
    nw = max([len(h) for h in headers[1:]] + [len(c) for r in body for c in r[1:]])
    widths = [max([len(headers[0])] + [len(r[0]) for r in body])] \
        + [nw] * (len(headers) - 1)

    def fmt(r, sep):
        return sep.join([r[0].ljust(widths[0])]
                        + [r[c].rjust(widths[c]) for c in range(1, len(r))])

    if args.markdown:
        print("| " + fmt(headers, " | ") + " |")
        print("| " + " | ".join("-" * w for w in widths) + " |")
        for r in body:
            print("| " + fmt(r, " | ") + " |")
    else:
        print(fmt(headers, "  "))
        print("  ".join("-" * w for w in widths))
        for r in body:
            print(fmt(r, "  "))
    print()
    print(legend)


if __name__ == "__main__":
    main()
