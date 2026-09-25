#!/usr/bin/env python3
"""Summarize learning-curve results (run_learning_curve.py) by training size.

Sibling of summarize_results.py, but for the extra `size` axis those runs add.
Same selector interface: pass one or more globs of *selector* files, each of
which names a condition (label).  Two kinds are recognized:

  - ``<label>.log``          (a run log)
  - ``<label>_config.yaml``  (the run-config every run writes)

The numbers come from the sibling ``results.csv`` (authoritative, per-seed),
whose rows are ``label,dataset,size,seed,bestj,err``.  So:

    uv run python scripts/summarize_learning_curve.py "scripts/lc-out/*_config.yaml"

quote the glob so the script (not the shell) expands it.

Two views are printed:

  1. HEADLINE -- macro test error by (label x size): rows are labels, columns
     are training sizes, each cell ``mean +/- SEM`` of the per-dataset means at
     that size (equal-weight macro, SEM propagated as in summarize_results).
     A final ``avg rank`` column gives each model's mean rank (1 = best) over
     every (size,dataset) cell -- a scale-robust score that a single easy or
     hard dataset can't dominate.  When exactly two labels are selected a
     ``delta`` row shows the second minus the first -- this is the "does
     slopper's edge grow as n shrinks?" answer.

Macro and rank are computed on a COMMON BASIS: at each size, only the datasets
every selected label has finished.  A curve that is still running has done only
part of the suite, and averaging it over just those datasets is not comparable
to a finished label's average over all of them -- if it happened to finish the
easy datasets first, its macro looks unfairly good.  ``(D)`` marks a restricted
basis and the note under the table lists what was excluded.  ``--macro-all``
restores the old per-label-coverage behaviour.  Per-dataset columns are never
restricted; only the summary statistics are.

  2. PER-SIZE SUBTABLES -- one table per training size, laid out exactly like
     summarize_results.py: rows are models (labels), columns are datasets, and
     ``macro`` + ``rank`` columns on the right (row mean of per-dataset means,
     and mean rank across that size's datasets).  Rows are sorted by macro
     (lower test error first), so as more models are added each size's table
     ranks them directly.  ``--no-detail`` suppresses these.  ``--bestj`` adds a
     mean-bestj-by-size table (the round budget slipper-default CV picked).

Ranks are computed per (size,dataset) cell over the models present there (ties
averaged); a cell with fewer than two models is skipped so a lone model isn't
handed a free rank of 1.

``--min-seeds N`` drops any (dataset,size) cell backed by fewer than N seeds so
half-finished curves don't show noisy single-seed cells.  ``--markdown`` emits
GitHub tables instead of aligned plain text.
"""
import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import DATASETS  # canonical dataset order

ABBREV = {
    "hypothyroid": "hypothy", "breast_cancer_wisconsin": "breast",
    "cylinder_bands": "cylinder", "pima_diabetes": "pima", "titanic": "titanic",
    "abalone": "abalone", "german_credit": "german", "obesity": "obesity",
    "student_por": "student", "move1": "move1",
}


def abbrev(ds):
    return ABBREV.get(ds, ds[:8])


def label_of(path):
    """The condition label a selector file names, or None if it isn't one."""
    b = os.path.basename(path)
    if b.endswith("_config.yaml"):
        return b[:-len("_config.yaml")]
    if b.endswith(".log"):
        return b[:-len(".log")]
    return None


def load_csv(path):
    """(label, dataset, size) -> {seed: err} and (label, dataset, size) ->
    [bestj] from a learning-curve results.csv."""
    errs, bestj = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["label"], r["dataset"], int(r["size"]))
            errs.setdefault(key, {})[int(r["seed"])] = float(r["err"])
            bestj.setdefault(key, []).append(int(r["bestj"]))
    return errs, bestj


def mean_sem(vals):
    """(mean, SEM, n) for a list; SEM = sd/sqrt(n)."""
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


def macro(per_ds):
    """Equal-weight macro over a {dataset: (mean, sem, n)} dict: mean of the
    per-dataset means, SEM propagated with weights 1/D.  Returns (mean, sem, D)
    or None if empty."""
    if not per_ds:
        return None
    means = [s[0] for s in per_ds.values()]
    sems = [s[1] for s in per_ds.values()]
    D = len(means)
    mvar = sum((sem / D) ** 2 for sem in sems)
    return sum(means) / D, math.sqrt(mvar), D


def common_datasets(labels, stats, z):
    """Datasets present for EVERY label that has any data at size `z`.

    The macro basis.  A label whose curve is still running has only some of the
    suite finished, and its macro over just those datasets is not comparable to
    a finished label's macro over all of them -- an in-progress row that happens
    to have done the easy datasets first looks artificially good.  Restricting
    every row to the intersection keeps the column apples-to-apples.  Labels
    absent from this size entirely don't shrink the set."""
    sets = [set(stats[l][z]) for l in labels if stats[l].get(z)]
    return set.intersection(*sets) if sets else set()


def union_datasets(labels, stats, z):
    return {d for l in labels for d in stats[l].get(z, {})}


def curve_basis(labels, stats, sizes, fixed=False):
    """A common dataset set for every curve at a size, optionally across sizes.

    Missing results must not change a learner's macro denominator relative to
    its competitors.  A fixed curve also needs every retained dataset at every
    plotted size, not merely in the union at the largest size.
    """
    basis = {z: common_datasets(labels, stats, z) for z in sizes}
    if fixed and basis:
        keep = set.intersection(*basis.values())
        basis = {z: keep.copy() for z in sizes}
    return basis


def rank_in_cell(err_by_label):
    """{label: mean_err} -> {label: rank}, 1 = lowest error (best), ties get the
    average of the ranks they span (fractional ranking)."""
    order = sorted(err_by_label.items(), key=lambda kv: kv[1])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        avg = (i + j) / 2 + 1                      # ranks i+1..j+1, averaged
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg
        i = j + 1
    return ranks


def avg_ranks(labels, stats, cells):
    """Mean rank per label over `cells` (a list of (size, dataset)).  A cell is
    only ranked where >= 2 labels have a value there, so a lone model isn't
    handed a free rank of 1.  Returns {label: (mean_rank, n_cells_ranked)}."""
    acc = {l: [] for l in labels}
    for z, d in cells:
        errs = {l: stats[l][z][d][0] for l in labels
                if stats[l].get(z, {}).get(d) is not None}
        if len(errs) < 2:
            continue
        for l, rk in rank_in_cell(errs).items():
            acc[l].append(rk)
    return {l: (sum(v) / len(v), len(v)) for l, v in acc.items() if v}


def collect(globs, min_seeds=1):
    """Resolve selector globs to per-condition stats.  Returns

        labels  -- condition labels, first-seen order (only ones with results)
        stats   -- stats[label][size][dataset] = (mean, sem, n_seeds)
        bj      -- bj[label][size][dataset]    = mean bestj
        sizes   -- sorted training sizes seen

    (dataset,size) cells backed by fewer than `min_seeds` seeds are dropped.
    Shared with plot_learning_curve.py, which takes the same selector args."""
    selectors = []
    for pat in globs:
        selectors.extend(glob.glob(pat))
    if not selectors:
        sys.exit("no selector files matched: %s" % " ".join(globs))

    # selector -> label -> results.csv beside it (dedupe, preserve first-seen).
    labels, label_csv, seen = [], {}, set()
    for sel in sorted(selectors):
        label = label_of(sel)
        if label is None or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        label_csv[label] = os.path.join(os.path.dirname(sel), "results.csv")

    csv_cache = {}
    for path in set(label_csv.values()):
        if os.path.exists(path):
            csv_cache[path] = load_csv(path)
        else:
            print("// warning: no results.csv at %s" % path, file=sys.stderr)
            csv_cache[path] = ({}, {})

    stats, bj, sizes_seen = {}, {}, set()
    for label in labels:
        errs, bestj = csv_cache[label_csv[label]]
        st, bjl = {}, {}
        for (lab, ds, sz), seedmap in errs.items():
            if lab != label:
                continue
            s = mean_sem(list(seedmap.values()))
            if s is None or s[2] < min_seeds:
                continue
            st.setdefault(sz, {})[ds] = s
            b = bestj[(lab, ds, sz)]
            bjl.setdefault(sz, {})[ds] = sum(b) / len(b)
            sizes_seen.add(sz)
        stats[label] = st
        bj[label] = bjl
    labels = [l for l in labels if stats[l]]
    if not labels:
        sys.exit("no results found for any selected label")
    return labels, stats, bj, sorted(sizes_seen)


def render(headers, matrix, markdown, align_first_left=True):
    """Print a table (list-of-str rows) plain-aligned or as GitHub markdown."""
    if markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for r in matrix:
            print("| " + " | ".join(r) + " |")
        return
    widths = [max(len(headers[c]), max((len(r[c]) for r in matrix), default=0))
              for c in range(len(headers))]

    def fmt(r):
        first = r[0].ljust(widths[0]) if align_first_left else r[0].rjust(widths[0])
        return "  ".join([first] + [r[c].rjust(widths[c]) for c in range(1, len(r))])

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in matrix:
        print(fmt(r))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", metavar="GLOB",
                    help="glob(s) of selector files; each <label>.log or "
                         "<label>_config.yaml selects a condition")
    ap.add_argument("--min-seeds", type=int, default=1, metavar="N",
                    help="drop (dataset,size) cells backed by fewer than N seeds")
    ap.add_argument("--markdown", action="store_true",
                    help="emit GitHub-markdown tables instead of aligned text")
    ap.add_argument("--no-detail", action="store_true",
                    help="only print the headline macro-by-size table")
    ap.add_argument("--bestj", action="store_true",
                    help="also print a mean-bestj-by-size table (round budget)")
    ap.add_argument("--macro-all", action="store_true",
                    help="macro/rank each label over ALL datasets it has, not "
                         "just the ones common to every label (the old, "
                         "not-apples-to-apples behaviour)")
    args = ap.parse_args()

    labels, stats, bj, sizes = collect(args.globs, args.min_seeds)

    # Macro/rank basis per size: the datasets every selected label has finished,
    # so a still-running label can't flatter itself with a partial average.
    basis = {z: (union_datasets(labels, stats, z) if args.macro_all
                 else common_datasets(labels, stats, z)) for z in sizes}
    dropped = {z: sorted(union_datasets(labels, stats, z) - basis[z]) for z in sizes}

    def macro_at(label, z):
        return macro({d: s for d, s in stats[label].get(z, {}).items()
                      if d in basis[z]})

    # ---- HEADLINE: macro test error by (label x size), plus an overall
    # average rank per model (across every (size,dataset) cell on the basis). ----
    all_cells = [(z, d) for z in sizes for d in basis[z]]
    overall_rank = avg_ranks(labels, stats, all_cells)
    print("\n## macro test error by training size  (mean +/- SEM over datasets)\n")
    headers = ["label"] + [f"n={z}" for z in sizes] + ["avg rank"]
    matrix = []
    # best (lowest) average rank first; unranked labels last
    ordered = sorted(labels, key=lambda l: overall_rank.get(l, (float("inf"),))[0])
    for label in ordered:
        row = [label]
        for z in sizes:
            m = macro_at(label, z)
            n_all = len(union_datasets(labels, stats, z))
            row.append("--" if m is None
                       else "%.2f±%.2f%s" % (m[0], m[1],
                                             "" if m[2] == n_all else "(%d)" % m[2]))
        rk = overall_rank.get(label)
        row.append("--" if rk is None else "%.2f" % rk[0])
        matrix.append(row)
    # delta row (on the same basis) when exactly two labels
    if len(labels) == 2:
        a, b = ordered
        drow = ["Δ(%s-%s)" % (b, a)]
        for z in sizes:
            ma, mb = macro_at(a, z), macro_at(b, z)
            drow.append("--" if ma is None or mb is None
                        else "%+.2f" % (mb[0] - ma[0]))
        drow.append("--")
        matrix.append(drow)
    render(headers, matrix, args.markdown)
    print("\ncell = macro over datasets (equal-weight mean of per-dataset means) "
          "+/- propagated SEM. delta = second label minus first. "
          "avg rank = mean rank (1=best) over every (size,dataset) cell with >=2 models.")
    if args.macro_all:
        print("--macro-all: each row averaged over whatever datasets IT has, so "
              "rows with different coverage are NOT comparable; (D) marks how many.")
    else:
        print("macro/rank use only the datasets COMMON to every selected label at "
              "that size, so a half-finished curve stays comparable; (D) marks a "
              "restricted basis. Excluded: %s"
              % ("; ".join("n=%d: %s" % (z, ", ".join(dropped[z]))
                           for z in sizes if dropped[z]) or "none"))

    if args.no_detail:
        return

    # ---- PER-SIZE SUBTABLES: one table per size, rows=models, cols=datasets,
    # macro on the right (exactly the summarize_results.py layout).  Rows sorted
    # by macro so the models rank directly within each size. ----
    for z in sizes:
        present = sorted({d for label in labels for d in stats[label].get(z, {})},
                         key=lambda d: DATASETS.index(d) if d in DATASETS else 99)
        rows_at_z = [label for label in labels if stats[label].get(z)]
        # macro/rank on the common basis (see the headline note); the per-dataset
        # columns still show everything a label has.
        macros = {label: macro_at(label, z) for label in rows_at_z}
        rows_at_z.sort(key=lambda label: (macros[label] or (float("inf"),))[0])
        ranks = avg_ranks(rows_at_z, stats, [(z, d) for d in basis[z]])

        print("\n## n=%d  (test err %%, mean +/- SEM over seeds)\n" % z)
        headers = ["model"] + [abbrev(d) for d in present] + ["macro", "rank"]
        matrix = []
        for label in rows_at_z:
            st = stats[label][z]
            row = [label]
            for d in present:
                s = st.get(d)
                row.append("--" if s is None
                           else ("%.2f±%.2f" % (s[0], s[1]) if s[2] > 1
                                 else "%.2f" % s[0]))
            if macros[label] is None:
                row.append("--")
            else:
                mm, ms, D = macros[label]
                row.append("%.2f±%.2f%s"
                           % (mm, ms, "" if D == len(present) else " (%d)" % D))
            rk = ranks.get(label)
            row.append("--" if rk is None else "%.2f" % rk[0])
            matrix.append(row)
        render(headers, matrix, args.markdown)

    if not args.bestj:
        return

    # ---- mean bestj by size (the slipper-default CV round budget; ~identical
    # across models since it's recomputed on the same subset). ----
    present = sorted({d for label in labels for z in bj[label] for d in bj[label][z]},
                     key=lambda d: DATASETS.index(d) if d in DATASETS else 99)
    print("\n## mean bestj by size  (round budget from slipper-default CV)\n")
    headers = ["dataset"] + [f"n={z}" for z in sizes]
    matrix = []
    for ds in present:
        row = [ds]
        for z in sizes:
            vals = [bj[label].get(z, {}).get(ds) for label in labels]
            vals = [v for v in vals if v is not None]
            row.append("--" if not vals else "%.0f" % (sum(vals) / len(vals)))
        matrix.append(row)
    render(headers, matrix, args.markdown)


if __name__ == "__main__":
    main()
