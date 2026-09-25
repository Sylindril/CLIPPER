#!/usr/bin/env python3
"""shyp_size.py - count the rules in a slipper `.shyp` hypothesis.

The learning-curve harnesses record test error but (until now) not model size,
so this recovers the size of any run whose cell directory survives.  Also used
by run_learning_curve.py to fill the `rules` column going forward.

What counts as a rule
---------------------
A `.shyp` holds one rule per line, then a lone `.` terminator and a couple of
trailer lines (`use_best_rule`, `epsilon ...`).  Two rule forms appear:

    income=>50K 0.125 0.001 IF capital-gain >= 7298 .      # grammar rule
    pyrule survived 0.211 0.006 SOURCE "def covers(x)..."  # LLM PyRule

The final grammar line is the **default rule** -- an empty antecedent (`IF .`)
that fires on everything and just carries the class prior.  It is not a learned
rule, so it is excluded: a hypothesis that learned nothing reports 0, matching
the `Rules` column of the results tables (e.g. `stroke` at full size).

Note antecedent *conditions* are deliberately not counted -- symbolic values can
contain spaces (`InternetService = Fiber optic`), so token-splitting misreports
them, and PyRules have no comparable notion of a condition anyway.

Usage::

    uv run python scripts/shyp_size.py FILE.shyp ...          # one row per file
    uv run python scripts/shyp_size.py --cells scripts/test-lc-out/cells
"""
import argparse
import csv
import os
import re
import sys

# "<label>__<dataset>__<seed>" cell dirs under ".../cells/size.<N>/"
_CELL_RE = re.compile(r"size\.(\d+)[/\\]([^/\\]+)__([^/\\]+)__(\d+)[/\\]")


def count_rules(path):
    """Number of learned rules in a .shyp (PyRules included, default excluded)."""
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("pyrule "):
                n += 1
            elif " IF " in line:
                # grammar rule: antecedent is between " IF " and the trailing "."
                if line.split(" IF ", 1)[1].strip(" .").strip():
                    n += 1
    return n


def walk_cells(root):
    """Yield (label, dataset, size, seed, rules) for every .shyp under a cells/ tree."""
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".shyp"):
                continue
            path = os.path.join(dirpath, fn)
            m = _CELL_RE.search(path + os.sep)
            if not m:
                continue
            size, label, ds, seed = m.groups()
            yield label, ds, int(size), int(seed), count_rules(path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".shyp files to count")
    ap.add_argument("--cells", metavar="DIR",
                    help="walk a harness cells/ tree and emit CSV to stdout")
    args = ap.parse_args()

    if args.cells:
        w = csv.writer(sys.stdout)
        w.writerow(["label", "dataset", "size", "seed", "rules"])
        for row in sorted(walk_cells(args.cells)):
            w.writerow(row)
    elif args.files:
        for p in args.files:
            print("%6d  %s" % (count_rules(p), p))
    else:
        ap.error("give .shyp files or --cells DIR")


if __name__ == "__main__":
    main()
