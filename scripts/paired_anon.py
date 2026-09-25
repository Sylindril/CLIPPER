#!/usr/bin/env python3
"""Paired test: a condition on the anonymized suite vs the same condition on
the natural test suite.

The anonymized results carry gensym dataset names (anon_suite/MAPPING.json maps
them back); cells are paired on (original dataset, seed) at one training size.
Both runs must share subsets and bestj -- the anonymization command in README.md guarantees that
via Suite.sample_key and --bestj-from; the `bestj` column reports mismatches.

Per dataset: mean±SEM of each side, the paired delta (anon - natural, positive
= anonymized is worse), a paired t-test and a Wilcoxon signed-rank test over the
seeds, and mean hypothesis size.  The macro row is a paired t-test over the
per-dataset means.  Output is a markdown table (readable as plain text).

    uv run python scripts/paired_anon.py
    uv run python scripts/paired_anon.py --label slopper_notools --size 1200
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_lc import fmt_p, paired
from suite import REPO, TEST


def load(path, label, size, rename=lambda d: d):
    """{(dataset, seed): row} for label at size, datasets passed through rename."""
    with open(path) as fp:
        return {(rename(r["dataset"]), int(r["seed"])): r for r in csv.DictReader(fp)
                if r["label"] == label and int(r["size"]) == size}


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="slopper", help="label on the natural side")
    ap.add_argument("--anon-label", default=None, help="label on the anon side (default: --label)")
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--natural-csv", default=os.path.join(REPO, "scripts", "test-lc-out", "results.csv"))
    ap.add_argument("--anon-csv", default=os.path.join(REPO, "scripts", "anon-lc-out", "results.csv"))
    ap.add_argument("--mapping", default=os.path.join(REPO, "anon_suite", "MAPPING.json"))
    args = ap.parse_args()

    with open(args.mapping) as fp:
        orig = {k: v["original"] for k, v in json.load(fp).items()}
    nat = load(args.natural_csv, args.label, args.size)
    ano = load(args.anon_csv, args.anon_label or args.label, args.size,
               lambda d: orig.get(d, d))
    common = set(nat) & set(ano)
    if not common:
        sys.exit("no paired cells for label %r at n=%d" % (args.label, args.size))
    unpaired = len(nat) + len(ano) - 2 * len(common)
    mismatch = sum(nat[k]["bestj"] != ano[k]["bestj"] for k in common)

    print("anon vs natural: label=%s n=%d  (delta = anon - natural, positive = anon worse)"
          % (args.label, args.size))
    print("%d paired cells, %d unpaired ignored, %d bestj mismatches\n"
          % (len(common), unpaired, mismatch))
    print("| dataset | natural | anon | delta | t-test p | Wilcoxon p | rules nat/anon |")
    print("|---|---|---|---|---|---|---|")
    present = {d for d, _ in common}
    order = [d for d in TEST.datasets if d in present] + sorted(present - set(TEST.datasets))
    mn, ma = [], []
    for ds in order:
        seeds = sorted(s for d, s in common if d == ds)
        a = np.array([float(ano[(ds, s)]["err"]) for s in seeds])
        b = np.array([float(nat[(ds, s)]["err"]) for s in seeds])
        p, sig = paired(a, b)
        wp = (float(stats.wilcoxon(a, b).pvalue)
              if len(a) > 1 and not np.allclose(a, b) else float("nan"))
        rules = lambda rows: np.mean([float(rows[(ds, s)]["rules"] or "nan") for s in seeds])
        mn.append(b.mean()); ma.append(a.mean())
        print("| %s | %.2f±%.2f | %.2f±%.2f | %+.2f±%.2f | %s %s | %s | %.1f / %.1f |"
              % (ds, b.mean(), sem(b), a.mean(), sem(a), (a - b).mean(), sem(a - b),
                 fmt_p(p).strip(), "" if sig == "n.s." else sig, fmt_p(wp).strip(),
                 rules(nat), rules(ano)))
    if len(mn) > 1:
        a, b = np.array(ma), np.array(mn)
        p, sig = paired(a, b)
        print("| **macro** | %.2f | %.2f | %+.2f | %s %s | | anon better on %d/%d |"
              % (b.mean(), a.mean(), a.mean() - b.mean(), fmt_p(p).strip(),
                 "" if sig == "n.s." else sig, int((a < b).sum()), len(a)))


if __name__ == "__main__":
    main()
