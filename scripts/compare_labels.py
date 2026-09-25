#!/usr/bin/env python3
"""Compare two bestj-suite conditions from scripts/bestj-out/results.csv.

Prints a per-dataset test-error table (mean +/- sd) for a NEW label next to a
BASE label, with the delta, restricted to datasets the NEW label has finished.
Handy while a run is still filling in: shows only what's comparable so far.

    uv run python scripts/compare_labels.py --new tv2_lambda_toolsv1 --base tv2_lambda

--complete-only (default) shows only datasets with all --seeds seeds for NEW;
--partial also lists datasets still in progress (n < seeds), excluded from the
macro average.  The macro row averages per-dataset means over the datasets
counted (both conditions on the same dataset set, so the delta is apples-to-apples).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import BESTJ, DATASETS, REPO

DEFAULT_CSV = os.path.join(REPO, "scripts", "bestj-out", "results.csv")


def load(csv_path, label):
    """label -> {dataset: {seed: err}}."""
    out = {}
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            if r["label"] == label:
                out.setdefault(r["dataset"], {})[int(r["seed"])] = float(r["err"])
    return out


def stat(errs):
    v = np.array(list(errs.values()))
    if len(v) == 0:
        return None
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return v.mean(), sd, len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="in-progress / new label")
    ap.add_argument("--base", default="tv2_lambda", help="baseline label (default tv2_lambda)")
    ap.add_argument("--seeds", type=int, default=10, help="seeds per dataset (default 10)")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--partial", action="store_true",
                    help="also list datasets still in progress (n < seeds)")
    args = ap.parse_args()

    new, base = load(args.csv, args.new), load(args.csv, args.base)
    print(f"\n## `{args.new}` vs `{args.base}`  (test err %, completed datasets)\n")
    print(f"| Dataset | {args.new} | {args.base} | Δ |")
    print("|---|---|---|---:|")

    new_macro, base_macro, deltas = [], [], []
    pending = []
    for ds in DATASETS:
        sn = stat(new.get(ds, {}))
        if sn is None:
            continue
        nmean, nsd, nn = sn
        if nn < args.seeds:
            pending.append((ds, nn))
            continue
        sb = stat(base.get(ds, {}))
        base_cell = f"{sb[0]:.2f} ± {sb[1]:.2f}" if sb else "—"
        delta = f"{nmean - sb[0]:+.2f}" if sb else "—"
        print(f"| {ds} | {nmean:.2f} ± {nsd:.2f} (n={nn}) | {base_cell} | {delta} |")
        new_macro.append(nmean)
        if sb:
            base_macro.append(sb[0])
            deltas.append(nmean - sb[0])

    if new_macro:
        bm = f"{np.mean(base_macro):.2f}" if base_macro else "—"
        dm = f"{np.mean(deltas):+.2f}" if deltas else "—"
        print(f"| **macro ({len(new_macro)} done)** | **{np.mean(new_macro):.2f}** | **{bm}** | **{dm}** |")
    else:
        print("| _(no datasets fully complete yet)_ |||| ")

    if args.partial and pending:
        print("\nin progress: " + ", ".join(f"{d} ({n}/{args.seeds})" for d, n in pending))


if __name__ == "__main__":
    main()
