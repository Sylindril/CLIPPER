#!/usr/bin/env python3
"""Run one slipper/slopper configuration across the whole tuning suite at each
dataset's fixed bestj round budget, N seeds, and print a per-dataset mean±sd
test-error table.

Each run:  slipper --seed S <extra flags> -i0 -n<bestj> <stem>

Pass the condition's slipper flags after ``--``; everything there is forwarded
verbatim.  Name the condition with ``--label`` (used as the results.csv key and
the table title); it defaults to a slug of the flags.

Examples:
    # discrete slipper2 at bestj (full sample), 10 seeds
    uv run python scripts/run_bestj.py --label discrete -- -R ''

    # discrete at bestj with M100 subsampling
    uv run python scripts/run_bestj.py --label discrete_M100 -- -R '' -M100

    # slopper (gemini, table examples + full schema) at bestj
    uv run python scripts/run_bestj.py --label slopper_table -- \
        -R gemini/gemini-3.1-flash-lite --example-format table

    # a subset of the suite: 3 seeds, everything except the slow move1
    uv run python scripts/run_bestj.py --seeds 3 --exclude move1 \
        --label variant1 -- -R gemini/gemini-3.1-flash-lite --prompt-variant task_variant_1

Select datasets with --datasets ds1,ds2 (include list) and/or --exclude ds1
(skip list); default is the full suite.

Needs GEMINI_API_KEY when the condition uses a Gemini model.
Resumable: results append to <out>/results.csv keyed on (label, dataset, seed);
a rerun skips rows already present.  Re-print with --tabulate.
"""
import argparse
import csv
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import BESTJ, DATASETS, REPO, run_slipper_isolated, write_run_config

DEFAULT_OUT = os.path.join(REPO, "scripts", "bestj-out")
_lock = threading.Lock()


def slugify(parts):
    s = "_".join(parts).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s or "default"


def load_done(results_path, label):
    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if row["label"] == label:
                    done[(row["dataset"], int(row["seed"]))] = float(row["err"])
    return done


def parse_datasets(include, exclude):
    """Resolve --datasets/--exclude (comma-separated names) against DATASETS,
    preserving suite order.  Unknown names are fatal so a typo can't silently
    shrink the run."""
    def split(s):
        return [d.strip() for d in s.split(",") if d.strip()] if s else []
    inc, exc = split(include), split(exclude)
    for d in inc + exc:
        if d not in DATASETS:
            sys.exit(f"unknown dataset {d!r}; known: {', '.join(DATASETS)}")
    selected = [d for d in DATASETS if (not inc or d in inc) and d not in exc]
    if not selected:
        sys.exit("no datasets selected after applying --datasets/--exclude")
    return selected


def run_cell(griddir, label, extra, ds, seed, log_tools=False):
    celldir = os.path.join(griddir, f"{label}__{ds}__{seed}")
    flags = ["--seed", str(seed), *extra, "-i0", "-n", str(BESTJ[ds])]
    if log_tools:
        # -L stem = this cell's own dataset stem, so each cell writes its
        # per-round tool-call sidecars (STEM-NNN-STAGE.tools.json) into its own
        # dir instead of racing on a single shared path.
        flags += ["-L", os.path.join(celldir, ds)]
    return run_slipper_isolated(celldir, ds, flags)


def run_suite(out, label, extra, seeds, workers, datasets, log_tools=False):
    griddir = os.path.join(out, "cells")
    results_path = os.path.join(out, "results.csv")
    os.makedirs(griddir, exist_ok=True)
    write_run_config(out, label, {
        "label": label,
        "seeds": "1..%d" % seeds,
        "workers": workers,
        "budget": "bestj (-i0 -n<bestj> per dataset)",
        "slipper_flags": " ".join(extra) or "(none)",
        "command": "uv run slipper --seed <S> %s -i0 -n<bestj> tuning_suite/<stem>"
                   % (" ".join(extra)),
        "datasets": datasets,
        "bestj": {ds: BESTJ[ds] for ds in datasets},
    })
    done = load_done(results_path, label)
    tasks = [(ds, s) for ds in datasets for s in range(1, seeds + 1)
             if (ds, s) not in done]
    n_done = sum(1 for ds, s in done if ds in datasets and s <= seeds)
    print(f"suite '{label}' (flags: {' '.join(extra) or '(none)'}): "
          f"{len(datasets) * seeds} cells, {n_done} done, {len(tasks)} to run "
          f"({workers}-way parallel)", flush=True)
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(["label", "dataset", "seed", "err"])

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_cell, griddir, label, extra, ds, s, log_tools): (ds, s)
                for ds, s in tasks}
        for fut in as_completed(futs):
            ds, s = futs[fut]
            err = fut.result()
            n += 1
            if err is None:
                continue
            with _lock:
                with open(results_path, "a", newline="") as f:
                    csv.writer(f).writerow([label, ds, s, err])
            print(f"  [{n}/{len(tasks)}] {ds:24} seed{s:2} -> {err:5.2f}", flush=True)
    print("suite complete", flush=True)


def tabulate(out, label, seeds, datasets):
    done = load_done(os.path.join(out, "results.csv"), label)
    print(f"\n## bestj suite: `{label}`\n")
    print("| Dataset | bestj | test err % (mean ± sd) |")
    print("|---|---:|---|")
    macro = []
    for ds in datasets:
        vals = np.array([done[(ds, s)] for s in range(1, seeds + 1) if (ds, s) in done])
        if len(vals):
            macro.append(vals.mean())
            if len(vals) > 1:
                cell = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f} (n={len(vals)})"
            else:
                cell = f"{vals[0]:.2f} (n=1)"
        else:
            cell = "—"
        print(f"| {ds} | {BESTJ[ds]} | {cell} |")
    if macro:
        print(f"| **macro-avg** | | **{np.mean(macro):.2f}** |")


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, extra = argv[:i], argv[i + 1:]
    else:
        own, extra = argv, []
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=None, help="condition name (default: slug of flags)")
    ap.add_argument("--seeds", type=int, default=10, help="seeds 1..N per dataset (default 10)")
    ap.add_argument("--workers", type=int, default=12, help="concurrent slipper runs")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (results.csv + cells/)")
    ap.add_argument("--tabulate", action="store_true", help="re-print table from results.csv")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated datasets to run (default: all)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated datasets to skip (e.g. move1)")
    ap.add_argument("--log-tools", action="store_true",
                    help="write each cell's per-round tool-call log "
                         "(STEM-NNN-STAGE.tools.json) into its cell dir")
    args = ap.parse_args(own)
    label = args.label or slugify(extra)
    datasets = parse_datasets(args.datasets, args.exclude)
    if args.tabulate:
        tabulate(args.out, label, args.seeds, datasets)
    else:
        run_suite(args.out, label, extra, args.seeds, args.workers, datasets,
                  args.log_tools)
        tabulate(args.out, label, args.seeds, datasets)


if __name__ == "__main__":
    main()
