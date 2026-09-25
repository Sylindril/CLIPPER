#!/usr/bin/env python3
"""Learning curve over training-set size for the imodels / CORELS baselines.

Sibling of run_learning_curve.py, and deliberately parasitic on it: the training
subsets come from that script's ``subsample_train``, so a baseline cell sees the
*byte-identical* subset a slipper/slopper cell at the same (dataset,size,seed)
saw, and the test split is the full committed one.  Results append to the same
``scripts/lc-out/results.csv``, so summarize_learning_curve.py and
plot_learning_curve.py pick the baselines up with no changes.

The models themselves come from ``scripts/imodels_baselines.py``.  This script
only supplies the subsets and reuses that harness's encoders, per-fit timeout,
and error handling, so the baselines are fit exactly as in the full-data table
on the development suite.  The imodels + vendored-CORELS dependencies live in the
``baselines`` group, so run it as:

    uv run --group baselines python scripts/run_baseline_curve.py \
        --suite test --models figs,rulefit,corels

A bare model name is prefixed with ``imodels_`` (``figs`` ->
``imodels_figs``); the results.csv label stays the bare name.  ``--list`` shows
what the harness registers.

The ``bestj`` column is written as 0 for every baseline row: these models have
no round budget to cross-validate, so there is nothing comparable to put there
(don't read the summarizer's ``--bestj`` table across baseline labels).
"""
import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_learning_curve as rlc
import suite as suite_mod
from run_learning_curve import load_source, parse_datasets, subsample_train
from suite import add_suite_arg, write_run_config

HARNESS = os.path.dirname(os.path.abspath(__file__))   # imodels_baselines.py
DEFAULT_MODELS = ["figs", "rulefit", "corels"]

_tl = threading.local()      # per-worker: the cell dir the harness should read
_ib = None                   # the imodels_baselines module, once imported


class _CellSuite:
    """Stand-in for the harness's ``SUITE`` directory constant.  It reads that
    global as ``os.path.join(SUITE, ds + '.json')``, so a path-like object that
    resolves per-cell redirects its loader at our subsample without patching it
    (and without a shared mutable global across workers)."""

    def __fspath__(self):
        return _tl.celldir


def ib_name(model):
    return model if model.startswith("imodels_") else "imodels_" + model


def init_worker(timeout, suite_name="tuning"):
    """Import the baselines harness once per process and point it at our cells.
    Also re-resolves the active suite: run_cell calls load_source() in the
    worker, and ProcessPoolExecutor does not inherit the parent's module state
    on spawn-based platforms."""
    global _ib
    rlc.SUITE = suite_mod.get(suite_name)
    sys.path.insert(0, HARNESS)
    import imodels_baselines as ib
    ib.MODELS = ib._make_models()
    ib.SUITE = _CellSuite()
    ib.TIMEOUT = timeout
    _ib = ib


def run_cell(griddir, model, ds, size, seed):
    """Write the subset, fit the baseline on it, score the full test split.
    Returns the harness's result dict, or None if the size isn't available."""
    spec = subsample_train(load_source(ds), ds, size, seed)
    if spec is None:
        return None                             # size > available train rows
    celldir = os.path.join(griddir, f"size.{size}", f"{model}__{ds}__{seed}")
    os.makedirs(celldir, exist_ok=True)
    with open(os.path.join(celldir, f"{ds}.json"), "w") as f:
        json.dump(spec, f)
    _tl.celldir = celldir
    name = ib_name(model)
    return _ib.run_one(name, _ib.MODELS[name], ds, seed)


def load_done(results_path, labels):
    done = set()
    if os.path.exists(results_path):
        for row in csv.DictReader(open(results_path)):
            if row["label"] in labels:
                done.add((row["label"], row["dataset"], int(row["size"]),
                          int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated baseline names (default %(default)s)")
    add_suite_arg(ap)
    ap.add_argument("--sizes", default=None,
                    help="comma-separated training sizes (default: the suite's)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel fits (separate processes: the harness's "
                         "SIGALRM timeout needs each fit in a main thread)")
    ap.add_argument("--timeout", type=float, default=180, metavar="SECS",
                    help="per-fit wall-clock cap (default %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: the suite's scripts/*lc-out)")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--list", action="store_true",
                    help="list the harness's model names and exit")
    args = ap.parse_args()

    if args.list:
        init_worker(None)
        print("\n".join(sorted(_ib.MODELS)))
        return

    rlc.SUITE = suite_mod.get(args.suite)
    out = args.out or rlc.SUITE.out("lc")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = ([int(s) for s in args.sizes.split(",") if s.strip()]
             if args.sizes else rlc.SUITE.sizes)
    datasets = parse_datasets(args.datasets, args.exclude)
    griddir = os.path.join(out, "cells")
    results_path = os.path.join(out, "results.csv")
    os.makedirs(griddir, exist_ok=True)
    for model in models:
        write_run_config(out, model, {
            "label": model, "model": ib_name(model), "sizes": sizes,
            "seeds": "1..%d" % args.seeds, "suite": rlc.SUITE.name,
            "timeout": args.timeout, "datasets": datasets,
            "bestj": "n/a for baselines (written as 0)",
        })

    done = load_done(results_path, set(models))
    trainlen = {ds: len(load_source(ds).get("train") or []) for ds in datasets}
    tasks = [(m, ds, sz, s) for m in models for ds in datasets for sz in sizes
             for s in range(1, args.seeds + 1)
             if sz <= trainlen[ds] and (m, ds, sz, s) not in done]
    print("baselines %s: %d cells to run, %d already done (%d-way parallel)"
          % (", ".join(models), len(tasks), len(done), args.workers), flush=True)
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(["label", "dataset", "size", "seed", "bestj", "err",
                                    "rules"] + rlc._SIDE_COLS)

    bad = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.timeout, rlc.SUITE.name)) as ex:
        futs = {ex.submit(run_cell, griddir, m, ds, sz, s): (m, ds, sz, s)
                for m, ds, sz, s in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            m, ds, sz, s = futs[fut]
            res = fut.result()
            if res is None or res.get("status") != "ok":
                bad += 1
                sys.stderr.write("[%s] %s size=%d seed=%d: %s\n" % (
                    m, ds, sz, s, res and (res.get("err") or res["status"])))
                continue
            with open(results_path, "a", newline="") as f:
                rules = res.get("rules")
                csv.writer(f).writerow([m, ds, sz, s, 0, "%.4f" % res["test"],
                                        "" if rules is None else rules])
            print("  [%d/%d] %-9s %-22s size%-5d seed%-3d rules%-5s -> %5.2f"
                  % (i, len(tasks), m, ds, sz, s, res.get("rules"), res["test"]),
                  flush=True)
    print("done (%d cells failed/skipped)" % bad, flush=True)


if __name__ == "__main__":
    main()
