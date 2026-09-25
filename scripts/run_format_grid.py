#!/usr/bin/env python3
"""Run the example-format x schema-format grid for slopper across the tuning
suite and tabulate per-dataset means + paired t-tests.

Grid (per dataset):  {dict,table} x {full,compressed} x seed 1..N
Each cell:  slipper --seed S -R <model> -i0 -n<bestj> -M100
                    --example-format EF --schema-format SF  <stem>

- Isolation: each run gets its own temp dir with a symlink to the dataset
  json, so concurrent runs never race on the shared STEM.shyp / STEM.pyhyp.py
  and the committed tuning_suite/*.shyp reference files are left untouched.
- Resumable: results append to <out>/results.csv; existing
  (dataset,ef,sf,seed) rows are skipped on restart.
- Needs GEMINI_API_KEY (or set --model to another litellm string).

Usage:
    uv run python scripts/run_format_grid.py                # run + tabulate
    uv run python scripts/run_format_grid.py --workers 16
    uv run python scripts/run_format_grid.py --tabulate     # re-emit table only

The bestj round budgets below are the internal-CV round counts from the
`slipper2_default` condition on the development suite (the `bestj` column).
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "scripts", "format-grid-out")
DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"

SEEDS = list(range(1, 11))
EFMTS = ["dict", "table"]
SFMTS = ["full", "compressed"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import BESTJ, DATASETS, write_run_config  # noqa: E402

TEST_ERR_RE = re.compile(r"^Test error rate:\s+([\d.]+)%", re.MULTILINE)
_lock = threading.Lock()


def load_done(results_path):
    done = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for row in csv.DictReader(f):
                done[(row["dataset"], row["ef"], row["sf"], int(row["seed"]))] = float(row["err"])
    return done


def run_cell(griddir, model, ds, ef, sf, seed):
    celldir = os.path.join(griddir, f"{ds}__{ef}__{sf}__{seed}")
    os.makedirs(celldir, exist_ok=True)
    link = os.path.join(celldir, f"{ds}.json")
    if not os.path.exists(link):
        os.symlink(os.path.join(REPO, "tuning_suite", f"{ds}.json"), link)
    cmd = ["uv", "run", "slipper", "--seed", str(seed),
           "-R", model, "-i0", "-n", str(BESTJ[ds]), "-M100",
           "--example-format", ef, "--schema-format", sf,
           "--cell-format", "str",  # pin: this grid's committed CSV predates the repr default
           os.path.join(celldir, ds)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    m = TEST_ERR_RE.search(proc.stdout)
    if not m:
        sys.stderr.write(f"[FAIL] {ds} {ef} {sf} seed{seed} rc={proc.returncode}\n"
                         f"{(proc.stderr or proc.stdout)[-500:]}\n")
        return None
    return float(m.group(1))


def run_grid(out, model, max_workers):
    griddir = os.path.join(out, "cells")
    results_path = os.path.join(out, "results.csv")
    os.makedirs(griddir, exist_ok=True)
    write_run_config(out, "grid", {
        "model": model,
        "seeds": "1..%d" % len(SEEDS),
        "workers": max_workers,
        "example_formats": EFMTS,
        "schema_formats": SFMTS,
        "cell_format": "str (pinned; grid predates the repr default)",
        "budget": "bestj (-i0 -n<bestj>), M100",
        "command": ("uv run slipper --seed <S> -R %s -i0 -n<bestj> -M100 "
                    "--example-format <ef> --schema-format <sf> --cell-format str "
                    "tuning_suite/<stem>" % model),
        "bestj": {ds: BESTJ[ds] for ds in DATASETS},
    })
    done = load_done(results_path)
    tasks = [(ds, ef, sf, seed)
             for ds in DATASETS for ef in EFMTS for sf in SFMTS for seed in SEEDS
             if (ds, ef, sf, seed) not in done]
    total = len(DATASETS) * len(EFMTS) * len(SFMTS) * len(SEEDS)
    print(f"grid: {total} cells, {len(done)} done, {len(tasks)} to run "
          f"({max_workers}-way parallel)", flush=True)
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(["dataset", "ef", "sf", "seed", "err"])

    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_cell, griddir, model, *t): t for t in tasks}
        for fut in as_completed(futs):
            ds, ef, sf, seed = futs[fut]
            err = fut.result()
            n_done += 1
            if err is None:
                continue
            with _lock:
                with open(results_path, "a", newline="") as f:
                    csv.writer(f).writerow([ds, ef, sf, seed, err])
            print(f"  [{n_done}/{len(tasks)}] {ds:24} {ef:5} {sf:10} "
                  f"seed{seed:2} -> {err:5.2f}", flush=True)
    print("grid complete", flush=True)


def _cell_vals(done, ds, ef, sf):
    return np.array([done[(ds, ef, sf, s)] for s in SEEDS
                     if (ds, ef, sf, s) in done])


def _paired(a, b):
    """paired t-test of a vs b; returns (p, sig)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), "?"
    if np.allclose(a, b):
        return 1.0, "n.s."
    p = float(stats.ttest_rel(a, b).pvalue)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return p, sig


def _cell(a):
    if len(a) > 1:
        return f"{a.mean():.2f}±{a.std(ddof=1):.1f}"
    return f"{a[0]:.2f}" if len(a) else "—"


def tabulate(out):
    done = load_done(os.path.join(out, "results.csv"))
    print("\n## Condition: example-format x schema-format grid "
          "(gemini flash, bestj, M100, 10 seeds)\n")
    print("| Dataset | dict+full | dict+comp | table+full | table+comp "
          "| table−dict (full) | table full−comp |")
    print("|" + "---|" * 7)
    macro = {(ef, sf): [] for ef in EFMTS for sf in SFMTS}
    for ds in DATASETS:
        df = _cell_vals(done, ds, "dict", "full")
        dc = _cell_vals(done, ds, "dict", "compressed")
        tf = _cell_vals(done, ds, "table", "full")
        tc = _cell_vals(done, ds, "table", "compressed")
        for key, arr in [(("dict", "full"), df), (("dict", "compressed"), dc),
                         (("table", "full"), tf), (("table", "compressed"), tc)]:
            if len(arr):
                macro[key].append(arr.mean())
        p_fmt, s_fmt = _paired(tf, df)   # table+full vs dict+full
        p_sch, s_sch = _paired(tf, tc)   # table full vs table compressed
        dfmt = f"{(tf.mean()-df.mean()):+.2f} ({p_fmt:.3g} {s_fmt})" if len(tf) and len(df) else "—"
        dsch = f"{(tf.mean()-tc.mean()):+.2f} ({p_sch:.3g} {s_sch})" if len(tf) and len(tc) else "—"
        print(f"| {ds} | {_cell(df)} | {_cell(dc)} | {_cell(tf)} | {_cell(tc)} | {dfmt} | {dsch} |")
    m = {k: (np.mean(v) if v else float('nan')) for k, v in macro.items()}
    print(f"| **macro-avg** | **{m[('dict','full')]:.2f}** | **{m[('dict','compressed')]:.2f}** | "
          f"**{m[('table','full')]:.2f}** | **{m[('table','compressed')]:.2f}** | | |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (results.csv + cells/)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="litellm model string for -R")
    ap.add_argument("--workers", type=int, default=12, help="concurrent slipper runs")
    ap.add_argument("--tabulate", action="store_true", help="re-emit the table from results.csv, no runs")
    args = ap.parse_args()
    if args.tabulate:
        tabulate(args.out)
    else:
        run_grid(args.out, args.model, args.workers)
        tabulate(args.out)


if __name__ == "__main__":
    main()
