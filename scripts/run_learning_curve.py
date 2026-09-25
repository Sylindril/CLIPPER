#!/usr/bin/env python3
"""Learning curve over training-set size for one slipper/slopper condition.

Extends run_bestj.py: instead of one fixed bestj per dataset, each cell
subsamples the training set to a given size, recomputes bestj on that subset
via slipper-default cross-validation, then runs the condition at that bestj.
The test set is the full committed test split (identical across all sizes), so
error is comparable down the curve.  Motivation: slopper (LLM) is expected to
beat discrete slipper more as the training set shrinks.

Each cell (dataset, size, seed):
  1. seed-sample `size` examples from the dataset's train split (deterministic
     in (dataset, size, seed), so a slipper and a slopper invocation see the
     SAME subset); the test split is left whole.
  2. run slipper-default (-R '' -v1, internal 5-fold CV) on the subset ->
     parse the chosen bestj from "// best result ... after N rounds".
  3. run the condition (flags after --) at -i0 -n <bestj> on the subset ->
     parse the test error.
Run data lands under <out>/cells/size.<NNN>/<label>__<ds>__<seed>/.

Results append to <out>/results.csv with columns label,dataset,size,seed,
bestj,err (note the extra `size` and `bestj` columns vs run_bestj).  Resumable:
a rerun skips (label,dataset,size,seed) rows already present.

    # discrete slipper learning curve, 10 seeds, default sizes
    uv run python scripts/run_learning_curve.py --label slipper -- -R ''

    # slopper (best LLM config) learning curve on a subset of datasets
    uv run python scripts/run_learning_curve.py --label slopper \
        --datasets abalone,titanic,move1 -- \
        -R gemini/gemini-3.1-flash-lite --example-format table --schema-format full \
        --cell-format repr --prompt-variant task_variant_2 -T6 -b --no-count-coverage \
        --tools-prompt variant1

    uv run python scripts/run_learning_curve.py --label slipper --tabulate

    # another suite (e.g. the text suite; results land in its own
    # scripts/text-lc-out via Suite.out)
    uv run python scripts/run_learning_curve.py --suite text \
        --label slipper --datasets sentiment_yelp -- -R ''
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite as suite_mod
from shyp_size import count_rules
from suite import REPO, TUNING, _advice_ext, add_suite_arg, write_run_config
from slipper.errors import ACCOUNT_ERROR_EXIT

# Which suite the module-level helpers (load_source / link_sidecars /
# parse_datasets) resolve against.  Set once from --suite in main();
# run_baseline_curve.py, which imports those helpers, sets it the same way.
SUITE = TUNING
DEFAULT_OUT = TUNING.out("lc")
DEFAULT_SIZES = TUNING.sizes
_TEST_ERR_RE = re.compile(r"^Test error rate:\s+([\d.]+)%", re.MULTILINE)
_BESTJ_RE = re.compile(r"best result .* after (\d+) rounds")
# cli.py prints this only when the run made LLM calls (see its `LLM side info:`
# emitter); discrete runs have no such line, so these columns stay blank there.
_SIDE_RE = re.compile(
    r"^LLM side info:\s+(\d+) calls, ([\d.]+)s latency, "
    r"(\d+) in / (\d+) out tokens, (?:\$([\d.]+)|cost n/a)", re.MULTILINE)
_SIDE_COLS = ["llm_calls", "latency_s", "in_tok", "out_tok", "cost_usd"]


# cli.py's second summary line: every provider request (successful attempt or
# not) and the failures themselves.  Appended as new results.csv columns; rows
# written before it existed simply leave them blank.
_ACCT_RE = re.compile(
    r"^LLM accounting:\s+(\d+) requests \((\d+) failed\), \d+ in / \d+ out tokens, "
    r"(?:\$([\d.]+)|cost n/a[^,]*), (\d+) failed attempts, (\d+) lost rounds", re.MULTILINE)
_ACCT_COLS = ["api_requests", "api_failures", "api_cost_usd", "failed_attempts",
              "lost_rounds"]
_HEADER = (["label", "dataset", "size", "seed", "bestj", "err", "rules"]
           + _SIDE_COLS + _ACCT_COLS)


def parse_accounting(stdout):
    """dict of the `LLM accounting:` fields, or {} for a run with no LLM
    requests (the discrete path).  api_cost_usd is blank for an unpriced model."""
    m = _ACCT_RE.search(stdout)
    if not m:
        return {}
    return dict(zip(_ACCT_COLS, [m.group(1), m.group(2), m.group(3) or "",
                                 m.group(4), m.group(5)]))


def _ensure_header(results_path):
    """Create results.csv with the full header, or extend an older header in
    place (new columns are only ever appended, so existing rows keep their
    positions and read back with the new fields blank)."""
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)
        return
    with open(results_path, newline="") as f:
        lines = f.readlines()
    old = next(csv.reader([lines[0]])) if lines else []
    if old == _HEADER[:len(old)] and len(old) < len(_HEADER):
        lines[0] = ",".join(_HEADER) + "\n"
        tmp = results_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            f.writelines(lines)
        os.replace(tmp, results_path)


def parse_side_info(stdout):
    """dict of the `LLM side info:` fields, or {} for a run that made no calls.
    cost is left blank for an unpriced model (cli prints 'cost n/a')."""
    m = _SIDE_RE.search(stdout)
    if not m:
        return {}
    return dict(zip(_SIDE_COLS, [m.group(1), m.group(2), m.group(3), m.group(4),
                                 m.group(5) or ""]))
_lock = threading.Lock()
_json_cache = {}


class SuiteAbort(RuntimeError):
    """A cell's slipper run exited with ACCOUNT_ERROR_EXIT (no credit, bad
    key): every later cell would fail the same way, so the suite stops."""


class _Skipped(Exception):
    """A queued cell that saw the abort flag before starting."""


# Set by the first SuiteAbort.  Future.cancel() alone is not enough: a pool
# worker picks up the next queued cell the moment the failing one returns,
# before the main thread has seen the failure and cancelled anything -- so
# every cell checks this flag itself before doing any work.
_abort = threading.Event()


def _guarded_cell(*args):
    if _abort.is_set():
        raise _Skipped()
    try:
        return run_cell(*args)
    except SuiteAbort:
        _abort.set()
        raise

_cache_lock = threading.Lock()


def load_source(ds):
    """Cached original dataset json (has classes/attributes/train/test)."""
    with _cache_lock:
        if ds not in _json_cache:
            with open(SUITE.path(ds, ".json")) as f:
                _json_cache[ds] = json.load(f)
        return _json_cache[ds]


def subsample_train(spec, ds, size, seed):
    """Deterministic seed-sample of `size` train rows (in (ds,size,seed)); the
    test split is left whole.  Returns a new spec dict, or None if size exceeds
    the available train rows."""
    train = spec.get("train") or []
    if size > len(train):
        return None
    key = SUITE.sample_key.get(ds, ds)
    h = int(hashlib.md5(f"{key}|{size}|{seed}".encode()).hexdigest(), 16) % (2**32)
    sub = random.Random(h).sample(train, size)
    out = dict(spec)
    out["train"] = sub
    return out


def _run(flags, celljson):
    import subprocess
    cmd = ["uv", "run", "slipper", *flags, celljson]
    # utf-8 + replace: the child prints utf-8 (rules may quote emoji under
    # --use-text); without this the reader thread decodes with cp1252 and dies.
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                          encoding="utf-8", errors="replace")


def link_sidecars(celldir, ds, flags):
    """Link .info and any --include-advice file next to the subset json."""
    exts = [".info"] if os.path.exists(SUITE.path(ds, ".info")) else []
    adv = _advice_ext(flags)
    if adv and os.path.exists(SUITE.path(ds, adv)):
        exts.append(adv)
    for ext in exts:
        link = os.path.join(celldir, ds + ext)
        if not os.path.exists(link):
            os.symlink(SUITE.path(ds, ext), link)


def load_bestj(spec):
    """{(dataset, size, seed): bestj} from a `RESULTS.csv:LABEL` spec."""
    path, label = spec.rsplit(":", 1)
    return {(r["dataset"], int(r["size"]), int(r["seed"])): int(r["bestj"])
            for r in csv.DictReader(open(path))
            if r["label"] == label and r["bestj"]}


def run_cell(griddir, label, extra, ds, size, seed, min_bestj=1, bestj_from=None):
    """Sample -> recompute bestj on the subset -> run the condition.  Returns
    (bestj, err, rules, side), or (None, None, None, {}) on skip/failure.
    `side` is the parsed `LLM side info:` line (empty for discrete runs).  `rules` is the
    learned hypothesis size read back from the cell's .shyp (default rule
    excluded); None if the file is missing.

    `min_bestj` floors the CV-chosen budget.  Internal CV can pick 0 rounds on a
    heavily imbalanced dataset (no rule beats the base rate), which yields an
    empty hypothesis that abstains to the majority class -- a degenerate cell
    that carries no signal.  Flooring at 2 forces a hypothesis; apply the same
    floor to every label so the comparison stays paired.

    `bestj_from` ({(dataset,size,seed): bestj}, keyed on the suite's sample_key
    name) replaces step 1 with a stored budget; cells missing from it fail."""
    spec = subsample_train(load_source(ds), ds, size, seed)
    if spec is None:
        return None, None, None, {}             # size > available train rows
    celldir = os.path.join(griddir, f"size.{size}", f"{label}__{ds}__{seed}")
    os.makedirs(celldir, exist_ok=True)
    celljson = os.path.join(celldir, f"{ds}.json")
    with open(celljson, "w") as f:
        json.dump(spec, f)
    link_sidecars(celldir, ds, extra)

    # 1. bestj from a slipper-default internal-CV run on the subset (or stored).
    if bestj_from is not None:
        bestj = bestj_from.get((SUITE.sample_key.get(ds, ds), size, seed))
        if bestj is None:
            sys.stderr.write(f"[FAIL bestj] {ds} size={size} seed={seed}: not in --bestj-from\n")
            return None, None, None, {}
    else:
        proc = _run(["--seed", str(seed), "-R", "", "-v1"], celljson)
        m = _BESTJ_RE.search(proc.stdout)
        if not m:
            sys.stderr.write(f"[FAIL bestj] {ds} size={size} seed={seed} "
                             f"rc={proc.returncode}\n{(proc.stderr or proc.stdout)[-400:]}\n")
            return None, None, None, {}
        bestj = max(int(m.group(1)), min_bestj)

    # 2. the condition at the recomputed bestj.
    proc = _run(["--seed", str(seed), *extra, "-i0", "-n", str(bestj)], celljson)
    if proc.returncode == ACCOUNT_ERROR_EXIT:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        raise SuiteAbort("%s size=%s seed=%s: %s"
                         % (ds, size, seed, tail[-1] if tail else "account error"))
    # The full slipper output, kept next to the cell's other records: per-round
    # warnings (failed attempts, no-op rules) used to be discarded whenever a
    # cell produced a test error, which is how silent failures went unseen.
    with open(os.path.join(celldir, "slipper.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout + "\n--- stderr ---\n" + (proc.stderr or ""))
    m = _TEST_ERR_RE.search(proc.stdout)
    if not m:
        sys.stderr.write(f"[FAIL run] {ds} size={size} seed={seed} bestj={bestj} "
                         f"rc={proc.returncode}\n{(proc.stderr or proc.stdout)[-400:]}\n")
        return bestj, None, None, {}
    shyp = os.path.splitext(celljson)[0] + ".shyp"
    rules = count_rules(shyp) if os.path.exists(shyp) else None
    return bestj, float(m.group(1)), rules, {**parse_side_info(proc.stdout),
                                            **parse_accounting(proc.stdout)}


def load_done(results_path, label):
    done = {}
    if os.path.exists(results_path):
        for row in csv.DictReader(open(results_path)):
            if row["label"] == label:
                done[(row["dataset"], int(row["size"]), int(row["seed"]))] = row
    return done


def parse_datasets(include, exclude):
    def split(s):
        return [d.strip() for d in s.split(",") if d.strip()] if s else []
    inc, exc = split(include), split(exclude)
    known = SUITE.datasets
    for d in inc + exc:
        if d not in known:
            sys.exit(f"unknown dataset {d!r}; known: {', '.join(known)}")
    sel = [d for d in known if (not inc or d in inc) and d not in exc]
    if not sel:
        sys.exit("no datasets selected")
    return sel


def run_suite(out, label, extra, sizes, seeds, workers, datasets, min_bestj=1,
              bestj_from=None):
    stored = load_bestj(bestj_from) if bestj_from else None
    griddir = os.path.join(out, "cells")
    results_path = os.path.join(out, "results.csv")
    os.makedirs(griddir, exist_ok=True)
    write_run_config(out, label, {
        "label": label, "sizes": sizes, "seeds": "1..%d" % seeds,
        "workers": workers, "slipper_flags": " ".join(extra) or "(none)",
        "bestj": ("stored, from bestj_from" if bestj_from else
                  "recomputed per (dataset,size,seed) via slipper-default CV"),
        "min_bestj": min_bestj,
        "bestj_from": bestj_from,
        "suite": SUITE.name,
        "datasets": datasets,
    })
    done = load_done(results_path, label)
    # A size point is only valid where the dataset has that many train rows;
    # report the dropped (dataset,size) pairs instead of leaving cells blank.
    trainlen = {ds: len(load_source(ds).get("train") or []) for ds in datasets}
    dropped = sorted({(ds, sz) for ds in datasets for sz in sizes
                      if sz > trainlen[ds]})
    if dropped:
        print("note: dropping %d size points beyond available train rows: %s"
              % (len(dropped), ", ".join(f"{ds}@{sz}(n={trainlen[ds]})"
                                         for ds, sz in dropped)), flush=True)
    tasks = [(ds, sz, s) for ds in datasets for sz in sizes
             for s in range(1, seeds + 1)
             if sz <= trainlen[ds] and (ds, sz, s) not in done]
    valid = sum(1 for ds in datasets for sz in sizes if sz <= trainlen[ds]) * seeds
    print(f"suite '{label}' (flags: {' '.join(extra) or '(none)'}): "
          f"{valid} valid cells, {len(done)} done, "
          f"{len(tasks)} to run ({workers}-way parallel)", flush=True)
    _ensure_header(results_path)

    n = 0
    abort = None
    _abort.clear()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_guarded_cell, griddir, label, extra, ds, sz, s, min_bestj, stored): (ds, sz, s)
                for ds, sz, s in tasks}
        for fut in as_completed(futs):
            ds, sz, s = futs[fut]
            try:
                bestj, err, rules, side = fut.result()
            except SuiteAbort as e:
                # First account error stops the suite: cancel every queued
                # cell (running ones finish or fail fast on their own) and
                # write no row, so a rerun resumes exactly the missing cells.
                if abort is None:
                    abort = e
                    sys.stderr.write("[ABORT] %s\n" % e)
                    for f in futs:
                        f.cancel()
                continue
            except (CancelledError, _Skipped):
                continue
            n += 1
            if err is None:
                continue
            with _lock, open(results_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [label, ds, sz, s, bestj, err,
                     "" if rules is None else rules]
                    + [side.get(c, "") for c in _SIDE_COLS + _ACCT_COLS])
            print(f"  [{n}/{len(tasks)}] {ds:22} size{sz:<4} seed{s:2} "
                  f"bestj{bestj:<3} rules{'-' if rules is None else rules:<4} "
                  f"-> {err:5.2f}", flush=True)
    if abort is not None:
        print("suite ABORTED on an LLM account error after %d cells were written "
              "this run.  Fix the account (credit / key) and rerun the same "
              "command: finished cells are skipped." % n, flush=True)
        raise SystemExit(ACCOUNT_ERROR_EXIT)
    print("suite complete", flush=True)


def tabulate(out, label, sizes, seeds, datasets):
    done = load_done(os.path.join(out, "results.csv"), label)
    def cell(ds, sz):
        v = np.array([float(r["err"]) for (d, z, _), r in done.items()
                      if d == ds and z == sz])
        return v
    print(f"\n## learning curve: `{label}`  (test err %, mean ± sd over seeds)\n")
    head = "| Dataset | " + " | ".join(f"n={z}" for z in sizes) + " |"
    print(head)
    print("|" + "---|" * (len(sizes) + 1))
    for ds in datasets:
        cells = []
        for sz in sizes:
            v = cell(ds, sz)
            cells.append(f"{v.mean():.2f}±{v.std(ddof=1):.2f}" if len(v) > 1
                         else (f"{v[0]:.2f}" if len(v) else "—"))
        print(f"| {ds} | " + " | ".join(cells) + " |")
    # mean bestj per (dataset,size), to show the round budget shrinking with n
    print(f"\nmean bestj by size:")
    for ds in datasets:
        bj = []
        for sz in sizes:
            b = [int(r["bestj"]) for (d, z, _), r in done.items() if d == ds and z == sz]
            bj.append(f"{np.mean(b):.0f}" if b else "—")
        print(f"  {ds:24} " + "  ".join(f"n{sz}:{v}" for sz, v in zip(sizes, bj)))


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, extra = argv[:i], argv[i + 1:]
    else:
        own, extra = argv, []
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="condition name (results.csv key)")
    add_suite_arg(ap)
    ap.add_argument("--sizes", default=None,
                    help="comma-separated training sizes (default: the suite's)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None,
                    help="output dir (default: the suite's scripts/*lc-out)")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--min-bestj", type=int, default=1,
                    help="floor on the CV-chosen round budget (default %(default)s)")
    ap.add_argument("--bestj-from", default=None, metavar="RESULTS.csv:LABEL",
                    help="reuse the bestj of LABEL's rows in RESULTS.csv (matched on "
                         "dataset/size/seed) instead of recomputing it by CV")
    ap.add_argument("--tabulate", action="store_true")
    args = ap.parse_args(own)

    global SUITE
    SUITE = suite_mod.get(args.suite)
    out = args.out or SUITE.out("lc")
    sizes = ([int(s) for s in args.sizes.split(",") if s.strip()]
             if args.sizes else SUITE.sizes)
    datasets = parse_datasets(args.datasets, args.exclude)
    if args.tabulate:
        tabulate(out, args.label, sizes, args.seeds, datasets)
    else:
        run_suite(out, args.label, extra, sizes, args.seeds, args.workers,
                  datasets, args.min_bestj, args.bestj_from)
        tabulate(out, args.label, sizes, args.seeds, datasets)


if __name__ == "__main__":
    main()
