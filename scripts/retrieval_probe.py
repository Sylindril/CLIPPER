#!/usr/bin/env python3
"""Can the model READ the prompt table?  A perception probe, no learner involved.

Tests form (b) of the large-table conjecture: that the model reads a
big markdown table worse than a small one (attention/KV effects on a 400-row
block), as opposed to form (a), column binding, which the
`--table-minibatch-size` nulls already ruled out.

Method: render the REAL prompt table (`llm_rule_builder._render_examples_table`,
same schema/cell formatting slopper uses) at several `-M` values, then ask the
model questions whose answers are computable from the rendered rows, and score
them.  No boosting, no rule building, so nothing here depends on the learner.

    uv run python scripts/retrieval_probe.py                 # run + summarize
    uv run python scripts/retrieval_probe.py --tabulate      # re-print from csv

Reading the result:

  - accuracy FLAT in M   => perception is fine; the -M penalty is about rule
                            *choice*, and form (b) is dead too.
  - accuracy DEGRADING   => form (b) confirmed; re-run with
                            --table-minibatch-size to see if it recovers.

The per-question `depth` (target row's position as a fraction of the table) is
recorded so accuracy can also be read against how deep the answer is buried --
the direct signature of "late rows get attenuated" as opposed to a general
size effect.
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import REPO

sys.path.insert(0, os.path.join(REPO, "src"))
from slipper import config, dataset as ds_mod, rng           # noqa: E402
from slipper import llm_rule_builder as L                    # noqa: E402

# Datasets with enough training rows that M=400 is a real subsample.
DATASETS = ["hypothyroid", "titanic", "abalone", "german_credit", "obesity", "move1"]
MS = [50, 100, 200, 400]
SEEDS = range(1, 6)
OUT = os.path.join(REPO, "scripts", "probe-out")
MODEL = "gemini/gemini-3.1-flash-lite"

SYSTEM = ("You are reading a markdown table of training examples.  Answer each "
          "question using ONLY the table.  Reply with a JSON object mapping "
          "question id to answer, e.g. {\"q1\": \"42\", \"q2\": \"'yes'\"}.  Give "
          "the answer alone, exactly as it appears in the table cell (keep the "
          "quoting used in the table), with no explanation.")


# ---------------------------------------------------------------------------
# question generation -- ground truth comes from the rendered rows themselves
# ---------------------------------------------------------------------------

def _rows(sample):
    """[(class_name, {attr: rendered_cell})] in table order."""
    out = []
    for ex in sample:
        d = L.inst_to_dict(ex.inst)
        out.append((ex.lab.nom.name, {k: L._cell(v) for k, v in d.items()}))
    return out


def make_questions(sample, gen, n_lookup=6):
    """Questions + expected answers, computed from the table as rendered.

    Mostly unique-key lookups (one row's cell, retrieved via another cell that
    is unique in the table) because those localize to a single row, so `depth`
    is meaningful.  Two aggregates are included as a contrast: they need the
    whole table, and should degrade first if the table is being read partially.
    """
    rows = _rows(sample)
    n = len(rows)
    keys = list(rows[0][1]) if rows else []
    qs = []

    # unique-key lookups: find (col, value) appearing in exactly one row
    counts = {}
    for i, (_, d) in enumerate(rows):
        for k in keys:
            counts.setdefault((k, d[k]), []).append(i)
    uniq = [(k, v, ix[0]) for (k, v), ix in counts.items()
            if len(ix) == 1 and v not in ("None", "'None'")]
    gen.shuffle(uniq)
    for k, v, i in uniq[:n_lookup]:
        other = [o for o in keys if o != k and rows[i][1][o] not in ("None", "'None'")]
        if not other:
            continue
        o = other[gen.integers(len(other))]
        qs.append(dict(kind="lookup", depth=i / max(n - 1, 1),
                       q="In the row where %s is %s, what is %s?" % (k, v, o),
                       want=rows[i][1][o]))

    # aggregate 1: class counts
    classes = sorted({c for c, _ in rows})
    if classes:
        c = classes[gen.integers(len(classes))]
        qs.append(dict(kind="class_count", depth=-1.0,
                       q="How many rows have class = %s?" % c,
                       want=str(sum(1 for cc, _ in rows if cc == c))))

    # aggregate 2: max of a numeric column, answered as the rendered cell
    kinds = L._attr_kind_map()
    nums = [k for k in keys if kinds.get(k) == "continuous"]
    if nums:
        k = nums[gen.integers(len(nums))]
        vals = [(float(re.sub(r"^'|'$", "", d[k])), d[k])
                for _, d in rows if d[k] not in ("None", "'None'")
                and re.fullmatch(r"'?-?[\d.eE+]+'?", d[k])]
        if vals:
            qs.append(dict(kind="col_max", depth=-1.0,
                           q="What is the largest value in column %s?" % k,
                           want=max(vals)[1]))
    return qs


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def norm(s):
    """Loose match: strip quotes/space, case-fold, canonicalize numbers."""
    s = str(s).strip().strip("'\"").strip()
    try:
        f = float(s)
        return "%g" % f
    except ValueError:
        return s.lower()


def correct(got, want):
    if got is None:
        return False
    a, b = norm(got), norm(want)
    if a == b:
        return True
    try:                                   # numeric tolerance for float cells
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# one cell = one (dataset, M, seed) table + one LLM call carrying its questions
# ---------------------------------------------------------------------------

def render(ds, m, seed):
    """The exact table slopper would put in a round-0 prompt, plus its questions."""
    config.Example_format, config.Cell_format = "table", "repr"
    config.Schema_format = "full"
    rng.seed(seed)
    train, _ = ds_mod.load_dataset(os.path.join(REPO, "tuning_suite", ds + ".json"))
    from slipper.data import weighted_sample
    for ex in train:                        # round 0: uniform weights
        ex.wt = 1.0 / len(train)
    sample = weighted_sample(train, m)
    return L._render_examples_table(sample), make_questions(sample, np.random.default_rng(seed))


def ask(table, qs):
    import litellm
    body = "\n".join("%s: %s" % ("q%d" % i, q["q"]) for i, q in enumerate(qs))
    r = litellm.completion(
        model=MODEL, timeout=120,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": table + "\n\nQuestions:\n" + body}],
        response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


def cell(ds, m, seed, table, qs):
    """Ask one cell's questions and score them.  `render` must already have run:
    it touches slipper's module globals (names/grammar/...), so rendering is done
    serially up front and only the LLM calls are parallelized."""
    try:
        got = ask(table, qs)
    except Exception as e:                  # a dead cell must not kill the sweep
        print("  FAIL %-16s M%-4d seed%-2d %s" % (ds, m, seed, e), file=sys.stderr)
        return []
    return [dict(dataset=ds, m=m, seed=seed, kind=q["kind"],
                 depth="%.3f" % q["depth"],
                 correct=int(correct(got.get("q%d" % i), q["want"])),
                 want=q["want"], got=str(got.get("q%d" % i))[:60])
            for i, q in enumerate(qs)]


# ---------------------------------------------------------------------------

def tabulate(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return print("no rows in %s" % path)
    ms = sorted({int(r["m"]) for r in rows})

    def acc(sel):
        v = [int(r["correct"]) for r in rows if sel(r)]
        return (100.0 * sum(v) / len(v), len(v)) if v else (float("nan"), 0)

    print("\n## retrieval accuracy by table size (%s, %d questions)\n"
          % (MODEL, len(rows)))
    print("%-22s %s" % ("slice", "  ".join("M=%-9d" % m for m in ms)))
    print("-" * (22 + 12 * len(ms)))
    for label, sel in ([("all questions", lambda r: True)]
                       + [(k, (lambda k: lambda r: r["kind"] == k)(k))
                          for k in sorted({r["kind"] for r in rows})]
                       + [(d, (lambda d: lambda r: r["dataset"] == d)(d))
                          for d in sorted({r["dataset"] for r in rows})]):
        cells = []
        for m in ms:
            a, n = acc(lambda r, m=m: int(r["m"]) == m and sel(r))
            cells.append("%5.1f%% (%3d)" % (a, n) if n else "     —     ")
        print("%-22s %s" % (label, "  ".join(cells)))

    print("\n## lookup accuracy by depth of the target row (fraction into the table)\n")
    bins = [(0.0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
    print("%-22s %s" % ("depth", "  ".join("M=%-9d" % m for m in ms)))
    print("-" * (22 + 12 * len(ms)))
    for lo, hi in bins:
        cells = []
        for m in ms:
            a, n = acc(lambda r, m=m, lo=lo, hi=hi: int(r["m"]) == m
                       and r["kind"] == "lookup" and lo <= float(r["depth"]) < hi)
            cells.append("%5.1f%% (%3d)" % (a, n) if n else "     —     ")
        print("%-22s %s" % ("%.2f-%.2f" % (lo, min(hi, 1.0)), "  ".join(cells)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabulate", action="store_true", help="summarize the csv, no runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.csv")
    if args.tabulate:
        return tabulate(path)

    done = set()
    if os.path.exists(path):
        done = {(r["dataset"], int(r["m"]), int(r["seed"]))
                for r in csv.DictReader(open(path))}
    todo = [(d, m, s) for d in args.datasets.split(",")
            for m in map(int, args.ms.split(",")) for s in SEEDS
            if (d, m, s) not in done]
    print("%d cells to run (%d already done), %d-way parallel"
          % (len(todo), len(done), args.workers))

    # Serial: rendering mutates slipper's global names/grammar state.
    built = [(d, m, s) + render(d, m, s) for d, m, s in todo]

    fields = ["dataset", "m", "seed", "kind", "depth", "correct", "want", "got"]
    new = os.path.exists(path)
    with open(path, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        if not new:
            w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(cell, *t): t[:3] for t in built}
            for i, f in enumerate(as_completed(futs), 1):
                d, m, s = futs[f]
                rows = f.result()
                for r in rows:
                    w.writerow(r)
                fp.flush()
                ok = sum(r["correct"] for r in rows)
                print("  [%d/%d] %-16s M%-4d seed%-2d  %d/%d"
                      % (i, len(todo), d, m, s, ok, len(rows)))
    tabulate(path)


if __name__ == "__main__":
    main()
