#!/usr/bin/env python3
"""Learning curve for RuleChef (github.com/KRLabsOrg/rulechef) on the text suite.

RuleChef asks an LLM to write regex/code rules from labeled examples and then
revises them, without boosting; it is the external LLM rule-learner comparison
in the paper (Figure 4c).

Run it in its own environment so RuleChef (an alpha-stage package, 0.2.2) does
not change the locked project environment:

    uv venv .rulechef-venv --python 3.12
    VIRTUAL_ENV=.rulechef-venv uv pip install rulechef==0.2.2 openai
    PYTHONIOENCODING=utf-8 OPENROUTER_API_KEY=... \
        .rulechef-venv/bin/python scripts/run_rulechef_curve.py

COMPARABILITY.  Each cell subsamples the training split with the same
md5(dataset|size|seed) hash as run_learning_curve.py, so it trains on exactly
the documents the CLIPPER and SLIPPER cells saw, and error is measured on the
full test split.  The model is gemini-3.1-flash-lite (via OpenRouter), the
same model CLIPPER used.  Results append to scripts/rulechef-lc-out/results.csv
with the learning-curve columns plus `coverage`.

CONFIGURATION.  RuleChef defaults: max_rules=10, max_rules_per_class=5,
max_samples=50, three refinement iterations -- as for the imodels baselines,
we use library defaults.  `--max-rules`/`--max-samples` exist for sensitivity
checks; leaving them unset is the reported condition.

SCORING.  RuleChef's rules abstain when nothing matches.  An abstention is
scored as the majority class of the cell's training subsample, the analogue of
SLIPPER's default rule.  The `coverage` column records how often a rule fired.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(REPO, "text_suite")
DATASETS = ["sentiment_amazon", "sentiment_imdb", "sentiment_yelp",
            "sms_spam", "youtube_spam"]
SIZES = [50, 100, 200, 300, 400, 500, 750, 1000]
COLS = ["label", "dataset", "size", "seed", "bestj", "err", "rules",
        "coverage", "llm_calls", "latency_s", "in_tok", "out_tok", "cost_usd"]
# gemini-3.1-flash-lite via OpenRouter: the same model CLIPPER used on this
# suite, so the comparison isolates the method rather than the model.
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
PRICE = {"in": 0.25e-6, "out": 1.50e-6}      # $/token, gemini-3.1-flash-lite

_lock = threading.Lock()
_cache = {}
_cache_lock = threading.Lock()


def spec_for(ds):
    with _cache_lock:
        if ds not in _cache:
            with open(os.path.join(SUITE, ds + ".json"), encoding="utf-8") as fp:
                _cache[ds] = json.load(fp)
        return _cache[ds]


def subsample(train, ds, size, seed):
    """Byte-identical to run_learning_curve.subsample_train."""
    h = int(hashlib.md5(f"{ds}|{size}|{seed}".encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(h).sample(train, size)


def text_index(attrs):
    for i, a in enumerate(attrs):
        if a.get("kind") == "text":
            return i
    raise SystemExit("no text attribute in this dataset")


class Tally:
    """Wrap the OpenAI client to count tokens; rulechef surfaces no usage."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = self.pin = self.pout = 0
        self.chat = self
        self.completions = self

    def create(self, *a, **kw):
        with _lock:
            self.calls += 1
        r = self._inner.chat.completions.create(*a, **kw)
        u = getattr(r, "usage", None)
        if u:
            self.pin += getattr(u, "prompt_tokens", 0) or 0
            self.pout += getattr(u, "completion_tokens", 0) or 0
        return r

    def __getattr__(self, n):
        return getattr(self._inner, n)


def run_cell(ds, size, seed, args):
    from openai import OpenAI
    from rulechef import Task, TaskType
    from rulechef.engine import RuleChef

    spec = spec_for(ds)
    ti = text_index(spec["attributes"])
    train = subsample(spec["train"], ds, size, seed)
    test = spec["test"]

    client = Tally(OpenAI(base_url=args.base_url,
                          api_key=os.environ["OPENROUTER_API_KEY"]))
    task = Task(name="%s classification" % ds,
                description="Classify the document into one of: %s"
                            % ", ".join(spec["classes"]),
                input_schema={"text": "str"}, output_schema={"label": "str"},
                type=TaskType.CLASSIFICATION, text_field="text")

    work = tempfile.mkdtemp(prefix="rc_")
    t0 = time.time()
    try:
        # rulechef prints a progress narrative per cell; keep it out of the
        # harness log (and out of reach of the console encoding).
        with contextlib.redirect_stdout(io.StringIO()):
            chef = RuleChef(task, client, storage_path=work, model=args.model,
                            use_grex=False, max_rules=args.max_rules,
                            max_rules_per_class=args.max_rules_per_class,
                            max_samples=args.max_samples)
            for r in train:
                chef.add_example({"text": r["values"][ti] or ""},
                                 {"label": r["label"]})
            chef.learn_rules()

            labels = [r["label"] for r in train]
            fallback = max(set(labels), key=labels.count)
            wrong = fired = 0
            for r in test:
                try:
                    out = chef.extract({"text": r["values"][ti] or ""})
                    pred = (out or {}).get("label")
                except Exception:
                    pred = None
                if pred:
                    fired += 1
                else:
                    pred = fallback
                wrong += (pred != r["label"])
            summary = chef.get_rules_summary()
    except Exception as e:
        sys.stderr.write("[FAIL] %s size=%d seed=%d: %s: %s\n"
                         % (ds, size, seed, type(e).__name__, str(e)[:200]))
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)

    n_rules = len(summary) if isinstance(summary, list) else summary
    return {
        "label": args.label, "dataset": ds, "size": size, "seed": seed,
        "bestj": 0, "err": round(100.0 * wrong / len(test), 2),
        "rules": n_rules, "coverage": round(100.0 * fired / len(test), 1),
        "llm_calls": client.calls, "latency_s": round(time.time() - t0, 1),
        "in_tok": client.pin, "out_tok": client.pout,
        "cost_usd": round(client.pin * PRICE["in"] + client.pout * PRICE["out"], 5),
    }


def load_done(path, label):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fp:
            for row in csv.DictReader(fp):
                if row["label"] == label:
                    done.add((row["dataset"], int(row["size"]), int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="rulechef")
    ap.add_argument("--out", default=os.path.join(REPO, "scripts", "rulechef-lc-out"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="keep modest: concurrent LLM grids have throttled each "
                         "other into degenerate cells before (default %(default)s)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    # Capacity knobs -- stock rulechef defaults; see CONFIGURATION above.
    ap.add_argument("--max-rules", type=int, default=10)
    ap.add_argument("--max-rules-per-class", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=50)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set")
    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                if args.datasets else DATASETS)
    sizes = ([int(s) for s in args.sizes.split(",") if s.strip()]
             if args.sizes else SIZES)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.csv")
    done = load_done(results_path, args.label)
    trainlen = {d: len(spec_for(d)["train"]) for d in datasets}
    tasks = [(d, z, s) for d in datasets for z in sizes for s in range(1, args.seeds + 1)
             if z <= trainlen[d] and (d, z, s) not in done]
    dropped = sorted({(d, z) for d in datasets for z in sizes if z > trainlen[d]})
    if dropped:
        print("note: dropping %d size points beyond available train rows: %s"
              % (len(dropped), ", ".join("%s@%d" % p for p in dropped)), flush=True)
    print("rulechef '%s' (model %s, caps %d/%d/%d): %d to run, %d already done "
          "(%d-way)" % (args.label, args.model, args.max_rules,
                        args.max_rules_per_class, args.max_samples,
                        len(tasks), len(done), args.workers), flush=True)

    with open(os.path.join(args.out, "%s_config.yaml" % args.label), "w") as fp:
        json.dump({"label": args.label, "model": args.model,
                   "base_url": args.base_url, "seeds": args.seeds,
                   "sizes": sizes, "datasets": datasets,
                   "max_rules": args.max_rules,
                   "max_rules_per_class": args.max_rules_per_class,
                   "max_samples": args.max_samples,
                   "rulechef_version": "0.2.2",
                   "scoring": "abstention -> train-majority class",
                   "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, fp, indent=2)

    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as fp:
            csv.writer(fp).writerow(COLS)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, d, z, s, args): (d, z, s) for d, z, s in tasks}
        for fut in as_completed(futs):
            d, z, s = futs[fut]
            row = fut.result()
            n += 1
            if not row:
                continue
            with _lock, open(results_path, "a", newline="") as fp:
                csv.writer(fp).writerow([row[c] for c in COLS])
            print("  [%d/%d] %-18s size%-5d seed%-3d rules%-4s cov%5.1f%% -> %5.2f"
                  % (n, len(tasks), d, z, s, row["rules"], row["coverage"],
                     row["err"]), flush=True)
    print("suite complete", flush=True)


if __name__ == "__main__":
    main()
