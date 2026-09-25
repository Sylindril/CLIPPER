#!/usr/bin/env python3
"""SLIPPER, CLIPPER and RuleChef learning curves on the 5-intent BANKING77 subset.

prepare  writes one training subsample per (size, seed) and the cell grid.  The
         subsample uses run_learning_curve.subsample_train, so all three learners
         train on identical rows.  Rerun with more seeds or sizes to extend the grid.
run      runs the pending cells in isolated processes, with separate CPU, LLM and
         RuleChef queues.  Finished cells are skipped, so a rerun resumes.
summarize writes results.csv, one row per complete (label, size, seed).

SLIPPER and CLIPPER use MedCalc's one-vs-rest variant: one two-class fit per
intent (`intent=c` vs `intent=rest`), decoded by the centred argmax
argmax_c(score_c - default_alpha_c).  SLIPPER reads the WORDS token set; CLIPPER
reads the raw TEXT with its coverage tool switched off.  RuleChef runs its own
benchmark script through scripts/run_rulechef_banking77.py in its own interpreter.

    uv run python scripts/run_banking77.py prepare --seeds 1 2 3
    uv run python scripts/run_banking77.py run --rulechef-python PY --only 'rulechef/n100/*'
    uv run python scripts/run_banking77.py summarize
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from run_learning_curve import subsample_train
from run_medcalc_repeats import (MODEL, evaluation_examples, file_hash, make_conditions,
                                 read_json, write_csv, write_json)

DATASET = "banking77_5"
SUITE_JSON = REPO / "text_suite" / f"{DATASET}.json"
DEFAULT_OUT = REPO / "runs" / "banking77"
SIZES = [50, 100, 200, 400, 600]
RULECHEF_MODEL = "gemini-3.1-flash-lite"
# Which attribute each of our learners reads: SLIPPER the token set, CLIPPER the raw text.
ATTRIBUTE = {"slipper_full": "WORDS", "clipper_raw_M100": "TEXT", "clipper_raw_M100_tools": "TEXT"}


def conditions():
    medcalc = make_conditions(MODEL)
    return {"slipper_full": medcalc["slipper_full"],
            "clipper_raw_M100": dict(medcalc["slopper_raw_M100"], Tool_budget=0),
            "clipper_raw_M100_tools": medcalc["slopper_raw_M100"]}


def queue(condition):
    if condition == "rulechef":
        return "rulechef"
    return "llm" if conditions()[condition]["Rule_llm_model"] else "cpu"


def prepare(out, sizes, seeds):
    spec = read_json(SUITE_JSON)
    names = [a["name"] for a in spec["attributes"]]

    def row(r):
        values = dict(zip(names, r["values"]))
        return dict(text=values["TEXT"], words=values["WORDS"], label=r["label"])

    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else dict(
        dataset=DATASET, source=spec["source"], suite_hash=file_hash(SUITE_JSON), classes=spec["classes"],
        conditions=conditions(), rulechef_model=RULECHEF_MODEL, inputs={}, cells=[])
    assert manifest["suite_hash"] == file_hash(SUITE_JSON), "text_suite/banking77_5.json changed"
    for name, settings in manifest["conditions"].items():
        assert conditions()[name] == settings, f"learner settings changed: {name}"
    manifest["conditions"] = conditions()
    known = {c["id"] for c in manifest["cells"]}
    for size in sizes:
        for seed in seeds:
            sub = subsample_train(spec, DATASET, min(size, len(spec["train"])), seed)
            path = f"inputs/n{size}_seed{seed:02d}.json"
            bundle = dict(classes=spec["classes"], train=[row(r) for r in sub["train"]],
                          test=[row(r) for r in spec["test"]])
            if path in manifest["inputs"]:
                assert file_hash(out / path) == manifest["inputs"][path], f"frozen input changed: {path}"
            else:
                write_json(out / path, bundle)
                manifest["inputs"][path] = file_hash(out / path)
            cells = [dict(condition="rulechef", target=None)] + [
                dict(condition=c, target=t) for c in ATTRIBUTE for t in spec["classes"]]
            for c in cells:
                cid = f"{c['condition']}/n{size}/seed{seed:02d}" + (f"/{c['target']}" if c["target"] else "")
                if cid not in known:
                    manifest["cells"].append(dict(c, id=cid, size=size, seed=seed, input=path))
    manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    write_json(out / "manifest.json", manifest)
    print(f"{len(manifest['cells'])} cells over {len(manifest['inputs'])} subsamples", flush=True)


def worker(out, cell_id, attempt):
    """Fit one one-vs-rest SLIPPER/CLIPPER model and score it on the test rows."""
    from slipper import boosting, config, dataset, grammar, record, rng, trace
    manifest = read_json(out / "manifest.json")
    cell = next(c for c in manifest["cells"] if c["id"] == cell_id)
    bundle = read_json(out / cell["input"])
    attr = ATTRIBUTE[cell["condition"]]
    key = {"WORDS": "words", "TEXT": "text"}[attr]
    kind = {"WORDS": "set", "TEXT": "text"}[attr]
    pos = "intent=" + cell["target"]

    def rows(split):
        return [dict(values=[r[key]], label=pos if r["label"] == cell["target"] else "intent=rest")
                for r in bundle[split]]

    path = attempt / "train.json"
    write_json(path, dict(attributes=[dict(name=attr, kind=kind)], classes=[pos, "intent=rest"],
                          train=rows("train")))
    for name, value in manifest["conditions"][cell["condition"]].items():
        setattr(config, name, value)
    config.Log_stem = str(attempt / "rule") if config.Rule_llm_model else None
    rng.seed(cell["seed"])
    trace.set_trace_level(trace.SUMM)
    config.save_yaml(attempt / "resolved_config.yaml")
    train, _ = dataset.load_dataset(path)
    if not config.Rule_llm_model and not grammar.ld_grammar(train):
        raise RuntimeError("No discrete grammar")
    start = time.monotonic()
    with record.recorder() as records:
        hyp = boosting.soft_model(train, None)
    if config.Rule_llm_model and not records["n_llm_calls"]:
        raise RuntimeError("No successful LLM requests; refusing to count an infrastructure failure")
    with (attempt / "model.shyp").open("w") as f:
        boosting.fshow_soft_concept(f, hyp)
    # soft_classify's positive orientation is opposite hyp.def_.
    default_alpha = boosting._alphah(hyp.nposx, hyp.nnegx)
    if hyp.def_.nom.name == pos:
        default_alpha = -default_alpha
    boosting._round_exceptions.clear()
    scores = []
    for ex in evaluation_examples(rows("test")):
        label, confidence = boosting.soft_classify(hyp, ex.inst)
        scores.append(abs(confidence) if label.name == pos else -abs(confidence))
    write_json(attempt / "test.predictions.json", scores)
    write_json(attempt / "result.json", dict(
        cell, records=records, rules=len(hyp.rules), default_alpha=default_alpha,
        elapsed_s=time.monotonic() - start, attempt=str(attempt.relative_to(out)),
        exceptions={f"{t}: {m}": n for (t, m), n in boosting._round_exceptions.items()}))


def run_one(out, cell, rulechef_python, timeout):
    root = out / "cells" / cell["id"]
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
    attempt.mkdir()
    if cell["condition"] == "rulechef":
        command = [rulechef_python, str(REPO / "scripts/run_rulechef_banking77.py"),
                   "--input", str(out / cell["input"]), "--seed", str(cell["seed"]),
                   "--out", str(attempt), "--model", RULECHEF_MODEL]
    else:
        command = [sys.executable, __file__, "worker", "--out", str(out),
                   "--cell-id", cell["id"], "--attempt", str(attempt)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    with (attempt / "stdout.txt").open("w") as stdout, (attempt / "stderr.txt").open("w") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, cwd=REPO)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return dict(failed=cell["id"], error="timeout")
    if rc != 0 or not (attempt / "result.json").exists():
        tail = (attempt / "stderr.txt").read_text().strip().splitlines()[-1:]
        return dict(failed=cell["id"], error=f"exit {rc}: {' '.join(tail)}")
    result = dict(read_json(attempt / "result.json"), attempt=str(attempt.relative_to(out)))
    write_json(root / "result.json", result)
    (attempt / "train.json").unlink(missing_ok=True)
    return result


def run(out, rulechef_python, jobs, only, timeout):
    manifest = read_json(out / "manifest.json")
    cells = [c for c in manifest["cells"] if not (out / "cells" / c["id"] / "result.json").exists()
             and (not only or any(fnmatch.fnmatch(c["id"], p) for p in only))]
    assert rulechef_python or all(c["condition"] != "rulechef" for c in cells), "--rulechef-python required"
    by_queue = collections.defaultdict(collections.deque)
    for cell in cells:
        by_queue[queue(cell["condition"])].append(cell)
    pools = {q: futures.ThreadPoolExecutor(max_workers=jobs[q]) for q in by_queue}
    print(f"{len(cells)} pending: " + ", ".join(f"{q}={len(d)}" for q, d in by_queue.items()), flush=True)
    pending, failures, consecutive = {}, [], collections.Counter()
    for q, cells_q in by_queue.items():
        for cell in cells_q:
            pending[pools[q].submit(run_one, out, cell, rulechef_python, timeout)] = (cell, q)
    try:
        for future in futures.as_completed(pending):
            if future.cancelled():
                continue
            cell, q = pending[future]
            row = future.result()
            if "failed" in row:
                failures.append(row)
                consecutive[q] += 1
                print("FAILED " + json.dumps(row), flush=True)
                # A broken provider or harness fails every cell; stop feeding that queue.
                if consecutive[q] >= 3:
                    for f, (_, fq) in pending.items():
                        if fq == q:
                            f.cancel()
            else:
                consecutive[q] = 0
                note = {k: row[k] for k in ("err", "coverage", "rules_by_iteration", "llm_errors") if k in row}
                print(f"DONE {cell['id']} rules={row['rules']} {json.dumps(note)}", flush=True)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    summarize(out)
    if failures:
        raise SystemExit(f"{len(failures)} failed cells; rerun to resume")


def llm_cost(prompt_tokens, completion_tokens):
    import litellm
    return sum(litellm.cost_per_token(model=MODEL, prompt_tokens=prompt_tokens,
                                      completion_tokens=completion_tokens))


def summarize(out):
    manifest = read_json(out / "manifest.json")
    classes = manifest["classes"]
    groups = collections.defaultdict(list)
    for cell in manifest["cells"]:
        path = out / "cells" / cell["id"] / "result.json"
        if path.exists():
            groups[(cell["condition"], cell["size"], cell["seed"])].append(read_json(path))
    rows = []
    for (condition, size, seed), group in sorted(groups.items()):
        test = read_json(out / f"inputs/n{size}_seed{seed:02d}.json")["test"]
        if condition == "rulechef":
            r, = group
            rows.append(dict(label=condition, size=size, seed=seed, err=r["err"], err_strict=r["err_strict"],
                             coverage=r["coverage"], rules=r["rules"], llm_calls=r["llm_calls"],
                             cost_usd=llm_cost(r["prompt_tokens"], r["completion_tokens"])))
            continue
        if len(group) != len(classes):
            continue
        by_target = {r["target"]: r for r in group}
        scores = {c: read_json(out / by_target[c]["attempt"] / "test.predictions.json") for c in classes}
        pred = [max(classes, key=lambda c: scores[c][i] - by_target[c]["default_alpha"])
                for i in range(len(test))]
        wrong = sum(p != r["label"] for p, r in zip(pred, test))
        rows.append(dict(label=condition, size=size, seed=seed, err=100 * wrong / len(test),
                         err_strict="", coverage="", rules=sum(r["rules"] for r in group),
                         llm_calls=sum(r["records"]["n_llm_calls"] for r in group),
                         cost_usd=sum(r["records"]["cost_usd"] for r in group)))
    write_csv(out / "results.csv", rows)
    for (label, size), reps in sorted(collections.Counter((r["label"], r["size"]) for r in rows).items()):
        errs = [r["err"] for r in rows if (r["label"], r["size"]) == (label, size)]
        print(f"{label:18s} n={size:<4d} seeds={reps:<2d} err={sum(errs) / len(errs):6.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "run", "worker", "summarize"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--only", nargs="+", help="fnmatch patterns on cell ids")
    ap.add_argument("--rulechef-python")
    ap.add_argument("--cpu-jobs", type=int, default=8)
    ap.add_argument("--llm-jobs", type=int, default=32)
    ap.add_argument("--rulechef-jobs", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--cell-id")
    ap.add_argument("--attempt", type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        prepare(out, args.sizes, args.seeds)
    elif args.action == "run":
        run(out, args.rulechef_python, dict(cpu=args.cpu_jobs, llm=args.llm_jobs, rulechef=args.rulechef_jobs),
            args.only, args.timeout)
    elif args.action == "worker":
        worker(out, args.cell_id, args.attempt.resolve())
    else:
        summarize(out)


if __name__ == "__main__":
    main()
