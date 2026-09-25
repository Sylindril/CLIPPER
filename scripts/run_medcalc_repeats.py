#!/usr/bin/env python3
"""Frozen, resumable ten-seed MedCalc experiments and seed-level summaries.

prepare freezes inputs, learner sources, resolved settings, and the complete grid.
run uses isolated processes, separate CPU/LLM queues, and durable attempt logs.
Workers fit on training rows only, then evaluate the same hypothesis on both splits.
summarize includes only complete seed ensembles and never treats keys as replicates.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/tools")]
DEFAULT_OUT = REPO / "runs/medcalc"
EXTRACT_ROOT = REPO / "datasets/medcalc_extract"   # build_medcalc_extraction_suite.py output
POOLS = REPO / "experiments/medcalc/pools"            # hand-written reference regex pools
EXPECTED = REPO / "experiments/medcalc/expected_hashes.json"
CAPS = [1, 4, 16, 64]
SEEDS = list(range(1, 11))
MODEL = "gemini/gemini-3.1-flash-lite"
PROTOCOL = 3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def choose_keys(counts, n=10):
    return sorted(counts, key=lambda k: (-counts[k]["train"], k))[:n]


def make_conditions(model):
    from slipper import config
    common = config.config_dict()
    common.update(Slipper_folds=0, Maximize_criterion=False, Claimed_tool_budget=None,
                  Weighting="boost", Upward_trend_max=0, Log_stem=None,
                  Prompt_file=None, Advice_ext=None, Advice_text=None)
    # Advice_text is runtime-only and must not be included in a serialized config.
    common.pop("Advice_text", None)
    result = {}
    for name, sample, llm in [("slipper_full", 0, False), ("slipper_M20", 20, False),
                              ("slipper_M100", 100, False),
                              ("slopper_raw_M20", 20, True), ("slopper_raw_M100", 100, True)]:
        result[name] = dict(common, Sample_size=sample, N_boost=6, Stratify=True,
                            Rule_llm_model=model if llm else "", Tool_budget=6 if llm else 0)
    for body in ("python", "regex"):
        for sample in (20, 55, 110):
            result[f"extract_{body}_M{sample}"] = dict(
                common, Sample_size=sample, N_boost=10, Stratify=False,
                Rule_llm_model=model, Tool_budget=0, Rule_body=body)
    result["extract_pool"] = dict(common, Sample_size=0, N_boost=10, Stratify=False,
                                  Rule_llm_model="", Tool_budget=0)
    return result


def prepare(out, model=MODEL):
    from build_medcalc_suite import (HF_REPO, HF_REVISION, split_train, build_vocab,
                                    tokenize, make_rows, content_hash)
    import datasets
    out = Path(out).resolve()
    if (out / "manifest.json").exists():
        manifest = read_json(out / "manifest.json")
        assert manifest["model"] == model, "model differs from the frozen run"
        verify_inputs(out, manifest)
        print(f"Frozen run already prepared: {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    expected = read_json(EXPECTED)
    data = datasets.load_dataset(HF_REPO, revision=HF_REVISION)
    gold = {s: [int(c) for c in data[s]["Calculator ID"]] for s in ("train", "test")}
    calculators = sorted(set(gold["test"]))
    assert len(calculators) == 55
    inputs, cells, reference_dev = {}, [], None
    for cap in CAPS:
        pool, held = split_train(gold["train"], cap, 10, 1)
        if reference_dev is None:
            reference_dev = held
        assert held == reference_dev, "development rows changed across caps"
        for representation in ("tokens", "raw"):
            raw = representation == "raw"
            columns = {} if raw else {"WORDS": "Patient Note"}
            raw_columns = {"NOTE": "Patient Note"} if raw else {}
            vocabs = {} if raw else {"WORDS": build_vocab(
                [tokenize(data["train"]["Patient Note"][i]) for i in pool], 5, 5000)}
            attrs = ([{"name": "NOTE", "kind": "text"}] if raw
                     else [{"name": "WORDS", "kind": "set"}])
            train = make_rows(data["train"], pool, columns, vocabs, raw_columns)
            dev = make_rows(data["train"], held, columns, vocabs, raw_columns)
            final = make_rows(data["test"], range(len(gold["test"])), columns, vocabs, raw_columns)
            for name, rows in (("train", train), ("dev_test", dev), ("final_test", final)):
                assert content_hash(rows) == expected["row_content"][f"{cap}/{representation}"][name], \
                    ("rows differ from the paper's", cap, representation, name)
            bundle = {"attributes": attrs, "train": train, "train_gold": [gold["train"][i] for i in pool],
                      "dev": dev, "dev_gold": [gold["train"][i] for i in held],
                      "final": final, "final_gold": gold["test"],
                      "row_ids": {"train": pool, "dev": held, "final": list(range(len(gold["test"])))}}
            path = f"inputs/classification_cap{cap}_{representation}.json"
            write_json(out / path, bundle)
            inputs[path] = file_hash(out / path)
            assert inputs[path] == expected["input_files"][path], f"{path} differs from the paper's input"
            conditions = (["slopper_raw_M20", "slopper_raw_M100"] if raw
                          else ["slipper_full", "slipper_M20", "slipper_M100"])
            for condition in conditions:
                for seed in SEEDS:
                    for calc in calculators:
                        cells.append(dict(id=f"classification/{condition}/cap{cap}/seed{seed:02d}/calc_{calc}",
                                          kind="classification", condition=condition, seed=seed,
                                          cap=cap, target=str(calc), input=path))
    extract_root = EXTRACT_ROOT
    extract_manifest = read_json(extract_root / "manifest.json")
    keys = choose_keys(extract_manifest["instance_counts"])
    key_counts = {key: extract_manifest["instance_counts"][key] for key in keys}
    for key in keys:
        dev = read_json(extract_root / f"dev/{key}.json")
        final = read_json(extract_root / f"final/{key}.json")
        assert dev["train"] == final["train"] and dev["attributes"] == final["attributes"]
        bundle = {"task": "extract", "attributes": dev["attributes"], "train": dev["train"],
                  "dev": dev["test"], "final": final["test"]}
        path = f"inputs/extraction_{key}.json"
        write_json(out / path, bundle)
        inputs[path] = file_hash(out / path)
        assert inputs[path] == expected["input_files"][path], f"{path} differs from the paper's input"
        pool_path = f"inputs/pools/{key}.pool.json"
        (out / pool_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(POOLS / f"{key}.pool.json", out / pool_path)
        inputs[pool_path] = file_hash(out / pool_path)
        for condition in [f"extract_{body}_M{m}" for body in ("python", "regex") for m in (20, 55, 110)] + ["extract_pool"]:
            for seed in ([1] if condition == "extract_pool" else SEEDS):
                cells.append(dict(id=f"extraction/{condition}/seed{seed:02d}/{key}", kind="extraction",
                                  condition=condition, seed=seed, cap=150, target=key, input=path,
                                  pool=pool_path if condition == "extract_pool" else None))
    code = out / "code"
    shutil.copytree(REPO / "src/slipper", code / "src/slipper", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "scripts").mkdir(parents=True, exist_ok=True)
    (code / "src/tools").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, code / "scripts/run_medcalc_repeats.py")
    shutil.copyfile(REPO / "scripts/plot_medcalc_repeats.py", code / "scripts/plot_medcalc_repeats.py")
    shutil.copyfile(REPO / "scripts/medcalc_regex_runtime.py", code / "scripts/medcalc_regex_runtime.py")
    shutil.copyfile(REPO / "src/tools/score_extraction.py", code / "src/tools/score_extraction.py")
    sources = {str(p.relative_to(out)): file_hash(p) for p in sorted(code.rglob("*.py"))}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "smolagents", "litellm", "pydantic-ai", "pydantic-ai-litellm", "datasets")}
    manifest = dict(protocol=PROTOCOL, model=model, caps=CAPS, seeds=SEEDS, calculators=calculators,
                    regex_timeout_s=1.0, python_timeout_s=1.0,
                    extraction_keys=keys, extraction_counts=key_counts,
                    data_revision=HF_REVISION, data_seed=1, conditions=make_conditions(model),
                    inputs=inputs, sources=sources, packages=versions, cells=cells,
                    git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                              capture_output=True).stdout.strip() or None,
                    seed_protocol="Fixed data rows; seeds control local learner RNGs. Provider seed/temperature unset.",
                    inference_protocol="Fit with no evaluation examples loaded; score same hypothesis on dev and final.",
                    runtime_protocol="Existing learner, prompts, retries and exception-as-abstention semantics; exceptions logged.")
    manifest["identity"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(cells)} cells; extraction keys: {', '.join(keys)}", flush=True)


def verify_inputs(out, manifest):
    payload = dict(manifest)
    identity = payload.pop("identity")
    assert digest(payload) == identity, "frozen manifest was changed"
    for path, expected in {**manifest["inputs"], **manifest["sources"]}.items():
        assert file_hash(out / path) == expected, f"frozen file changed: {path}"


def training_spec(bundle, cell):
    spec = {"attributes": bundle["attributes"]}
    if cell["kind"] == "classification":
        pos = "calc=" + cell["target"]
        spec["classes"] = [pos, "calc=rest"]
        spec["train"] = [dict(row, label=pos if g == int(cell["target"]) else "calc=rest")
                         for row, g in zip(bundle["train"], bundle["train_gold"])]
    else:
        spec.update(task="extract", train=bundle["train"])
    return spec


def evaluation_examples(rows):
    """Convert rows without changing fitted domains or grammar symbol types.

    Grammar construction can reuse interned words such as 'body' as grammar
    symbols. The training loader's domain registration must not run again.
    """
    from slipper import dataset, names
    from slipper.data import Aval, Example
    from slipper.names import AttrKind
    from slipper.symbols import Atom, intern
    examples = []
    for index, row in enumerate(rows):
        inst = []
        assert len(row["values"]) == len(names.Names)
        for definition, value in zip(names.Names, row["values"]):
            kind = definition.kind
            if value is None or kind == AttrKind.IGNORE:
                inst.append(Aval(AttrKind.MISSING_VALUE))
            elif kind == AttrKind.CONTINUOUS:
                inst.append(Aval(kind, num=float(value)))
            elif kind == AttrKind.SYMBOL:
                inst.append(Aval(kind, sym=intern(str(value))))
            elif kind == AttrKind.TEXT:
                inst.append(Aval(kind, text=str(value)))
            else:
                values = [intern(str(v)) for v in value]
                if definition.isbag:
                    values = dataset._bagify(values)
                inst.append(Aval(kind, set=values))
        examples.append(Example(inst=inst, lab=Atom(nom=intern(str(row["label"]))),
                                wt=float(row.get("weight", 1.0)), index=index))
    return examples


def worker(out, cell_id, attempt):
    from slipper import boosting, config, dataset, extract, grammar, record, rng, trace
    from slipper import python_emit
    out, attempt = Path(out).resolve(), Path(attempt).resolve()
    manifest = read_json(out / "manifest.json")
    cell = next(c for c in manifest["cells"] if c["id"] == cell_id)
    assert file_hash(out / cell["input"]) == manifest["inputs"][cell["input"]]
    bundle = read_json(out / cell["input"])
    regex_runtime = replay_state = None
    python_runtime = None
    if manifest.get("regex_timeout_s") and cell["kind"] == "extraction":
        from medcalc_regex_runtime import install_limit, install_replay
        regex_runtime = install_limit(manifest["regex_timeout_s"])
    if manifest.get("python_timeout_s") and cell["kind"] == "extraction":
        from medcalc_regex_runtime import install_python_limit
        python_runtime = install_python_limit(manifest["python_timeout_s"])
        if cell.get("replay"):
            replay_state = install_replay(out / cell["replay"])
    attempt.mkdir(parents=True, exist_ok=True)
    spec = training_spec(bundle, cell)
    path = attempt / "train.json"
    write_json(path, spec)
    for key, value in manifest["conditions"][cell["condition"]].items():
        setattr(config, key, value)
    if cell.get("pool"):
        config.Rule_pool = str(out / cell["pool"])
    config.Log_stem = str(attempt / "rule") if config.Rule_llm_model else None
    rng.seed(cell["seed"])
    trace.set_trace_level(trace.SUMM)
    config.save_yaml(attempt / "resolved_config.yaml")
    write_json(attempt / "cell.json", dict(cell, manifest_identity=manifest["identity"]))
    train, test = dataset.load_dataset(path)
    assert test is None
    if cell["kind"] == "classification":
        if not grammar.ld_grammar(train) and not config.Rule_llm_model:
            raise RuntimeError("No discrete grammar")
    records = {}
    start = time.monotonic()
    try:
        with record.recorder() as records:
            if cell.get("replay"):
                previous_usage = read_json_lines(out / cell["replay"] / "usage.jsonl")[-1]
                records.update({key: previous_usage[key] for key in records})
            original_note = record.note_llm_call

            def note_call(*args, **kwargs):
                original_note(*args, **kwargs)
                with (attempt / "usage.jsonl").open("a") as f:
                    f.write(json.dumps(dict(records, at=time.time())) + "\n")

            record.note_llm_call = note_call
            try:
                hyp = boosting.soft_model(train, None)
            finally:
                record.note_llm_call = original_note
        if config.Rule_llm_model and not records["n_llm_calls"]:
            raise RuntimeError("No successful LLM requests; refusing to count an infrastructure failure as a seed")
        with (attempt / "model.shyp").open("w") as f:
            boosting.fshow_soft_concept(f, hyp)
        python_emit.write_python(str(attempt / "model.pyhyp.py"), hyp)
        results = dict(cell, manifest_identity=manifest["identity"], records=records,
                       rules=len(hyp.rules), elapsed_s=time.monotonic()-start,
                       attempt=str(attempt.relative_to(out)), splits={})
        if replay_state is not None:
            results["proposal_replay"] = replay_state
        if cell["kind"] == "classification":
            # soft_classify's positive orientation is opposite hyp.def_.
            default_alpha = boosting._alphah(hyp.nposx, hyp.nnegx)
            if hyp.def_.nom.name == "calc=" + cell["target"]:
                default_alpha = -default_alpha
            results["default_alpha"] = default_alpha
        for split in ("dev", "final"):
            rows = bundle[split]
            if cell["kind"] == "classification":
                pos = "calc=" + cell["target"]
                rows = [dict(r, label=pos if g == int(cell["target"]) else "calc=rest")
                        for r, g in zip(rows, bundle[split + "_gold"])]
            examples = evaluation_examples(rows)
            boosting._round_exceptions.clear()
            pred = []
            with (attempt / (split + ".pred")).open("w") as f:
                for row, ex in zip(rows, examples):
                    if cell["kind"] == "extraction":
                        value, confidence = extract.extract_classify(hyp, extract.inst_dict(ex.inst))
                        f.write(f"{value!r}\t{confidence:.17g}\t0\t{ex.lab.nom.name}\n")
                        pred.append(value)
                    else:
                        label, confidence = boosting.soft_classify(hyp, ex.inst)
                        score = abs(confidence) if label.name == pos else -abs(confidence)
                        f.write(f"{label.name} {abs(confidence):.17g} 0 {ex.lab.nom.name}\n")
                        pred.append(score)
            write_json(attempt / (split + ".predictions.json"), pred)
            results["splits"][split] = dict(n=len(pred), exceptions=sum(boosting._round_exceptions.values()),
                exception_types={f"{typ}: {msg}": n for (typ,msg),n in boosting._round_exceptions.items()})
        if regex_runtime is not None:
            results["regex_runtime"] = {k: v for k, v in regex_runtime.items() if k != "restore"}
        if python_runtime is not None:
            results["python_runtime"] = {k: v for k, v in python_runtime.items() if k != "restore"}
        write_json(attempt / "result.json", results)
        print(json.dumps({"completed": cell_id, "cost_usd": records["cost_usd"],
                          "rules": len(hyp.rules), "seconds": results["elapsed_s"]}), flush=True)
    except BaseException as e:
        write_json(attempt / "failure.json", dict(cell=cell_id, error=f"{type(e).__name__}: {e}",
                   records=records, elapsed_s=time.monotonic()-start))
        raise


def load_completed(out, manifest):
    result = {}
    for cell in manifest["cells"]:
        path = out / "cells" / cell["id"] / "result.json"
        if path.exists():
            row = read_json(path)
            assert row["manifest_identity"] == manifest["identity"], f"stale result: {path}"
            assert row["id"] == cell["id"]
            for split in ("dev", "final"):
                assert (out / row["attempt"] / (split + ".predictions.json")).exists()
            result[cell["id"]] = row
    return result


def run_one(out, manifest, cell, timeout):
    root = out / "cells" / cell["id"]
    root.mkdir(parents=True, exist_ok=True)
    number = len(list(root.glob("attempt_*"))) + 1
    attempt = root / f"attempt_{number:03d}"
    attempt.mkdir()
    command = [sys.executable, str(out / "code/scripts/run_medcalc_repeats.py"),
               "worker", "--out", str(out), "--cell-id", cell["id"], "--attempt", str(attempt)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    with (attempt / "stdout.txt").open("w") as stdout, (attempt / "stderr.txt").open("w") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, cwd=out)
        write_json(attempt / "process.json", {"pid":proc.pid,"started_at":time.time(),"command":command})
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            return {"failed": cell["id"], "error": "worker timeout", "attempt":str(attempt.relative_to(out))}
    if rc != 0 or not (attempt / "result.json").exists():
        failure = read_json(attempt / "failure.json") if (attempt / "failure.json").exists() else {"error":f"exit {rc}"}
        return dict(failure, failed=cell["id"], attempt=str(attempt.relative_to(out)))
    result = read_json(attempt / "result.json")
    assert result["manifest_identity"] == manifest["identity"]
    write_json(root / "result.json", result)
    # Training rows are already in the frozen bundle, so keep one shared copy.
    (attempt / "train.json").unlink()
    return result


def status(out, manifest, completed, running=(), failures=(), started=None):
    count = collections.Counter(row["condition"] for row in completed.values())
    paid = sum(r["records"]["cost_usd"] for r in completed.values())
    report = dict(completed=len(completed), total=len(manifest["cells"]), by_condition=dict(count),
                  successful_calls_cost_usd=paid,
                  llm_calls=sum(r["records"]["n_llm_calls"] for r in completed.values()),
                  cost_unavailable=any(r["records"]["cost_unavailable"] for r in completed.values()),
                  running=list(running), failures=list(failures), updated_at=time.time(), started_at=started)
    write_json(out / "status.json", report)
    return report


def run(out, cpu_jobs=4, llm_jobs=8, pilot=False, timeout=3600):
    import fcntl
    out = Path(out).resolve()
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(out / "manifest.json")
    verify_inputs(out, manifest)
    completed = load_completed(out, manifest)
    cells = [c for c in manifest["cells"] if c["id"] not in completed]
    if pilot:
        picked = []
        for condition in ("slipper_full", "slopper_raw_M20", "extract_python_M20", "extract_regex_M20"):
            eligible = [c for c in cells if c["condition"] == condition]
            if eligible:
                picked.append(eligible[0])
        cells = picked
    # Finish extraction first in the model queue, then complete classifier ensembles.
    cells.sort(key=lambda c: (c["kind"] != "extraction", c["cap"],
                              manifest["conditions"][c["condition"]]["Sample_size"],
                              c["condition"], c["seed"], c["target"]))
    queues = {True:collections.deque(), False:collections.deque()}
    for cell in cells:
        queues[bool(manifest["conditions"][cell["condition"]]["Rule_llm_model"])].append(cell)
    limits = {True:llm_jobs, False:cpu_jobs}
    pools = {kind:futures.ThreadPoolExecutor(max_workers=n) for kind,n in limits.items()}
    pending, failures, consecutive = {}, [], {True:0,False:0}
    started, last_report = time.time(), 0
    print(f"Starting {'pilot' if pilot else 'grid'}: {len(cells)} pending, {len(completed)} complete; "
          f"CPU workers={cpu_jobs}, LLM workers={llm_jobs}", flush=True)
    try:
        while pending or any(queues.values()):
            if (out / "STOP").exists():
                for q in queues.values(): q.clear()
            for kind in queues:
                active = sum(k == kind for _,k in pending.values())
                while queues[kind] and active < limits[kind]:
                    cell = queues[kind].popleft()
                    future = pools[kind].submit(run_one,out,manifest,cell,timeout)
                    pending[future] = (cell,kind)
                    active += 1
            if not pending: break
            done, _ = futures.wait(pending,timeout=15,return_when=futures.FIRST_COMPLETED)
            for future in done:
                cell,kind = pending.pop(future)
                try: row=future.result()
                except Exception as e: row={"failed":cell["id"],"error":f"{type(e).__name__}: {e}"}
                if "failed" in row:
                    failures.append(row); consecutive[kind]+=1
                    print("FAILED " + json.dumps(row),flush=True)
                    # Stop dispatching that queue if the provider or runner is broken.
                    if consecutive[kind] >= 3:
                        queues[kind].clear()
                        print(f"Stopped {'LLM' if kind else 'CPU'} queue after three consecutive failures",flush=True)
                else:
                    consecutive[kind]=0
                    completed[cell["id"]]=row
                    print(f"DONE {len(completed)}/{len(manifest['cells'])} {cell['id']} "
                          f"${row['records']['cost_usd']:.4f} {row['elapsed_s']:.1f}s",flush=True)
            if time.time()-last_report > 30 or not pending:
                report=status(out,manifest,completed,[c["id"] for c,_ in pending.values()],failures,started)
                print(f"STATUS {report['completed']}/{report['total']} complete; "
                      f"recorded successful calls ${report['successful_calls_cost_usd']:.2f}",flush=True)
                last_report=time.time()
    finally:
        for pool in pools.values(): pool.shutdown(wait=True)
        status(out,manifest,completed,failures=failures,started=started)
    summarize(out)
    if len(completed) == len(manifest["cells"]):
        subprocess.run([sys.executable, str(out / "code/scripts/plot_medcalc_repeats.py"),
                        "--out", str(out)], check=True)
    if failures:
        raise SystemExit(f"{len(failures)} failed cells; inspect status.json and rerun to resume")


def mean_sem(values):
    return statistics.mean(values), statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None


def extraction_rows(bundles, predictions, split):
    """Reconstruct selected-key row sets with the same positional join as the scorer."""
    from slipper.normalize import parse_value
    rows = collections.defaultdict(dict)
    for key, bundle in bundles.items():
        seen=collections.Counter()
        assert len(bundle[split]) == len(predictions[key])
        for row,pred in zip(bundle[split],predictions[key]):
            _,calc,entity=row["values"]
            seen[calc]+=1
            rows[(calc,seen[calc])][entity]=(parse_value(row["label"]),pred)
    return rows


def summarize(out):
    from score_extraction import score
    out=Path(out).resolve()
    manifest=read_json(out/"manifest.json")
    complete=load_completed(out,manifest)
    groups=collections.defaultdict(list)
    for row in complete.values(): groups[(row["kind"],row["condition"],row["cap"],row["seed"])].append(row)
    cls_rows,ext_rows=[],[]
    bundle_cache={}
    for (kind,condition,cap,seed),group in sorted(groups.items()):
        expected=55 if kind=="classification" else len(manifest["extraction_keys"])
        if len(group)!=expected: continue
        for row in group:
            if row["input"] not in bundle_cache: bundle_cache[row["input"]]=read_json(out/row["input"])
        for split in ("dev","final"):
            preds={row["target"]:read_json(out/row["attempt"]/(split+".predictions.json")) for row in group}
            if kind=="classification":
                ids=sorted(map(int,preds))
                alphas={int(row["target"]):row["default_alpha"] for row in group}
                gold=bundle_cache[group[0]["input"]][split+"_gold"]
                centered=[max(ids,key=lambda c:preds[str(c)][i]-alphas[c]) for i in range(len(gold))]
                raw=[max(ids,key=lambda c:preds[str(c)][i]) for i in range(len(gold))]
                f1=[]
                for cid in ids:
                    tp=sum(g==p==cid for g,p in zip(gold,centered))
                    fp=sum(p==cid and g!=cid for g,p in zip(gold,centered))
                    fn=sum(g==cid and p!=cid for g,p in zip(gold,centered))
                    f1.append(2*tp/(2*tp+fp+fn) if tp else 0.0)
                cls_rows.append(dict(condition=condition,cap=cap,seed=seed,split=split,n_eval=len(gold),
                    accuracy=100*sum(g==p for g,p in zip(gold,centered))/len(gold),
                    macro_f1=100*statistics.mean(f1),
                    accuracy_uncentered=100*sum(g==p for g,p in zip(gold,raw))/len(gold)))
                write_csv(out/f"predictions/{condition}_cap{cap}_seed{seed:02d}_{split}.csv",
                          [dict(row=i,gold=g,pred=p,pred_uncentered=r) for i,(g,p,r) in enumerate(zip(gold,centered,raw))])
            else:
                bundles={row["target"]:bundle_cache[row["input"]] for row in group}
                s=score(extraction_rows(bundles,preds,split))
                ext_rows.append(dict(condition=condition,M=manifest["conditions"][condition]["Sample_size"],
                    seed=seed,split=split,keys=expected,rows=s["rows"],tp=s["tp"],fp=s["fp"],fn=s["fn"],
                    micro_p=100*s["micro_p"],micro_r=100*s["micro_r"],micro_f1=100*s["micro_f1"],
                    exact_set=100*s["exact_set"]))
    write_csv(out/"classification_seeds.csv",cls_rows)
    write_csv(out/"extraction_seeds.csv",ext_rows)
    summaries=[]
    for kind,rows,axis,metrics in [("classification",cls_rows,"cap",["accuracy","macro_f1"]),
                                  ("extraction",ext_rows,"M",["micro_p","micro_r","micro_f1","exact_set"])]:
        by=collections.defaultdict(list)
        for row in rows: by[(row["condition"],row[axis],row["split"])].append(row)
        for (condition,x,split),reps in sorted(by.items()):
            for metric in metrics:
                mean,sem=mean_sem([r[metric] for r in reps])
                summaries.append(dict(kind=kind,condition=condition,x=x,split=split,metric=metric,
                                      n_seeds=len(reps),mean=mean,sem=sem,
                                      complete=len(reps)==(1 if condition=="extract_pool" else 10)))
    write_csv(out/"summary.csv",summaries)
    print(f"Summarized {len(cls_rows)} classifier seed/splits and {len(ext_rows)} extraction seed/splits",flush=True)
    return summaries


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","run","worker","summarize","status"])
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--model",default=MODEL)
    ap.add_argument("--cpu-jobs",type=int,default=4)
    ap.add_argument("--llm-jobs",type=int,default=8)
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--timeout",type=int,default=3600)
    ap.add_argument("--cell-id")
    ap.add_argument("--attempt",type=Path)
    args=ap.parse_args()
    if args.action=="prepare": prepare(args.out,args.model)
    elif args.action=="run": run(args.out,args.cpu_jobs,args.llm_jobs,args.pilot,args.timeout)
    elif args.action=="worker": worker(args.out,args.cell_id,args.attempt)
    elif args.action=="summarize": summarize(args.out)
    else:
        m=read_json(args.out/"manifest.json")
        print(json.dumps(status(args.out,m,load_completed(args.out,m)),indent=2))


if __name__=="__main__":
    main()
