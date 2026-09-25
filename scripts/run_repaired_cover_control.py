#!/usr/bin/env python3
"""Complete the paper's repaired discrete-covering control on matched cells.

Reuses the recorded training-only CV budgets shared by the historical methods.
Freezes inputs and learner code, checks three historical boosting results before
dispatch, and retains each CLI invocation's configuration, logs, and predictions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["adult", "bank_marketing", "compas", "heloc", "stroke",
            "telco_churn", "water_potability"]
LABEL = "discrete_cover_repaired"
FLAGS = ["-R", "", "-M0", "--weighting", "cover", "--cover-sizing", "gate",
         "--cover-objective", "linear", "--cover-patience", "1"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare(out, sizes=(250, 1200), code_from=None):
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for path, expected in manifest["hashes"].items():
            assert sha(out / path) == expected, path
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    # Budgets come from the existing test-suite rows: SLIPPER, discrete set
    # cover, CLIPPER and CLIPPER set cover must all have run at these sizes and
    # agree on each cell's cross-validated budget.
    rows = list(csv.DictReader((REPO / "scripts/test-lc-out/results.csv").open()))
    relevant = [r for r in rows if r["dataset"] in DATASETS and int(r["size"]) in sizes]
    baseline = [r for r in relevant if r["label"] == "slipper"]
    assert len(baseline) == 10 * len(DATASETS) * len(sizes)
    cells = []
    for row in sorted(baseline, key=lambda r: (r["dataset"], int(r["size"]), int(r["seed"]))):
        ds, size, seed, bestj = row["dataset"], int(row["size"]), int(row["seed"]), int(row["bestj"])
        matches = [r for r in relevant if r["dataset"] == ds and int(r["size"]) == size
                   and int(r["seed"]) == seed
                   and r["label"] in ("slipper", "discrete_cover", "slopper", "slopper_cover")]
        assert len(matches) == 4 and {int(r["bestj"]) for r in matches} == {bestj}
        assert bestj >= 2
        spec = json.loads((REPO / f"test_suite/{ds}.json").read_text())
        sampling_seed = int(hashlib.md5(f"{ds}|{size}|{seed}".encode()).hexdigest(), 16) % (2**32)
        spec["train"] = random.Random(sampling_seed).sample(spec["train"], size)
        root = Path(f"inputs/{ds}/size{size}/seed{seed:02d}")
        save(out / root / f"{ds}.json", spec)
        info = REPO / f"test_suite/{ds}.info"
        if info.exists():
            shutil.copyfile(info, out / root / f"{ds}.info")
        cells.append(dict(dataset=ds, size=size, seed=seed, bestj=bestj,
                          baseline_error=float(row["err"]), input=str(root / f"{ds}.json")))
    # A frozen learner snapshot from an earlier run keeps new sizes on the
    # exact code that produced the existing sizes.
    shutil.copytree(code_from or REPO / "src/slipper", out / "code/src/slipper",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "code/scripts").mkdir(parents=True)
    shutil.copyfile(__file__, out / "code/scripts/run_repaired_cover_control.py")
    shutil.copyfile(REPO / "scripts/shyp_size.py", out / "code/scripts/shyp_size.py")
    shutil.copyfile(REPO / "scripts/test-lc-out/results.csv", out / "historical_results.csv")
    hashes = {str(p.relative_to(out)): sha(p) for root in (out / "code", out / "inputs")
              for p in sorted(root.rglob("*")) if p.is_file()}
    hashes["historical_results.csv"] = sha(out / "historical_results.csv")
    manifest = dict(label=LABEL, datasets=DATASETS, sizes=list(sizes), seeds=list(range(1, 11)),
                    flags=FLAGS, cells=cells, hashes=hashes,
                    budget_protocol="Recorded SLIPPER training-only CV budget, minimum two; all four historical methods agree",
                    evaluation_protocol="Ordinary tabular CLI, matching historical loading and scoring",
                    code_source=str(code_from or REPO / "src/slipper"),
                    git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                              capture_output=True).stdout.strip() or None)
    save(manifest_path, manifest)
    return manifest


def run_cell(out, cell, baseline=False):
    label = "reproduction_check" if baseline else LABEL
    root = out / f"cells/{label}/{cell['dataset']}/size{cell['size']}/seed{cell['seed']:02d}"
    result_path = root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
    attempt.mkdir(parents=True)
    data = attempt / (cell["dataset"] + ".json")
    data.symlink_to(out / cell["input"])
    info = (out / cell["input"]).with_suffix(".info")
    if info.exists():
        data.with_suffix(".info").symlink_to(info)
    flags = ["-R", "", "-M0"] if baseline else FLAGS
    command = [sys.executable, "-m", "slipper", "--seed", str(cell["seed"]), *flags,
               "-i0", "-n", str(cell["bestj"]), "-p", "-v1", "--save-config",
               str(attempt / "resolved_config.yaml"), str(data)]
    save(attempt / "command.json", command)
    env = dict(os.environ, PYTHONPATH=str(out / "code/src"), PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (attempt / "stdout.txt").open("w") as stdout, (attempt / "stderr.txt").open("w") as stderr:
        process = subprocess.run(command, cwd=out, env=env, stdout=stdout, stderr=stderr, timeout=1800)
    stdout = (attempt / "stdout.txt").read_text()
    match = re.search(r"^Test error rate:\s+([\d.]+)%", stdout, re.MULTILINE)
    if process.returncode or not match:
        raise RuntimeError(f"Cell failed: {attempt}")
    error = float(match.group(1))
    if baseline and abs(error - cell["baseline_error"]) > .000001:
        raise RuntimeError(f"Historical reproduction differs: {cell}, observed {error}")
    sys.path.insert(0, str(out / "code/scripts"))
    from shyp_size import count_rules
    result = dict(cell, label=label, err=error, rules=count_rules(str(data.with_suffix(".shyp"))),
                  attempt=str(attempt.relative_to(out)))
    save(result_path, result)
    return result


def summarize(out, rows, sizes=(250, 1200)):
    columns = ["label", "dataset", "size", "seed", "bestj", "err", "rules"]
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset"], r["size"], r["seed"])))
    if len(rows) != 10 * len(DATASETS) * len(sizes):
        return
    lines = ["# Repaired discrete-covering control", "",
             "Seven fixed test-suite datasets; ten matched training subsets per size.",
             "Error is a percentage; uncertainty follows the paper's propagated SEM.", "",
             "| Training size | Repaired covering error | Difference from boosting |",
             "|---:|---:|---:|"]
    for size in sizes:
        errors, deltas = [], []
        for dataset in DATASETS:
            group = [r for r in rows if r["size"] == size and r["dataset"] == dataset]
            assert sorted(r["seed"] for r in group) == list(range(1, 11))
            errors.append([r["err"] for r in group])
            deltas.append([r["err"] - r["baseline_error"] for r in group])
        def aggregate(groups):
            return (statistics.mean(statistics.mean(g) for g in groups),
                    math.sqrt(sum(statistics.variance(g) / len(g) for g in groups)) / len(groups))
        mean, sem = aggregate(errors)
        diff, diff_sem = aggregate(deltas)
        lines.append(f"| {size:,} | {mean:.2f} ± {sem:.2f} | {diff:+.2f} ± {diff_sem:.2f} |")
    lines += ["", "Budgets are the historical training-only CV selections, shared by all four matched methods.",
              "Boosting reproduction checks matched the original reported errors before dispatch.",
              "Sizes: " + ", ".join(f"{s:,}" for s in sizes) + ".", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sizes", default="250,1200",
                        help="comma-separated training sizes (default: the Table 1 sizes)")
    parser.add_argument("--code-from", type=Path,
                        help="frozen src/slipper snapshot to run instead of the working tree")
    args = parser.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(","))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(out, sizes, args.code_from and args.code_from.resolve())
    assert tuple(manifest["sizes"]) == sizes, "Output directory holds other sizes"
    if sizes == (250, 1200):
        checks = {("water_potability", 250, 1), ("adult", 1200, 1), ("bank_marketing", 250, 1)}
    else:
        checks = {(ds, size, 1) for size in sizes
                  for ds in ("water_potability", "adult", "bank_marketing")}
    for cell in manifest["cells"]:
        if (cell["dataset"], cell["size"], cell["seed"]) in checks:
            result = run_cell(out, cell, baseline=True)
            print("Historical reproduction matched: " + json.dumps(result), flush=True)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_cell, out, cell): cell for cell in manifest["cells"]}
        for future in as_completed(pending):
            try:
                row = future.result()
                rows.append(row)
                print(f"DONE {len(rows)}/{len(manifest['cells'])} {row['dataset']} size{row['size']} seed{row['seed']} error{row['err']}", flush=True)
            except Exception as error:
                failures.append(dict(cell=pending[future], error=str(error)))
                print("FAILED " + str(error), flush=True)
            save(out / "status.json", dict(completed=len(rows), total=len(manifest["cells"]),
                                           failures=failures))
    summarize(out, rows, sizes)
    if failures:
        raise SystemExit(f"{len(failures)} failed cells; rerun to resume")


if __name__ == "__main__":
    main()
