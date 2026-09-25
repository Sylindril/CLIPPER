#!/usr/bin/env python3
"""Learning curve for a general-purpose coding agent (Claude Code, Gemini CLI, or
Codex CLI) writing a whole rule set, run inside a sandy
(github.com/rappdw/sandy) container.

Sibling of run_baseline_curve.py and parasitic on run_learning_curve.py in the
same way: each cell's training subset comes from that script's
``subsample_train``, so an agent cell sees the byte-identical subset a
slipper/slopper cell at the same (dataset,size,seed) saw, and rows append to the
suite's ``scripts/*lc-out/results.csv`` so paired_lc.py pairs them directly.

Each cell (dataset, size, seed):
  1. build a fresh workspace ~/sandy-runs/cells/<hash>/ holding only train.csv,
     schema.json, and TASK.md (the test split never enters it; the hashed name
     keeps the dataset name out of the in-container path);
  2. run ``sandy --new -p "Read TASK.md and do the task."`` there, headless,
     with strict egress, no GitHub token, and no web tools;
  3. score the agent's rules.py on the full test split, executing
     ``classify`` in a smolagents LocalPythonExecutor (agent code never runs
     unsandboxed on the host).
Cell records (rules.py, agent JSON result, sandy log, meta.json) land under
<out>/cells/size.<NNN>/<label>__<ds>__<seed>/.  A cell with no usable rules.py
is logged and not written, so a rerun retries it.

results.csv columns: ``bestj`` is 0 (no round budget), ``rules`` is the number
of return statements in rules.py minus the default, ``llm_calls`` is agent turns
(claude), API requests (gemini), or shell commands + messages (codex), and
``latency_s`` is wall-clock for the whole sandy run.  Gemini and Codex CLIs
report no cost, so they are priced from token counts via litellm.  Only claude
has a dollar cap (--budget-usd); codex has no turn cap either, so --timeout is
its only bound.

Needs Docker, sandy on PATH, and ANTHROPIC_API_KEY / GEMINI_API_KEY /
OPENAI_API_KEY in ~/.sandy/.secrets.  Under Colima the VM must share both $HOME
(workspaces) and the macOS temp dir (sandy mounts codex's auth.json from it):
    colima start --mount "$HOME:w" --mount "$(getconf DARWIN_USER_TEMP_DIR):r"

    uv run python scripts/run_agent_curve.py --suite test --label cc_sonnet5 \\
        --agent claude --model claude-sonnet-5 --sizes 250,2000 --seeds 5

    uv run python scripts/run_agent_curve.py --suite test --label gcli_flashlite \\
        --agent gemini --model gemini-3.1-flash-lite --sizes 250,2000 --seeds 5

    uv run python scripts/run_agent_curve.py --suite test --label codex_luna \\
        --agent codex --model gpt-5.6-luna --sizes 250,2000 --seeds 5

    uv run python scripts/run_agent_curve.py --suite test --label cc_sonnet5 \\
        --sizes 250,2000 --seeds 5 --tabulate
"""
import argparse
import ast
import csv
import fcntl
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_learning_curve as rlc
import suite as suite_mod
from run_learning_curve import load_source, parse_datasets, subsample_train
from suite import add_suite_arg, write_run_config

RUNS_ROOT = os.path.expanduser("~/sandy-runs/cells")
PROMPT = "Read TASK.md and do the task."
_lock = threading.Lock()

TASK = """\
# Task: write an interpretable rule-based classifier

`train.csv` holds {n} labeled training examples (column `label` is the class,
one of {classes}).  `schema.json` gives each attribute's name, kind, and
(for symbolic attributes) allowed values.  Empty cells are missing values.

Write `rules.py` in this directory defining

    def classify(x: dict) -> str

where `x` maps attribute names to values and the return value is one of
{classes}.  Continuous attributes arrive as floats, symbolic ones as strings,
and missing values as `None`.

Requirements:
- `classify` must be a readable rule set: a short sequence of if/then rules
  over the attributes, ending in a default class.  No imports in `rules.py`,
  no lookup tables of training examples, no learned weight vectors.
- You will be scored on a held-out test set you do not have, drawn from the
  same distribution as `train.csv`.  The score is plain classification
  error: the fraction of test examples where `classify` returns the wrong
  class.  Every mistake costs the same, whichever class it is; there is no
  credit for recall, precision, or balanced accuracy.
- You may write and run Python scripts in this directory to explore the data
  and evaluate candidates on `train.csv`.

When done, `rules.py` must exist.  Reply with a one-paragraph summary.
"""

# Appended by --forbid-learners.  Codex-family agents otherwise pip-install
# scikit-learn, fit a cross-validated CART and transcribe its splits (see
# the harness audits); this makes the rule-finding process the condition
# rather than leaving it to the agent's discretion.  A prompt is not a sandbox:
# the per-cell `pypi.org` egress line in sandy.log still has to be checked.
FORBID_LEARNERS = """
You must find the rules by examining the data and reasoning about what rules
are likely to be helpful.  You may use tools like pandas to compute statistics
(e.g., correlation or PMI between an attribute and a class) or test a rule that
you have constructed, but you should not run any existing learning algorithm
(e.g., fitting a decision tree with sklearn), nor should you implement a Python
helper script that implements an existing learning algorithm.  Every helper
script should be annotated with a comment of the form "Helper script purpose:
XXXX" where XXX is a short but accurate explanation, so the scripts function
can be audited.  Submissions that do not follow these instructions will be
discarded.
"""


# --- workspace --------------------------------------------------------------

def build_workspace(ws, spec, max_turns, forbid_learners=False):
    """Fresh ws with train.csv / schema.json / TASK.md (+ gemini settings)."""
    shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(ws)
    names = [a["name"] for a in spec["attributes"]]
    with open(os.path.join(ws, "train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["label"])
        for ex in spec["train"]:
            w.writerow(["" if v is None else v for v in ex["values"]] + [ex["label"]])
    with open(os.path.join(ws, "schema.json"), "w") as f:
        json.dump({"classes": spec["classes"], "attributes": spec["attributes"]}, f, indent=1)
    with open(os.path.join(ws, "TASK.md"), "w") as f:
        f.write(TASK.format(n=len(spec["train"]),
                            classes=", ".join(f"`{c}`" for c in spec["classes"]))
                + (FORBID_LEARNERS if forbid_learners else ""))
    # Gemini CLI has no turn/budget flag; a project settings file caps turns and
    # drops its web tools.  Read at startup, so a mid-run edit can't lift it.
    os.makedirs(os.path.join(ws, ".gemini"))
    with open(os.path.join(ws, ".gemini", "settings.json"), "w") as f:
        json.dump({"model": {"maxSessionTurns": max_turns},
                   "tools": {"exclude": ["google_web_search", "web_fetch"]}}, f)


def nogh_dir():
    """Dir with a `gh` that always fails, prepended to PATH: sandy forwards the
    host's gh token into the container and has no switch to stop it."""
    d = os.path.join(os.path.dirname(RUNS_ROOT), "nogh")
    os.makedirs(d, exist_ok=True)
    stub = os.path.join(d, "gh")
    if not os.path.exists(stub):
        with open(stub, "w") as f:
            f.write("#!/bin/sh\nexit 1\n")
        os.chmod(stub, 0o755)
    return d


# --- agent run --------------------------------------------------------------

def agent_invocation(args):
    env = dict(os.environ, PATH=nogh_dir() + os.pathsep + os.environ["PATH"],
               SANDY_AGENT=args.agent, SANDY_EGRESS_STRICT="1",
               SANDY_EGRESS_LOG="summary", SANDY_CPUS=str(args.cpus),
               SANDY_MEM=args.mem)
    cmd = ["sandy", "--new", "-p", PROMPT]
    if args.agent == "codex":
        # Codex CLI has no budget/turn cap (only --timeout bounds a cell).  Its
        # built-in web search is server-side, so egress rules can't block it.
        # Long-form --config: sandy drops `-c` as a --continue alias.
        env.update(SANDY_CODEX_AUTH="api_key", CODEX_MODEL=args.model)
        cmd += ["--json", "--config", 'web_search="disabled"']
        if args.effort:
            cmd += ["--config", f'model_reasoning_effort="{args.effort}"']
        return cmd, env
    cmd += ["--output-format", "json"]
    if args.agent == "claude":
        env.update(SANDY_CLAUDE_AUTH="api_key", SANDY_MODEL=args.model)
        if args.effort:
            env["SANDY_EFFORT"] = args.effort
        # --setting-sources: ignore the host ~/.claude/settings.json sandy seeds
        # into the sandbox (its permission `ask` rules block Write headless).
        cmd += ["--setting-sources", "project,local",
                "--max-budget-usd", str(args.budget_usd),
                "--max-turns", str(args.max_turns),
                "--disallowedTools", "WebSearch,WebFetch"]
    else:
        env["GEMINI_MODEL"] = args.model
    return cmd, env


def kill_cell(proc, base, grace=30):
    """SIGTERM sandy's process group, SIGKILL after `grace`s, then force-remove
    the cell's containers (named after the workspace basename)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            pass
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={base}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)


_live = {}      # pid -> (proc, workspace basename), for the signal handler


def run_sandy(ws, cmd, env, timeout, launch_marker="[sandy] Launching"):
    """Run sandy in ws; returns (rc, stdout, stderr, wall_s).

    Concurrent launches race on sandy's shared ~/.sandy/Dockerfile*.new files,
    so a cross-process lock is held from start until sandy prints
    `launch_marker` (its setup is done and the agent is starting); the agent
    runs unlocked.  On timeout the cell is killed via kill_cell."""
    base = os.path.basename(ws)
    # Clear any sandbox left by an earlier attempt at this cell (pip cache,
    # transcripts), so a retry starts from the same state as a first run.
    for sb in glob.glob(os.path.expanduser(f"~/.sandy/sandboxes/{base}-*")):
        if os.path.isdir(sb):
            subprocess.run(["sandy", "--remove-sandbox", "--workspace", ws, "--yes"],
                           capture_output=True, env=env)
    os.makedirs(os.path.dirname(RUNS_ROOT), exist_ok=True)
    timed_out = threading.Event()
    with open(os.path.join(os.path.dirname(RUNS_ROOT), ".launch.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", start_new_session=True)
        _live[proc.pid] = (proc, base)
        err = []
        drain = threading.Thread(target=lambda: err.append(proc.stderr.read()))
        drain.start()
        timer = threading.Timer(timeout, lambda: (timed_out.set(), kill_cell(proc, base)))
        timer.start()
        head = []
        for line in proc.stdout:
            head.append(line)
            if line.startswith(launch_marker):
                break
    out = "".join(head) + proc.stdout.read()
    proc.wait()
    timer.cancel()
    drain.join()
    _live.pop(proc.pid, None)
    err = "".join(err)
    if timed_out.is_set():
        err += f"\n[run_agent_curve] timed out after {timeout}s\n"
    return proc.returncode, out, err, time.time() - t0


def parse_result(stdout):
    """The agent's JSON result, from stdout shared with sandy's [sandy] log
    lines and a terminal escape sequence: the one object claude/gemini print,
    or {"events": [...]} for codex's JSONL event stream; {} if absent."""
    body = "\n".join(l for l in stdout.splitlines() if not l.startswith("[sandy]"))
    dec, objs, i = json.JSONDecoder(), [], body.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(body, i)
        except ValueError:
            break
        objs.append(obj)
        i = body.find("{", end)
    if not objs:
        return {}
    return objs[0] if len(objs) == 1 else {"events": objs}


def result_models(agent, res):
    """Models the agent's own result says it used.  Gemini CLI silently falls
    back to its default for a model it doesn't know (0.60.0 ran every
    gemini-3.6/3.7/3.8 request as gemini-3.5-flash), and only its stats reveal
    it.  Empty for codex, whose event stream doesn't name the model."""
    if agent == "gemini":
        return sorted(res.get("stats", {}).get("models", {}))
    if agent == "claude":
        return sorted(res.get("modelUsage", {}))
    return []


def side_info(agent, res, wall_s, model=None):
    """results.csv side columns (rlc._SIDE_COLS) from the agent's JSON result.
    `model` prices codex (its events don't name the model)."""
    if agent == "codex":
        import litellm
        ev = res.get("events", [])
        usage = [e.get("usage", {}) for e in ev if e.get("type") == "turn.completed"]
        tok = {k: sum(u.get(k, 0) for u in usage)
               for k in ("input_tokens", "cached_input_tokens", "output_tokens")}
        try:
            cost = round(sum(litellm.cost_per_token(
                model=model, prompt_tokens=tok["input_tokens"],
                completion_tokens=tok["output_tokens"],
                cache_read_input_tokens=tok["cached_input_tokens"])), 6)
        except Exception:
            cost = ""
        steps = sum(e.get("type") == "item.completed" and
                    e.get("item", {}).get("type") in ("command_execution", "agent_message")
                    for e in ev)
        return dict(llm_calls=steps, latency_s=round(wall_s, 1),
                    in_tok=tok["input_tokens"], out_tok=tok["output_tokens"], cost_usd=cost)
    if agent == "claude":
        u = res.get("usage", {})
        in_tok = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                           "cache_read_input_tokens"))
        return dict(llm_calls=res.get("num_turns", ""), latency_s=round(wall_s, 1),
                    in_tok=in_tok, out_tok=u.get("output_tokens", ""),
                    cost_usd=res.get("total_cost_usd", ""))
    import litellm
    calls = in_tok = out_tok = 0
    cost = 0.0
    for model, m in res.get("stats", {}).get("models", {}).items():
        t = m.get("tokens", {})
        prompt, out = t.get("prompt", 0), t.get("candidates", 0) + t.get("thoughts", 0)
        calls += m.get("api", {}).get("totalRequests", 0)
        in_tok, out_tok = in_tok + prompt, out_tok + out
        try:
            pc, cc = litellm.cost_per_token(model="gemini/" + model, prompt_tokens=prompt,
                                            completion_tokens=out,
                                            cache_read_input_tokens=t.get("cached", 0))
            cost += pc + cc
        except Exception:
            cost = None
    return dict(llm_calls=calls, latency_s=round(wall_s, 1), in_tok=in_tok,
                out_tok=out_tok, cost_usd="" if cost is None else round(cost, 6))


# --- scoring ----------------------------------------------------------------

def score(source, spec):
    """(test err %, #rules, #imports) for rules.py source; classify runs in a
    smolagents sandbox, and an example it raises on counts as an error.  Rules
    = return statements minus the default (if-expressions are mostly None
    guards, so counting branches overstates the rule set)."""
    from smolagents.local_python_executor import LocalPythonExecutor
    tree = ast.parse(source)
    n_rules = max(sum(isinstance(n, ast.Return) for n in ast.walk(tree)) - 1, 0)
    n_imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    ex.send_tools({})
    ex(source)
    names = [a["name"] for a in spec["attributes"]]
    errs = 0
    for row in spec["test"]:
        ex.send_variables({"__instance": dict(zip(names, row["values"]))})
        try:
            pred = ex("classify(__instance)").output
        except Exception:
            pred = None
        errs += pred != row["label"]
    return round(100.0 * errs / len(spec["test"]), 2), n_rules, n_imports


# --- cells ------------------------------------------------------------------

def run_cell(griddir, label, args, ds, size, seed):
    """Returns (err, rules, side), or (None, None, {}) on skip/failure."""
    spec = subsample_train(load_source(ds), ds, size, seed)
    if spec is None:
        return None, None, {}
    celldir = os.path.join(griddir, f"size.{size}", f"{label}__{ds}__{seed}")
    os.makedirs(celldir, exist_ok=True)
    key = hashlib.md5(f"{rlc.SUITE.name}|{label}|{ds}|{size}|{seed}".encode()).hexdigest()[:12]
    ws = os.path.join(RUNS_ROOT, key)
    build_workspace(ws, spec, args.max_turns, args.forbid_learners)
    cmd, env = agent_invocation(args)
    rc, out, err, wall = run_sandy(ws, cmd, env, args.timeout)
    res = parse_result(out)
    with open(os.path.join(celldir, "sandy.log"), "w") as f:
        f.write(out + "\n--- stderr ---\n" + err)
    with open(os.path.join(celldir, "agent.json"), "w") as f:
        json.dump(res, f, indent=1)
    meta = dict(workspace=ws, rc=rc, wall_s=round(wall, 1), cmd=cmd)
    tag = f"{ds} size={size} seed={seed}"
    # Warn-only: a CLI that substitutes its default for an unknown --model
    # would otherwise mislabel the whole condition (see result_models).
    used = result_models(args.agent, res)
    if used:
        meta["models_used"] = used
        if args.model not in used:
            sys.stderr.write(f"[MODEL] {tag}: asked for {args.model}, agent reports "
                             f"{', '.join(used)}\n")
    rules_py = os.path.join(ws, "rules.py")
    try:
        if not os.path.exists(rules_py):
            raise RuntimeError("no rules.py")
        source = open(rules_py).read()
        shutil.copy(rules_py, celldir)
        e, n_rules, n_imports = score(source, spec)
    except Exception as exc:
        meta["failure"] = repr(exc)
        sys.stderr.write(f"[FAIL] {tag} rc={rc}: {exc!r}\n")
        e = None
    else:
        meta.update(err=e, rules=n_rules, imports=n_imports)
    with open(os.path.join(celldir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if e is None:
        return None, None, {}
    return e, n_rules, side_info(args.agent, res, wall, args.model)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="condition name (results.csv key)")
    add_suite_arg(ap)
    ap.add_argument("--agent", choices=["claude", "gemini", "codex"], default="claude")
    ap.add_argument("--model", default=None,
                    help="agent model (default: claude-sonnet-5 / gemini-3.1-flash-lite)")
    ap.add_argument("--forbid-learners", action="store_true",
                    help="append the no-external-learner instruction to TASK.md "
                         "(see FORBID_LEARNERS); run it as its own label")
    ap.add_argument("--effort", default="high",
                    help="claude/codex reasoning effort ('' = CLI default)")
    ap.add_argument("--budget-usd", type=float, default=2.0, help="claude --max-budget-usd")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1800, help="wall-clock seconds per cell")
    ap.add_argument("--cpus", type=int, default=2, help="SANDY_CPUS per container")
    ap.add_argument("--mem", default="4g", help="SANDY_MEM per container")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated training sizes (default: the suite's)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="output dir (default: the suite's scripts/*lc-out)")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--tabulate", action="store_true")
    args = ap.parse_args()
    args.model = args.model or {"claude": "claude-sonnet-5",
                                "gemini": "gemini-3.1-flash-lite",
                                "codex": "gpt-5.6-luna"}[args.agent]

    rlc.SUITE = suite_mod.get(args.suite)
    out = args.out or rlc.SUITE.out("lc")
    sizes = ([int(s) for s in args.sizes.split(",") if s.strip()]
             if args.sizes else rlc.SUITE.sizes)
    datasets = parse_datasets(args.datasets, args.exclude)
    if args.tabulate:
        rlc.tabulate(out, args.label, sizes, args.seeds, datasets)
        return

    griddir = os.path.join(out, "cells")
    results_path = os.path.join(out, "results.csv")
    os.makedirs(griddir, exist_ok=True)
    write_run_config(out, args.label, {
        "label": args.label, "suite": rlc.SUITE.name, "sizes": sizes,
        "seeds": "1..%d" % args.seeds, "workers": args.workers, "datasets": datasets,
        "agent": args.agent, "model": args.model,
        "effort": args.effort if args.agent in ("claude", "codex") else None,
        "budget_usd": args.budget_usd if args.agent == "claude" else None,
        "max_turns": args.max_turns if args.agent != "codex" else None,
        "timeout_s": args.timeout,
        "container": {"cpus": args.cpus, "mem": args.mem},
        "command": " ".join(agent_invocation(args)[0]),
        "sandy_version": subprocess.run(["sandy", "--print-version"], capture_output=True,
                                        text=True).stdout.strip(),
        "forbid_learners": args.forbid_learners,
        "task_md": TASK + (FORBID_LEARNERS if args.forbid_learners else ""),
    })
    done = rlc.load_done(results_path, args.label)
    trainlen = {ds: len(load_source(ds).get("train") or []) for ds in datasets}
    tasks = [(ds, sz, s) for ds in datasets for sz in sizes for s in range(1, args.seeds + 1)
             if sz <= trainlen[ds] and (ds, sz, s) not in done]
    print(f"agent suite '{args.label}' ({args.agent} {args.model}): {len(done)} done, "
          f"{len(tasks)} to run ({args.workers}-way parallel)", flush=True)
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(["label", "dataset", "size", "seed", "bestj", "err",
                                    "rules"] + rlc._SIDE_COLS)

    # sandy runs in its own process group, so killing this script alone would
    # orphan it and its containers; take them down too.
    def stop(signum, frame):
        for proc, base in list(_live.values()):
            kill_cell(proc, base, grace=10)
        os._exit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, griddir, args.label, args, ds, sz, s): (ds, sz, s)
                for ds, sz, s in tasks}
        for fut in as_completed(futs):
            ds, sz, s = futs[fut]
            err, rules, side = fut.result()
            n += 1
            if err is None:
                continue
            with _lock, open(results_path, "a", newline="") as f:
                csv.writer(f).writerow([args.label, ds, sz, s, 0, err, rules]
                                       + [side.get(c, "") for c in rlc._SIDE_COLS])
            print(f"  [{n}/{len(tasks)}] {ds:22} size{sz:<4} seed{s:2} rules{rules:<3} "
                  f"${side.get('cost_usd') or '?'} -> {err:5.2f}", flush=True)
    print("suite complete", flush=True)
    rlc.tabulate(out, args.label, sizes, args.seeds, datasets)


if __name__ == "__main__":
    main()
